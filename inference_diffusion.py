# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import numpy as np
import os
import click
import re
import json
import tempfile
import torch
import dnnlib
from PIL import Image
from clip import clip
from training import training_loop_diffusion_3d
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from training.inference_utils import save_visualization, save_visualization_with_cond, save_visualization_for_interpolation, \
    save_textured_mesh_for_inference, save_geo_for_inference
from denoising_diffusion_pytorch import Unet1D_cond, GaussianDiffusion1D_cond
from ema_pytorch import EMA
    
# ----------------------------------------------------------------------------
def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    inference_diff(**c)


# ----------------------------------------------------------------------------
def inference_diff(
        run_dir='.',  # Output directory.
        training_set_kwargs={},  # Options for training set.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus[.
        inference_vis=False,
        inference_to_generate_textured_mesh=False,
        resume_pretrain=None,
        backbone_pretrain=None,
        inference_save_interpolation=False,
        inference_compute_fid=False,
        inference_generate_geo=False,
        diff_ch=4,
        ema_update_every=10,
        ema_decay=0.995,
        image_cond=None,
        text_cond=None,
        **dummy_kawargs
):
    from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import filtered_lrelu
    upfirdn2d._init()
    bias_act._init()
    filtered_lrelu._init()

    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = True  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = True  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.


    common_kwargs = dict(
        c_dim=0, img_resolution=training_set_kwargs['resolution'] if 'resolution' in training_set_kwargs else 1024, img_channels=3)

    G_kwargs['device'] = device

    if num_gpus > 1:
        torch.distributed.barrier()
    
    G_ema = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).eval().requires_grad_(False).to(device)  # subclass of torch.nn.Module

    clip_model, preprocess = clip.load('ViT-B/16', device=device)

    geo_Unet = Unet1D_cond(dim=diff_ch, dim_mults=(1, 2, 4, 8), channels=1, cond_ch=1, resnet_block_groups=2).to(device)
    geo_diffusion_model = GaussianDiffusion1D_cond(geo_Unet, seq_length=G_kwargs.w_dim, timesteps=1000, objective='pred_v').to(device)
    tex_Unet = Unet1D_cond(dim=diff_ch, dim_mults=(1, 2, 4, 8), channels=2, cond_ch=2, resnet_block_groups=2).to(device)
    tex_diffusion_model = GaussianDiffusion1D_cond(tex_Unet, seq_length=G_kwargs.w_dim, timesteps=1000, objective='pred_v').to(device)

    if resume_pretrain is not None and (rank == 0):
        print('==> resume diffusion model from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        geo_diffusion_model.load_state_dict(model_state_dict['geo_diff'], strict=True)
        tex_diffusion_model.load_state_dict(model_state_dict['tex_diff'], strict=True)

    if backbone_pretrain is not None and (rank == 0):
        print('==> resume backbone from pretrained path %s' % (backbone_pretrain))
        model_state_dict = torch.load(backbone_pretrain, map_location=device)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)

    grid_size = (5, 5)
    n_shape = grid_size[0] * grid_size[1]
    grid_z = torch.randn([n_shape, G_kwargs.w_dim], device=device).unsqueeze(1).split(1)  # random code for geometry
    grid_tex_z = torch.randn([n_shape, G_kwargs.w_dim], device=device).unsqueeze(1).repeat(1, tex_diffusion_model.model.cond_ch, 1).split(1)  # random code for texture
    grid_c = torch.ones(n_shape, device=device).split(1)
    if image_cond is not None:
        print('==> use image as condition')
        cond = clip_model.encode_image(preprocess(Image.open(image_cond)).unsqueeze(0).to(device)).repeat(n_shape, 1).unsqueeze(1).split(1)

    elif text_cond is not None:
        print('==> use text as condition')
        cond = clip_model.encode_text(clip.tokenize(text_cond).to(device)).repeat(n_shape, 1).unsqueeze(1).split(1)
    else:
        print('==> no condition')
        cond = torch.zeros((n_shape, clip_model.visual.output_dim), device=device).unsqueeze(1).split(1)
    print('==> generate ')
    save_visualization_with_cond(
        G_ema, geo_diffusion_model, tex_diffusion_model, cond, grid_z, grid_c, run_dir, 0, grid_size, 0,
        save_all=False,
        grid_tex_z=grid_tex_z
    )

    # if inference_to_generate_textured_mesh:
    #     print('==> generate inference 3d shapes with texture')
    #     save_textured_mesh_for_inference(
    #         G_ema, grid_z, grid_c, run_dir, save_mesh_dir='texture_mesh_for_inference',
    #         c_to_compute_w_avg=None, grid_tex_z=grid_tex_z)

    # if inference_save_interpolation:
    #     print('==> generate interpolation results')
    #     save_visualization_for_interpolation(G_ema, save_dir=os.path.join(run_dir, 'interpolation'))

    # if inference_compute_fid:
    #     print('==> compute FID scores for generation')
    #     for metric in metrics:
    #         training_set_kwargs = clean_training_set_kwargs_for_metrics(training_set_kwargs)
    #         training_set_kwargs['split'] = 'test'
    #         result_dict = metric_main.calc_metric(
    #             metric=metric, G=G_ema,
    #             dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank,
    #             device=device)
    #         metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=resume_pretrain)

    # if inference_generate_geo:
    #     print('==> generate 7500 shapes for evaluation')
    #     save_geo_for_inference(G_ema, run_dir)







# ----------------------------------------------------------------------------
def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    # if c.inference_vis:
    #     c.run_dir = os.path.join(outdir, 'inference')
   
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.train_num_steps} steps')
    # print(f'Dataset path:        {c.training_set_kwargs.path}')
    # print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    # print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    # print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    # print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    if not os.path.exists(c.run_dir):
        os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn', force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)


# ----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


# ----------------------------------------------------------------------------

@click.command()
# Required from StyleGAN2.
@click.option('--outdir', help='Where to save the results', metavar='DIR', required=True)
@click.option('--cfg', help='Base configuration', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), default='stylegan2')
@click.option('--gpus', help='Number of GPUs to use', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--batch', help='Total batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--gamma', help='R1 regularization weight', metavar='FLOAT', type=click.FloatRange(min=0), required=True)
# My custom configs
### Configs for inference
@click.option('--resume_pretrain', help='Resume from given network pickle', metavar='[PATH|URL]', type=str)
@click.option('--backbone_pretrain', help='Resume from given network pickle', metavar='[PATH|URL]', type=str)
@click.option('--inference_vis', help='whther we run infernce', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--inference_to_generate_textured_mesh', help='inference to generate textured meshes', metavar='BOOL', type=bool, default=False, show_default=False)
@click.option('--inference_save_interpolation', help='inference to generate interpolation results', metavar='BOOL', type=bool, default=False, show_default=False)
@click.option('--inference_compute_fid', help='inference to generate interpolation results', metavar='BOOL', type=bool, default=False, show_default=False)
@click.option('--inference_generate_geo', help='inference to generate geometry points', metavar='BOOL', type=bool, default=False, show_default=False)
### Configs for dataset

@click.option('--data', help='Path to the Training data Images', metavar='[DIR]', type=str, default='./tmp')
@click.option('--camera_path', help='Path to the camera root', metavar='[DIR]', type=str, default='./tmp')
@click.option('--img_res', help='The resolution of image', metavar='INT', type=click.IntRange(min=1), default=1024)
@click.option('--data_camera_mode', help='The type of dataset we are using', type=str, default='shapenet_car', show_default=True)
@click.option('--use_shapenet_split', help='whether use the training split or all the data for training', metavar='BOOL', type=bool, default=False, show_default=False)
### Configs for 3D generator##########
@click.option('--use_style_mixing', help='whether use style mixing for generation during inference', metavar='BOOL', type=bool, default=True, show_default=False)
@click.option('--one_3d_generator', help='whether we detach the gradient for empty object', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--dmtet_scale', help='Scale for the dimention of dmtet', metavar='FLOAT', type=click.FloatRange(min=0, max=10.0), default=1.0, show_default=True)
@click.option('--n_implicit_layer', help='Number of Implicit FC layer for XYZPlaneTex model', metavar='INT', type=click.IntRange(min=1), default=1)
@click.option('--feat_channel', help='Feature channel for TORGB layer', metavar='INT', type=click.IntRange(min=0), default=16)
@click.option('--mlp_latent_channel', help='mlp_latent_channel for XYZPlaneTex network', metavar='INT', type=click.IntRange(min=8), default=32)
@click.option('--deformation_multiplier', help='Multiplier for the predicted deformation', metavar='FLOAT', type=click.FloatRange(min=1.0), default=1.0, required=False)
@click.option('--tri_plane_resolution', help='The resolution for tri plane', metavar='INT', type=click.IntRange(min=1), default=256)
@click.option('--n_views', help='number of views when training generator', metavar='INT', type=click.IntRange(min=1), default=1)
@click.option('--use_tri_plane', help='Whether use tri plane representation', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--tet_res', help='Resolution for teteahedron', metavar='INT', type=click.IntRange(min=1), default=90)
@click.option('--latent_dim', help='Dimention for latent code', metavar='INT', type=click.IntRange(min=1), default=512)
@click.option('--geometry_type', help='The type of geometry generator', type=str, default='conv3d', show_default=True)
@click.option('--render_type', help='Type of renderer we used', metavar='STR', type=click.Choice(['neural_render', 'spherical_gaussian']), default='neural_render', show_default=True)
@click.option('--use_opengl', help='Use OpenGL or not', metavar='BOOL', type=bool, default=True, show_default=True)
### Configs for training loss and discriminator#
@click.option('--d_architecture', help='The architecture for discriminator', metavar='STR', type=str, default='skip', show_default=True)
@click.option('--use_pl_length', help='whether we apply path length regularization', metavar='BOOL', type=bool, default=False, show_default=False)  # We didn't use path lenth regularzation to avoid nan error
@click.option('--gamma_mask', help='R1 regularization weight for mask', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, required=False)
@click.option('--add_camera_cond', help='Whether we add camera as condition for discriminator', metavar='BOOL', type=bool, default=True, show_default=True)
## Miscs
# Optional features.
@click.option('--cond', help='Train conditional model', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--freezed', help='Freeze first layers of D', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
# Misc hyperparameters.
@click.option('--cbase', help='Capacity multiplier', metavar='INT', type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax', help='Max. feature maps', metavar='INT', type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--lr', help='Diffusion model learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=1e-4, show_default=True)
@click.option('--map-depth', help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group', help='Minibatch std group size', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
# Misc settings.
@click.option('--desc', help='String to include in result dir name', metavar='STR', type=str)
@click.option('--num_steps', help='Total training duration', metavar='INT', type=click.IntRange(min=1), default=100000, show_default=True)
@click.option('--snap', help='How often to save snapshots', metavar='TICKS', type=click.IntRange(min=1), default=50, show_default=True)  ###
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32', help='Disable mixed-precision', metavar='BOOL', type=bool, default=True, show_default=True)  # Let's use fp32 all the case without clamping
@click.option('--nobench', help='Disable cuDNN benchmarking', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--workers', help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=0), default=3, show_default=True)
@click.option('--ema_update_every', help='How often update ema model',  metavar='INT', type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--ema_decay', help='The decay to update ema',  metavar='FLOAT', type=click.FloatRange(min=0), default=0.995, show_default=True)
@click.option('--diff_ch', help='first layer channel num for diffusion model',  metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# inference options
@click.option('--image_cond', help='The path of image for conditioning',  metavar='STR', type=str, default=None, show_default=True)
@click.option('--text_cond', help='The text for conditioning',  metavar='STR', type=str, default=None, show_default=True)

def main(**kwargs):
    # Initialize config.
    print('==> start')
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    c = dnnlib.EasyDict()  # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(
        class_name=None, z_dim=opts.latent_dim, w_dim=opts.latent_dim, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(
        class_name='training.networks_get3d.Discriminator', block_kwargs=dnnlib.EasyDict(),
        mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.diffusion_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0.9, 0.99], eps=1e-8)
    
    # Hyperparameters & settings.p
    c.G_kwargs.one_3d_generator = opts.one_3d_generator
    c.G_kwargs.n_implicit_layer = opts.n_implicit_layer
    c.G_kwargs.deformation_multiplier = opts.deformation_multiplier
    c.resume_pretrain = opts.resume_pretrain
    c.backbone_pretrain = opts.backbone_pretrain
    c.G_kwargs.use_style_mixing = opts.use_style_mixing
    c.G_kwargs.dmtet_scale = opts.dmtet_scale
    c.G_kwargs.feat_channel = opts.feat_channel
    c.G_kwargs.mlp_latent_channel = opts.mlp_latent_channel
    c.G_kwargs.tri_plane_resolution = opts.tri_plane_resolution
    c.G_kwargs.n_views = opts.n_views

    c.G_kwargs.render_type = opts.render_type
    c.G_kwargs.use_tri_plane = opts.use_tri_plane
    c.D_kwargs.data_camera_mode = opts.data_camera_mode
    c.D_kwargs.add_camera_cond = opts.add_camera_cond

    c.G_kwargs.use_opengl = opts.use_opengl

    c.G_kwargs.tet_res = opts.tet_res

    c.G_kwargs.geometry_type = opts.geometry_type
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.G_kwargs.data_camera_mode = opts.data_camera_mode
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax

    c.G_kwargs.mapping_kwargs.num_layers = 8

    c.D_kwargs.architecture = opts.d_architecture
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group

    c.diffusion_opt_kwargs.lr = opts.lr
    c.diff_ch = opts.diff_ch

    c.train_num_steps = opts.num_steps
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.network_snapshot_ticks = 5000
    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')

    # Base configuration.
    c.ema_update_every = opts.ema_update_every
    c.ema_decay = opts.ema_decay
    c.G_kwargs.class_name = 'training.networks_get3d.GeneratorDMTETMesh'
    # c.loss_kwargs.style_mixing_prob = 0.9  # Enable style mixing regularization.
    # c.loss_kwargs.pl_weight = 0.0  # Enable path length regularization.
    c.G_kwargs.fused_modconv_default = 'inference_only'  # Speed up training by using regular convolutions instead of grouped convolutions.
    c.image_cond = opts.image_cond
    c.text_cond = opts.text_cond
    c.random_seed = opts.seed

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-diffusion-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    # Launch.
    print('==> launch training')
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)


# ----------------------------------------------------------------------------
#
if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
