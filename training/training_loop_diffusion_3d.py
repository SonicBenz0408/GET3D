# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Main training loop."""

import os
import copy
import json
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from metrics import metric_main
import nvdiffrast.torch as dr
import time
from tqdm.auto import tqdm
from training.inference_utils import save_image_grid, save_visualization
from denoising_diffusion_pytorch import Unet1D_cond, GaussianDiffusion1D_cond
from accelerate import Accelerator
from clip import clip
from ema_pytorch import EMA
from torchvision import transforms

# ----------------------------------------------------------------------------
# Function to save the real image for discriminator training
def setup_snapshot_image_grid(img_res, inference=False):
    grid_w = 7
    grid_h = 4
    min_w = 8 if inference else grid_w
    min_h = 9 if inference else grid_h
    gw = np.clip(1024 // img_res, min_w, 32)
    gh = np.clip(1024 // img_res, min_h, 32)

    return (gw, gh)


def clean_training_set_kwargs_for_metrics(training_set_kwargs):
    # We use this function to remove or change custom kwargs for dataset
    # we used these kwargs to comput md5 for the cache file of FID
    if 'add_camera_cond' in training_set_kwargs:
        training_set_kwargs['add_camera_cond'] = True
    return training_set_kwargs


# ----------------------------------------------------------------------------
def training_loop(
        run_dir='.',  # Output directory.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        diffusion_opt_kwargs={}, # Options for diffusion model optimizer
        img_res=1024,
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus].
        batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        train_num_steps=100000,  # Total length of the training.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
        network_snapshot_ticks=5000,  # How often to save network snapshots? None = disable.
        # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
        inference_vis=False,  # Whether running inference or not.
        resume_pretrain=None,
        diff_ch=4,
        amp=False,
        fp32=True,
        split_batches=True,
        ema_update_every=10,
        ema_decay=0.995,
):
    from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import filtered_lrelu
    upfirdn2d._init()
    bias_act._init()
    filtered_lrelu._init()
    if num_gpus > 1:
        torch.distributed.barrier()
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = True  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = True  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    if rank == 0:
        print('Constructing networks...')

    # Constructing networks
    common_kwargs = dict(
        c_dim=0, img_resolution=img_res, img_channels=3)
    G_kwargs['device'] = device
    D_kwargs['device'] = device

    if num_gpus > 1:
        torch.distributed.barrier()
    G_ema = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).eval().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).eval().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    clip_model, preprocess = clip.load('ViT-B/16', device=device)
    
    if resume_pretrain is not None and (rank == 0):
        # We're not reusing the loading function from stylegan3 codebase,
        # since we have some variables that are not picklable.
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        D.load_state_dict(model_state_dict['D'], strict=True)
    
    # Freeze G & D
    G_ema.requires_grad_(False)
    D.requires_grad_(False)
    clip_model.requires_grad_(False)

    tensor_to_img = transforms.ToPILImage()

    # create two diffusion model
    geo_Unet = Unet1D_cond(dim=diff_ch, dim_mults=(1, 2, 4, 8), channels=1, cond_ch=1, resnet_block_groups=2).to(device)
    geo_diffusion_model = GaussianDiffusion1D_cond(geo_Unet, seq_length=G_kwargs.w_dim, timesteps=1000, objective='pred_v').to(device)
    tex_Unet = Unet1D_cond(dim=diff_ch, dim_mults=(1, 2, 4, 8), channels=2, cond_ch=2, resnet_block_groups=2).to(device)
    tex_diffusion_model = GaussianDiffusion1D_cond(tex_Unet, seq_length=G_kwargs.w_dim, timesteps=1000, objective='pred_v').to(device)

    if rank == 0:
        print('Setting up augmentation...')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')

    if rank == 0:
        print('Setting up training phases...')

    # Constructing loss functins and optimizer
    geo_diffusion_opt = dnnlib.util.construct_class_by_name(
        params=geo_diffusion_model.parameters(), **diffusion_opt_kwargs)  # subclass of torch.optim.Optimizer
    tex_diffusion_opt = dnnlib.util.construct_class_by_name(
        params=tex_diffusion_model.parameters(), **diffusion_opt_kwargs)  # subclass of torch.optim.Optimizer

    # accelerator
    accelerator = Accelerator(
        split_batches = split_batches,
        mixed_precision = 'fp16' if not fp32 else 'no'
    )

    accelerator.native_amp = amp

    geo_diffusion_model, geo_diffusion_opt = accelerator.prepare(geo_diffusion_model, geo_diffusion_opt)
    tex_diffusion_model, tex_diffusion_opt = accelerator.prepare(tex_diffusion_model, tex_diffusion_opt)

    #if accelerator.is_main_process:
    ema_geo_diffusion_model = EMA(geo_diffusion_model, beta=ema_decay, update_every=ema_update_every)
    ema_tex_diffusion_model = EMA(tex_diffusion_model, beta=ema_decay, update_every=ema_update_every)
    ema_geo_diffusion_model.to(device)
    ema_tex_diffusion_model.to(device)

    grid_size = None
    grid_z = None
    grid_c = None

    if rank == 0:
        print('Create grid size...')
        grid_size = setup_snapshot_image_grid(img_res=img_res, inference=inference_vis)

        torch.manual_seed(1234)
        grid_z = torch.randn([grid_size[0] * grid_size[1], G_ema.z_dim], device=device).split(1)  # This one is the latent code for shape generation
        grid_c = torch.ones(grid_size[0] * grid_size[1], device=device).split(1)  # This one is not used, just for the compatiable with the code structure.


    if rank == 0:
        print('Initializing logs...')
 
    if rank == 0:
        print(f'Training for {train_num_steps} steps...')
        print()

    if progress_fn is not None:
        progress_fn(0, train_num_steps)

    # Training Iterations
    step = 0

    with tqdm(initial=step, total=train_num_steps, disable = not accelerator.is_main_process) as pbar:
        
        while step < train_num_steps:
            # Fetch training data.
            with torch.autograd.profiler.record_function('data_fetch'):
                gen_geo_z = torch.randn([(batch_size // num_gpus), G_ema.z_dim], device=device)
                gen_tex_z = torch.randn_like(gen_geo_z)
                gen_c = [np.array([], dtype=np.float32) for _ in range((batch_size // num_gpus))]
                gen_c = torch.from_numpy(np.stack(gen_c)).pin_memory().to(device)

            ws_tex = G_ema.mapping(gen_tex_z, gen_c)
            ws_geo = G_ema.mapping_geo(gen_geo_z, gen_c)
            
            img, sdf, syn_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, _, _ = G_ema.synthesis(
                ws=ws_tex, update_emas=False,
                return_shape=True,
                ws_geo=ws_geo,
            )
            
            preprocess_img = []
            for im in img:
                preprocess_img.append(preprocess(tensor_to_img(im)))
            preprocess_img = torch.stack(preprocess_img)

            # training data
            ws_tex, ws_geo, preprocess_img = ws_tex[:, 0].unsqueeze(1).detach().to(device), ws_geo[:, 0].unsqueeze(1).detach().to(device), preprocess_img.detach().to(device)
            ws_tex = ws_tex.repeat(1, 2, 1)

            clip_c = clip_model.encode_image(preprocess_img)
            geo_cond = clip_c.unsqueeze(1)
            tex_cond = torch.cat([ws_geo, clip_c.unsqueeze(1)], dim=1)

            total_loss = 0.

            with accelerator.autocast():
                geo_loss = geo_diffusion_model(ws_geo, geo_cond)
                tex_loss = tex_diffusion_model(ws_tex, tex_cond)
                loss = geo_loss + tex_loss
                total_loss += loss.item()

            accelerator.backward(loss)
            # loss.backward()

            accelerator.clip_grad_norm_(geo_diffusion_model.parameters(), 1.0)
            accelerator.clip_grad_norm_(tex_diffusion_model.parameters(), 1.0)
            pbar.set_description(f'loss: {total_loss:.4f}')

            accelerator.wait_for_everyone()

            geo_diffusion_opt.step()
            tex_diffusion_opt.step()
            geo_diffusion_opt.zero_grad()
            tex_diffusion_opt.zero_grad()

            accelerator.wait_for_everyone()

            step += 1
            if accelerator.is_main_process:
                ema_geo_diffusion_model.update()
                ema_tex_diffusion_model.update()

                # if step != 0 and step % save_and_sample_every == 0:
                #     ema_geo_diffusion_model.ema_model.eval()
                #     ema_tex_diffusion_model.ema_model.eval()

            if step % network_snapshot_ticks == 0:
                #snapshot_data = dict(geo_diff=geo_diffusion_model, tex_diff=tex_diffusion_model)
                snapshot_data = dict(geo_diff=geo_diffusion_model, tex_diff=tex_diffusion_model,
                                     ema_geo_diff=ema_geo_diffusion_model, ema_tex_diff=ema_tex_diffusion_model)
                snapshot_pkl = os.path.join(run_dir, f'diffusion-network-snapshot-{step}.pkl')
                all_model_dict = {'geo_diff': snapshot_data['geo_diff'].state_dict(), 'tex_diff': snapshot_data['tex_diff'].state_dict(),
                                  'ema_geo_diff': snapshot_data['ema_geo_diff'].state_dict(), 'ema_tex_diff': snapshot_data['ema_tex_diff'].state_dict()}
                torch.save(all_model_dict, snapshot_pkl.replace('.pkl', '.pt'))

            pbar.update(1)



        accelerator.print('training complete')
        
       
