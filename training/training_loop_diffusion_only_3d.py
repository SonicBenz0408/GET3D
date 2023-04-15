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
from training.inference_utils import save_image_grid, save_visualization, save_visualization_with_cond
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
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
        image_snapshot_ticks=3000,  # How often to save image snapshots? None = disable.
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
    
    if resume_pretrain is not None and (rank == 0):
        # We're not reusing the loading function from stylegan3 codebase,
        # since we have some variables that are not picklable.
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
    
    # Freeze G & D
    G_ema.requires_grad_(False)

    tensor_to_img = transforms.ToPILImage()

    # create two diffusion model
    geo_Unet = Unet1D(dim=diff_ch, dim_mults=(1, 2, 4, 8), channels=1, resnet_block_groups=2).to(device)
    geo_diffusion_model = GaussianDiffusion1D(geo_Unet, seq_length=G_kwargs.w_dim, timesteps=1000, objective='pred_v').to(device)
    tex_Unet = Unet1D(dim=diff_ch, dim_mults=(1, 2, 4, 8), channels=1, resnet_block_groups=2).to(device)
    tex_diffusion_model = GaussianDiffusion1D(tex_Unet, seq_length=G_kwargs.w_dim, timesteps=1000, objective='pred_v').to(device)

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

        n_shape = grid_size[0] * grid_size[1]
        grid_z = torch.randn([n_shape, G_kwargs.w_dim], device=device).unsqueeze(1).split(1)  # random code for geometry
        grid_tex_z = torch.randn([n_shape, G_kwargs.w_dim], device=device).unsqueeze(1).split(1)  # random code for texture
        grid_c = torch.ones(n_shape, device=device).split(1)


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
            
            # training data
            ws_tex, ws_geo = ws_tex[:, 0].unsqueeze(1).detach().to(device), ws_geo[:, 0].unsqueeze(1).detach().to(device)

            total_loss = 0.

            with accelerator.autocast():
                geo_loss = geo_diffusion_model(ws_geo)
                tex_loss = tex_diffusion_model(ws_tex)
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

                if step != 0 and step % image_snapshot_ticks == 0:
                    ema_geo_diffusion_model.ema_model.eval()
                    ema_tex_diffusion_model.ema_model.eval()
                    print('==> generate ')
                    save_visualization_with_cond(
                        G_ema, ema_geo_diffusion_model.ema_model, ema_tex_diffusion_model.ema_model, None, grid_z, grid_c, run_dir, step, grid_size, 0,
                        save_all=False,
                        grid_tex_z=grid_tex_z
                    )

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
        
       
