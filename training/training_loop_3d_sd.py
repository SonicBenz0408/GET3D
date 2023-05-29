# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Main training loop."""

import copy
import json
import math
import os
import time
import ruamel.yaml as yaml

import matplotlib.pyplot as plt
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from rich.progress import (Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)
from torchvision.transforms import Normalize
from tqdm import tqdm
from transformers import logging

import dnnlib
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from metrics import metric_main
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from training.inference_utils import save_image_grid, save_visualization
from training.sample_camera_distribution import create_camera_from_angle
from training.ldm_feature_extractor import FeatureExtractor
from blip.blip import blip_decoder

logging.set_verbosity_error()


# ----------------------------------------------------------------------------
# Function to save the real image for discriminator training
def setup_snapshot_image_grid(training_set, random_seed=0, inference=False):
    rnd = np.random.RandomState(random_seed)
    grid_w = 7
    grid_h = 4
    min_w = 8 if inference else grid_w
    min_h = 9 if inference else grid_h
    gw = np.clip(1024 // training_set.image_shape[2], min_w, 32)
    gh = np.clip(1024 // training_set.image_shape[1], min_h, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels, masks = zip(*[training_set[i][:3] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels), masks


def clean_training_set_kwargs_for_metrics(training_set_kwargs):
    # We use this function to remove or change custom kwargs for dataset
    # we used these kwargs to comput md5 for the cache file of FID
    if 'add_camera_cond' in training_set_kwargs:
        training_set_kwargs['add_camera_cond'] = True
    return training_set_kwargs


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = LatentDiffusion(**config.model.get("params", dict()))
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def transform(imgs, W, H):
    return F.interpolate(imgs, size=(W, H))

normalize = Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
def clip_preprocess(imgs, masks=None, res=224, background_aug=True):
    imgs = F.interpolate(imgs, size=(res, res), mode="bicubic")
    all_img = []
    if background_aug:
        masks = F.interpolate(masks.unsqueeze(1), size=(res, res), mode="bicubic").squeeze()
        for img, mask in zip(imgs, masks):
            background = torch.randn_like(img).to(imgs.device)
            img = img * (mask > 0).to(img.dtype) + background * (1 - (mask > 0).to(img.dtype))
            img = normalize(img)
            all_img.append(img)
    else:
        for img in imgs:
            img = normalize(img)
            all_img.append(img)
    imgs = torch.stack(all_img)
    return imgs
# ----------------------------------------------------------------------------
def training_loop(
        run_dir='.',  # Output directory.
        training_set_kwargs={},  # Options for training set.
        data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        G_opt_kwargs={},  # Options for generator optimizer.
        D_opt_kwargs={},  # Options for discriminator optimizer.
        loss_kwargs={},  # Options for loss function.
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        img_res=1024,
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus[.
        batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu=4,  # Number of samples processed at a time by one GPU.
        ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup=0.05,  # EMA ramp-up coefficient. None = no rampup.
        G_reg_interval=None,  # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
        total_kimg=100000,  # Total length of the training.
        kimg_per_tick=4,  # Progress snapshot interval.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
        resume_kimg=0,  # First kimg to report when resuming training. ######
        abort_fn=None,
        # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
        inference_vis=False,  # Whether running inference or not.
        detect_anomaly=False,
        resume_pretrain=None,
        config=None,
        sd_ckpt=None,
        blip_config=None
):
    from torch_utils.ops import bias_act, filtered_lrelu, upfirdn2d
    upfirdn2d._init()
    bias_act._init()
    filtered_lrelu._init()
    if num_gpus > 1:
        torch.distributed.barrier()
    time.time()
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
        print('Loading training set...')

    # Set up training dataloader
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)

    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set, sampler=training_set_sampler, batch_size=batch_size // num_gpus,
            **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    if rank == 0:
        print('Constructing networks...')

    # Constructing networks
    common_kwargs = dict(
        c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G_kwargs['device'] = device
    D_kwargs['device'] = device

    if num_gpus > 1:
        torch.distributed.barrier()
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()  # deepcopy can make sure they are correct.
    if resume_pretrain is not None and (rank == 0):
        # We're not reusing the loading function from stylegan3 codebase,
        # since we have some variables that are not picklable.
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G.load_state_dict(model_state_dict['G_ema'], strict=False)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=False)

    G.requires_grad_(True)

    F = FeatureExtractor().train().requires_grad_(False).to(device)

    blip_config = yaml.load(open(blip_config, 'r'), Loader=yaml.Loader)

    caption_model = blip_decoder(pretrained=blip_config['pretrained'], image_size=blip_config['image_size'], vit=blip_config['vit'], 
                           prompt=blip_config['prompt']).to(device).eval()
    caption_model.requires_grad_(False)

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    if rank == 0:
        print('Setting up training phases...')

    grid_size = None
    grid_z = None
    grid_c = None
    grid_images = None
    grid_captions = None

    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels, masks = setup_snapshot_image_grid(training_set=training_set, inference=inference_vis)
        masks = np.stack(masks)
        images = np.concatenate((images, masks[:, np.newaxis, :, :].repeat(3, axis=1) * 255.0), axis=-1)
        if not inference_vis:
            save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        torch.manual_seed(1234)
        grid_z = torch.randn([images.shape[0], G.z_dim], device=device).split(1)  # This one is the latent code for shape generation
        grid_c = torch.ones(images.shape[0], device=device).split(1)  # This one is not used, just for the compatiable with the code structure.
        grid_images = (torch.from_numpy(images).to(device).to(torch.float32) / 127.5 - 1).split(1)
        grid_captions = caption_model(clip_preprocess(torch.from_numpy(images).to(device).to(torch.float32) / 127.5 - 1, res=blip_config['image_size'], background_aug=False))

    print(grid_captions)

    if rank == 0:
        print('Initializing logs...')
 
    if rank == 0:
        print(f'Training for {total_kimg} epochs...')
        print()

    if progress_fn is not None:
        progress_fn(0, total_kimg)

    # Training Iterations
    step = 0
    cur_nimg = resume_kimg * 1000
    cur_tick = 0

    G_opt = dnnlib.util.construct_class_by_name(params=G.parameters(), **G_opt_kwargs)
    F_trainable_params = list(F.ldm_feature_extractor.clip_project.parameters()) + \
        list(F.ldm_feature_extractor.alpha_cond) + \
        list(F.ldm_feature_extractor.time_embed_project.parameters()) + \
        list(F.ldm_feature_extractor.alpha_cond_time_embed) + \
        list(F.downsample.parameters())
            
    F_opt = torch.optim.Adam(params=F_trainable_params, lr=1e-4, betas=[0, 0.99])
    
    criterion_clip = torch.nn.CosineEmbeddingLoss()
    criterion_recon = torch.nn.MSELoss()
    loss_log = os.path.join(run_dir, f'loss.txt')
    with open(loss_log, 'w') as f:
        pass

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        TextColumn("[green][bold]{task.fields[step]}[/bold]/{task.total}[/green] loss: {task.fields[loss_clip_high]:.5f} + {task.fields[loss_clip_low]:.5f} = [bold]{task.fields[loss]:.5f}[/bold]"),
        ) as progress:
        task = progress.add_task("Training...", total=total_kimg, step=step, loss_clip_high=0, loss_clip_low=0, loss=0)
        
        camera_radius = 1.2  # align with what ww did in blender
        camera_r = torch.zeros(batch_size * G_ema.synthesis.n_views, 1, device=device) + camera_radius

        while step < total_kimg:
            # ---------------------------------------------------------------------------------------
            G.train()
            F.train()
            F.requires_grad_(True)
            F.ldm_feature_extractor.training = True
            # Fetch training data.
            with torch.autograd.profiler.record_function('data_fetch'):
                real_img, real_camera, real_mask = next(training_set_iterator)
                real_img = (real_img.to(device).to(torch.float32) / 127.5 - 1)
                rot = real_camera[:, 0:1].repeat(1, G_ema.synthesis.n_views).reshape((batch_size * G_ema.synthesis.n_views, 1)).to(device)
                ele = real_camera[:, 1:2].repeat(1, G_ema.synthesis.n_views).reshape((batch_size * G_ema.synthesis.n_views, 1)).to(device)
                world2cam_matrix, forward_vector, camera_origin, elevation_angle, rotation_angle = create_camera_from_angle(ele, -rot - 0.5 * math.pi, camera_r, device=device)
                camera = (world2cam_matrix.reshape(batch_size, G_ema.synthesis.n_views, 4, 4), camera_origin.reshape(batch_size, G_ema.synthesis.n_views, 3), \
                    camera_r, rotation_angle, elevation_angle)
                real_mask = real_mask.to(device).to(torch.float32)
                all_tex_z = torch.randn([batch_size, G.z_dim], device=device)
                all_gen_c = torch.randn([batch_size, G.z_dim], device=device)
                all_geo_z = torch.randn_like(all_tex_z)

            text_cond = caption_model(clip_preprocess(real_img, res=blip_config['image_size']))
            print(text_cond)
            feature_map = F(real_img, text_cond)

            G.update_triplane_const(feature_map)

            ws_tex = G.mapping(all_tex_z, all_gen_c, update_emas=False)
            ws_geo = G.mapping_geo(all_geo_z, all_gen_c, update_emas=False)
            img, sdf, syn_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, _, _ = G.synthesis(
                ws=ws_tex, update_emas=False,
                return_shape=True,
                ws_geo=ws_geo,
                camera=camera
            )
            
            #  Calculate clip loss and feature loss
            gen_img, gen_mask = img[:, :training_set.num_channels, :, :], img[:, training_set.num_channels, :, :]
            
            ori_clip_features = F.ldm_feature_extractor.clip.embed_image(real_img).image_embed
            gen_clip_features = F.ldm_feature_extractor.clip.embed_image(gen_img).image_embed
            clip_text_features = F.ldm_feature_extractor.clip.embed_text(text_cond).text_embed

            loss_clip_high = criterion_clip(ori_clip_features, gen_clip_features, torch.ones(real_img.shape[0]).to(device))
            loss_clip_low = criterion_recon(ori_clip_features, clip_text_features, torch.ones(real_img.shape[0]).to(device))
            loss = loss_clip_high + loss_clip_low

            G_opt.zero_grad()
            F_opt.zero_grad()
            loss.backward()
            G_opt.step()
            F_opt.step()
            
            # Update G_ema.
            with torch.autograd.profiler.record_function('Gema'):
                ema_nimg = ema_kimg * 1000
                if ema_rampup is not None:
                    ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
                ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)

            if step % image_snapshot_ticks == 0:
                
                print('==> generate ')
                save_visualization(
                    G_ema, F, grid_z, grid_c, grid_images, grid_captions, run_dir, cur_nimg, grid_size, step,
                    image_snapshot_ticks,
                    save_all=(cur_tick % (image_snapshot_ticks * 4) == 0) and training_set.resolution < 512,
                )
            if step % network_snapshot_ticks == 0:
                snapshot_data = dict(F=F, G=G, G_ema=G_ema)
                for key, value in snapshot_data.items():
                    if isinstance(value, torch.nn.Module) and not isinstance(value, dr.ops.RasterizeGLContext):
                        snapshot_data[key] = value
                snapshot_pkl = os.path.join(run_dir, f'snapshot-{step}.pkl')
                all_model_dict = {'F': snapshot_data['F'].state_dict(), 'G': snapshot_data['G'].state_dict(), 'G_ema': snapshot_data['G_ema'].state_dict()}
                torch.save(all_model_dict, snapshot_pkl.replace('.pkl', '.pt'))
            
            step += 1
            cur_nimg += batch_size

            with open(loss_log, 'a') as f:
                f.write(f'CLIP_image_loss: {loss_clip_high.item():.4f} CLIP_text_loss: {loss_clip_low.item():.4f} loss: {loss.item()}\n')

            progress.update(task, advance=1, step=step, loss_clip_high=loss_clip_high.item(), loss_clip_low=loss_clip_low.item(), loss=loss.item())

    # Done.
    if rank == 0:
        print()
        print('Exiting...')
