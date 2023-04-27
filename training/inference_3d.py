# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import copy
import os
import pathlib

import numpy as np
import torch
from rich.progress import track

import dnnlib
from metrics import metric_main
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from training.inference_utils import (save_geo_for_inference,
                                      save_textured_mesh_for_inference,
                                      save_visualization,
                                      save_visualization_for_interpolation)


def clean_training_set_kwargs_for_metrics(training_set_kwargs):
    if 'add_camera_cond' in training_set_kwargs:
        training_set_kwargs['add_camera_cond'] = True
    return training_set_kwargs


# ----------------------------------------------------------------------------
def inference(
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
        inference_save_interpolation=False,
        inference_compute_fid=False,
        inference_generate_geo=False,
        **dummy_kawargs
):
    from torch_utils.ops import bias_act, filtered_lrelu, upfirdn2d
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
        img_resolution=training_set_kwargs['resolution'] if 'resolution' in training_set_kwargs else 1024, img_channels=3)
    G_kwargs['device'] = device

    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    # D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(
    #     device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()  # deepcopy can make sure they are correct.
    if resume_pretrain is not None and (rank == 0):
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G.load_state_dict(model_state_dict['G'], strict=True)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        # D.load_state_dict(model_state_dict['D'], strict=True)
    
    
    target_modules = ["mapping", "mapping_geo"]
    hook_data = {}
    def get_activation(name):
        def hook(module, input, output):
            if name not in hook_data:
                hook_data[name] = {
                    "input": [],
                    "c":[],
                    "output": [],
                }
            hook_data[name]["input"].append(input[0].detach().cpu().numpy())
            hook_data[name]["c"].append(input[1].detach().cpu().numpy())
            hook_data[name]["output"].append(output.detach().cpu().numpy())
        return hook

    for name, layer in G_ema.named_modules():
        if name in target_modules:
            print(name, layer)
            layer.register_forward_hook(get_activation(name))
    
    print('==> generate ')
    # for i in track(range(100)):
    grid_size = (10, 10)
    n_shape = grid_size[0] * grid_size[1]
    grid_z = torch.randn([n_shape, G.z_dim], device=device).split(1)  # random code for geometry
    grid_tex_z = torch.randn([n_shape, G.z_dim], device=device).split(1)  # random code for texture
    grid_c = torch.randint(0, D_kwargs['cmap_dim'], [n_shape, G.c_dim], device=device).split(1)

    save_visualization(
        G_ema, grid_z, grid_c, run_dir, 0, grid_size, 0,
        save_all=False,
        grid_tex_z=grid_tex_z,
        cmap_dim=D_kwargs['cmap_dim']
    )

    if inference_to_generate_textured_mesh:
        print('==> generate inference 3d shapes with texture')
        save_textured_mesh_for_inference(
            G_ema, grid_z, grid_c, run_dir, save_mesh_dir='texture_mesh_for_inference',
            c_to_compute_w_avg=None, grid_tex_z=grid_tex_z, cmap_dim=D_kwargs['cmap_dim'])

    if inference_save_interpolation:
        print('==> generate interpolation results')
        save_visualization_for_interpolation(G_ema, save_dir=os.path.join(run_dir, 'interpolation'))

    if inference_compute_fid:
        print('==> compute FID scores for generation')
        for metric in metrics:
            training_set_kwargs = clean_training_set_kwargs_for_metrics(training_set_kwargs)
            training_set_kwargs['split'] = 'test'
            result_dict = metric_main.calc_metric(
                metric=metric, G=G_ema,
                dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank,
                device=device)
            metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=resume_pretrain)

    if inference_generate_geo:
        print('==> generate 7500 shapes for evaluation')
        save_geo_for_inference(G_ema, run_dir)

    for name in target_modules:
        hook_save_dir = pathlib.Path(run_dir) / 'hook' / name
        hook_save_dir.mkdir(parents=True, exist_ok=True)
        
        all_input = np.concatenate(hook_data['mapping']["input"])
        all_c = np.concatenate(hook_data['mapping']["c"])
        all_output = np.concatenate(hook_data['mapping']["output"])

        print(all_input.shape)
        print(all_output.shape)
        
        np.save(hook_save_dir / 'input.npy', all_input)
        np.save(hook_save_dir / 'c.npy', all_c)
        np.save(hook_save_dir / 'output.npy', all_output)