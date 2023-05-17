from omegaconf import OmegaConf
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as  plt
import torch
from PIL import Image
import PIL
import numpy as np
from einops import rearrange, repeat
import argparse
import os
from training.feature_pyramid_network import SDFeatureExtractor


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

def load_img(path, W, H):
    image = Image.open(path).convert("RGB")
    image = image.resize((W, H))
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    opt = parser.parse_args()
    
    config = OmegaConf.load(f"pretrained_model/v1-inference.yaml")
    ckpt = "pretrained_model/sd-v1-4.ckpt"
    device = "cuda"

    feature_extractor = SDFeatureExtractor(channel_list=[1280, 1280, 640, 320], out_dim=512, init_dim=8, factor=8).to(device)
    SD_model = load_model_from_config(config, f"{ckpt}")
    SD_model = SD_model.to(device)
    SD_sampler = DDIMSampler(SD_model)


    prompts = [""]
    C, H, W, f = 4, 512, 512, 8
    steps = 1
    scale = 7.5
    strength = 0.0
    save_path = "test_img/"

    SD_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)

    init_image = load_img(opt.img, W, H).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=1)
    init_latent = SD_model.get_first_stage_encoding(SD_model.encode_first_stage(init_image))

    uc = SD_model.get_learned_conditioning(1 * [""])
    c = SD_model.get_learned_conditioning(prompts)

    shape = [C, H // f, W // f]
    
    t_enc =  torch.tensor([0], device=device).expand(1)
    _, unet_features = SD_model.model.diffusion_model(init_latent, t_enc, c)
    for feats in unet_features:
        print(feats.shape)
    features = feature_extractor(unet_features)
    print(features.shape)

    
    # save_img = 255. * rearrange(init_image[0].cpu().numpy(), 'c h w -> h w c')
    # save_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
    # Image.fromarray(save_img.astype(np.uint8)).save(os.path.join(save_path, f"ori.png"))
    # Image.fromarray(save_sample.astype(np.uint8)).save(os.path.join(save_path, f"sample.png"))