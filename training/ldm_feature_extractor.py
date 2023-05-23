import torch

from odise.modeling.meta_arch.ldm import LdmImplicitCaptionerExtractor


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ldm_feature_extractor = LdmImplicitCaptionerExtractor(
            encoder_block_indices=(5, 7),
            unet_block_indices=(2, 5, 8, 11),
            decoder_block_indices=(2, 5),
            steps=(0,),
            learnable_time_embed=True,
            num_timesteps=1,
            clip_model_name="ViT-L-14-336",
        )

        self.downsample = torch.nn.Sequential(
            torch.nn.Conv2d(2560, 512, kernel_size=(1,1)),
            torch.nn.Conv2d(512, 512, kernel_size=(3,3)),
            torch.nn.Conv2d(512, 512, kernel_size=(3,3)),
        )
        
    def forward(self, img):
        img = torch.nn.functional.interpolate(img, (512, 512))
        features = self.ldm_feature_extractor(dict(img=img))
        feature_map = self.downsample(features[2])
        return feature_map