import torch
import torch.nn as nn
from odise.modeling.meta_arch.ldm import LdmImplicitCaptionerExtractor, LdmExtractor


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.ldm_feature_extractor = LdmCaptionerExtractor(
            encoder_block_indices=(5, 7),
            unet_block_indices=(2, 5, 8, 11),
            decoder_block_indices=(2, 5),
            steps=(0,),
            learnable_time_embed=True,
            num_timesteps=1,
            clip_model_name="ViT-L-14-336",
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(2560, 512, kernel_size=(1,1)),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
        )
        
    def forward(self, img, caption):
        img = nn.functional.interpolate(img, (512, 512))
        features = self.ldm_feature_extractor(dict(img=img, caption=caption))
        feature_map = self.downsample(features[2])
        return feature_map


class LdmCaptionerExtractor(nn.Module):
    def __init__(
        self,
        learnable_time_embed=True,
        num_timesteps=1,
        clip_model_name="ViT-L-14",
        **kwargs,
    ):
        super().__init__()

        self.ldm_extractor = LdmExtractor(**kwargs)

    @property
    def feature_size(self):
        return self.ldm_extractor.feature_size

    @property
    def feature_dims(self):
        return self.ldm_extractor.feature_dims

    @property
    def feature_strides(self):
        return self.ldm_extractor.feature_strides

    @property
    def num_groups(self) -> int:

        return self.ldm_extractor.num_groups

    @property
    def grouped_indices(self):

        return self.ldm_extractor.grouped_indices

    def extra_repr(self):
        return f"learnable_time_embed={self.learnable_time_embed}"

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs (dict): expected keys: "img", Optional["caption"]

        """
        self.set_requires_grad(self.training)

        return self.ldm_extractor(batched_inputs)

    def set_requires_grad(self, requires_grad):
        for p in self.ldm_extractor.ldm.ldm.model.parameters():
            p.requires_grad = requires_grad