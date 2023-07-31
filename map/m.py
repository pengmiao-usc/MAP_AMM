import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor=4, dropout=0.):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    )

class MLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
        super().__init__()
        self.patch_size = patch_size
        image_h, image_w = image_size
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)

        self.patch_embedding = nn.Linear(channels * patch_size ** 2, dim)
        self.mix_layers = nn.ModuleList([
            nn.Sequential(
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout))
            ) for _ in range(depth)
        ])
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)(x)
        x = self.patch_embedding(x)

        for mix_layer in self.mix_layers:
            x = mix_layer(x)

        x = self.layer_norm(x)
        x = Reduce('b n c -> b c', 'mean')(x)
        x = self.mlp_head(x)

        return x

## -- Main -- 
# mlp_mixer_model = MLPMixer(image_size=(224, 224), channels=3, patch_size=16, dim=512, depth=6, num_classes=10)
# output = mlp_mixer_model(torch.randn(1, 3, 224, 224))
# print(output.shape)
