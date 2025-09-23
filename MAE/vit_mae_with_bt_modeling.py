from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEForPreTraining,
    ViTMAEForPreTrainingOutput,
    ViTMAEModelOutput,
    ViTMAEDecoderOutput,
)
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack


@dataclass
class ViTMAEForPreTrainingOutputBT(ViTMAEForPreTrainingOutput):
    mae_loss: Optional[torch.FloatTensor] = None
    bt_loss: Optional[torch.FloatTensor] = None


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd=5e-3, eps=1e-9):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, Z1, Z2):
        # Z1, Z2: (N x d)
        N, d = Z1.shape
        Z1_norm = (Z1 - Z1.mean(0)) / (Z1.std(0) + self.eps)
        Z2_norm = (Z2 - Z2.mean(0)) / (Z2.std(0) + self.eps)

        c = (Z1_norm.T @ Z2_norm) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambd * off_diag


class ViTMAEForPreTrainingWithBT(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.bt_loss = BarlowTwinsLoss(lambd=config.bt_lambda, eps=config.bt_eps)
        self.bt_weight = config.bt_weight
        self.bt_variant = config.bt_variant
    
    def compute_bt_loss_per_image(self, latent):
        z = latent  # (B, N, d)

        bt_losses = []
        for z_img in z:  # each z_img: (N, d)
            N = z_img.size(0)
            c = (z_img.T @ z_img) / N
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            bt_losses.append(on_diag + self.bt_loss.lambd * off_diag)
        bt_loss_val = torch.stack(bt_losses).mean()
        
        return bt_loss_val
    
    def compute_bt_loss_per_batch(self, latent):
        z = latent # (B, N, d)
        B, N, d = z.shape
        z_global = z.reshape(B * N, d)

        # covariance across whole batch
        c = (z_global.T @ z_global) / (B * N)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        bt_loss_val = on_diag + self.bt_loss.lambd * off_diag
        
        return bt_loss_val
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ViTMAEForPreTrainingOutput:
        outputs: ViTMAEModelOutput = self.vit(
            pixel_values, noise=noise, head_mask=head_mask, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs: ViTMAEDecoderOutput = self.decoder(
            latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding
        )
        logits = decoder_outputs.logits

        loss = self.forward_loss(pixel_values, logits, mask, interpolate_pos_encoding=interpolate_pos_encoding)
        if self.bt_variant == "per_image":
            bt_loss = self.compute_bt_loss_per_image(latent)
        elif self.bt_variant == "per_batch":
            bt_loss = self.compute_bt_loss_per_batch(latent)
        else:
            raise NotImplementedError()
        total_loss = loss + 0.005 * bt_loss

        return ViTMAEForPreTrainingOutputBT(
            loss=total_loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mae_loss=loss,
            bt_loss=bt_loss,
        )