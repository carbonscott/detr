"""
DETR

* Forward pass
* Backward pass
    - Set Criterion
    - Matcher

Reference: https://github.com/facebookresearch/detr/blob/main/models/detr.py#L83
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder
from .transformer   import TransformerEncoderDecoder

from ..configurator import Configurator


class DETR(nn.Module):

    @staticmethod
    def get_default_config():
        CONFIG = Configurator()
        with CONFIG.enable_auto_create():
            CONFIG.BACKBONE = ImageEncoder.get_default_config()
            CONFIG.BACKBONE.OUTPUT_CHANNELS = {
                'layer4' : 2048,
            }
            CONFIG.BACKBONE.OUTPUT_SHAPE = (60, 60)

            CONFIG.TRANSFORMER.NUM_OBJECT_QUERY   = 100
            CONFIG.TRANSFORMER.EMBD_SIZE          = 256
            CONFIG.TRANSFORMER.NUM_BLOCKS         = 6
            CONFIG.TRANSFORMER.NUM_HEADS          = 8
            CONFIG.TRANSFORMER.ATTENTION_DROPOUT  = 0.0
            CONFIG.TRANSFORMER.RESIDUAL_DROPOUT   = 0.0
            CONFIG.TRANSFORMER.FEEDFOWARD_DROPOUT = 0.0

            CONFIG.NUM_CLASSES = 2

        return CONFIG


    def __init__(self, config = None):
        super().__init__()

        self.config = DETR.get_default_config() if config is None else config

        # Create the image encoder...
        self.backbone = ImageEncoder(config = self.config.BACKBONE)

        # Transition layer...
        self.backbone_to_input = nn.Conv2d(in_channels  = self.config.BACKBONE.OUTPUT_CHANNELS['layer4'],
                                           out_channels = self.config.TRANSFORMER.EMBD_SIZE,
                                           kernel_size  = 1,
                                           stride       = 1,
                                           padding      = 0)

        # Create the learnable positional embedding...
        H, W = self.config.BACKBONE.OUTPUT_SHAPE
        self.pos_embd = nn.Parameter(torch.randn(1, self.config.TRANSFORMER.EMBD_SIZE, H, W))

        # Create the transformer module...
        context_length         = self.config.TRANSFORMER.NUM_OBJECT_QUERY
        memory_length          = H * W    # ...Spatial memory
        embd_size              = self.config.TRANSFORMER.EMBD_SIZE
        num_blocks             = self.config.TRANSFORMER.NUM_BLOCKS
        num_heads              = self.config.TRANSFORMER.NUM_HEADS
        attention_dropout      = self.config.TRANSFORMER.ATTENTION_DROPOUT
        residual_dropout       = self.config.TRANSFORMER.RESIDUAL_DROPOUT
        feedforward_dropout    = self.config.TRANSFORMER.FEEDFOWARD_DROPOUT
        self.transformer_endec = TransformerEncoderDecoder(
                                     context_length      = context_length,
                                     memory_length       = memory_length,
                                     embd_size           = embd_size,
                                     num_blocks          = num_blocks,
                                     num_heads           = num_heads,
                                     uses_causal_mask    = False,
                                     attention_dropout   = 0.0,
                                     residual_dropout    = 0.0,
                                     feedforward_dropout = 0.0,
                                 )

        # Create a shared prediction head for all object queries...
        num_classes     = self.config.NUM_CLASSES
        dim_box_repr    = 4    # ...Dimension of bounding box representation
        self.pred_heads = nn.Linear(embd_size, num_classes + dim_box_repr)


    def forward(self, x):
        """

        Arguments:

            x : torch.Tensor, (B, C, H, W)

        Returns:

            y_hat : torch.Tensor, (B, num_classes + dim_box_repr)

        """
        # Extract features through the backbone network...
        x = self.backbone(x)
        x = self.backbone_to_input(x)

        # Attach positional embedding...
        x = x + self.pos_embd

        # Flatten the embedding...
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(-1, -2).contiguous()

        # Encode and decode through transformer endec...
        x = self.transformer_endec(x)

        # Make predictions...
        x = self.pred_heads(x)

        return x
