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

from ..configurator import Configurator


class MultiHeadAttention(nn.Module):
    """
    DETR specific Multi-headed attention.
    """
    def __init__(self, embd_size,
                       context_length,
                       head_size,
                       uses_causal_mask  = False,
                       attention_dropout = 0.0,
                       residual_dropout  = 0.0):
        super().__init__()

        self.embd_size         = embd_size
        self.head_size         = head_size
        self.attention_dropout = attention_dropout
        self.residual_dropout  = residual_dropout
        self.uses_causal_mask  = uses_causal_mask

        # Internal variable
        self._num_heads = embd_size // head_size

        # Self-attention layer to update each node by aggregating features from all other nodes...
        # The message-passing based communication happens in another vector space.
        self.proj_q = nn.Linear(self.embd_size, self.embd_size)    # What do I (this node) want?
        self.proj_k = nn.Linear(self.embd_size, self.embd_size)    # What do I have publicly?
        self.proj_v = nn.Linear(self.embd_size, self.embd_size)    # What do I provide to update the entire graph?

        # Store a mask to prevent it from gradient tracking...
        mask = torch.ones(context_length, context_length).triu(diagonal=1).bool()
        self.register_buffer('mask', mask)

        # Linear projection...
        self.proj_linear = nn.Linear(embd_size, embd_size)

        # Use dropout after softmax...
        self.attention_dropout = nn.Dropout(attention_dropout)

        # Use dropout at the end...
        self.residual_dropout = nn.Dropout(residual_dropout)


    def forward(self, x, x_lookup_k = None, x_lookup_v):
        B, T, E = x.shape

        num_heads = self._num_heads
        head_size = self.head_size

        # Linearly project them to a vector space...
        q = self.proj_q(x)   # B, T, E
        k = self.proj_k(x if x_lookup_k is None else x_lookup_k)   # B, T, E
        v = self.proj_v(x if x_lookup_v is None else x_lookup_v)   # B, T, E

        # Changes the view to facilitate scaled dot product within each head...
        q = q.view(B, T, num_heads, head_size).transpose(1, 2)   # (B, num_heads, T, head_size)
        k = k.view(B, T, num_heads, head_size).transpose(1, 2)   # (B, num_heads, T, head_size)
        v = v.view(B, T, num_heads, head_size).transpose(1, 2)   # (B, num_heads, T, head_size)

        # Scaled dot product...
        w = q @ k.transpose(-1, -2)    # (B, num_heads, T, head_size) @ (B, num_heads, head_size, T) ->
                                       # (B, num_heads, T, T)
        w /= torch.sqrt(torch.tensor(head_size))

        # Use causal mask???
        if self.uses_causal_mask:
            # Masking in the decoder to enable causal relation...
            w[:, :,self.mask[:T,:T]] = float('-inf')    # (B, num_heads, :T, :T)   `:T` means upto `T`

        # Obtain the softmax...
        w = w.softmax(dim = -1)    # (B, num_heads, T, T)

        # Aggregate information from all nodes...
        a = w @ v    # (B, num_heads, T, T) @ (B, num_heads, T, head_size) ->
                     # (B, num_heads, T, head_size)

        a = self.attention_dropout(a)

        # Reshape it to (B, T, E)...
        a = a.transpose(2, 1).contiguous()    # (B, num_heads, T, head_size) -> (B, T, num_heads, head_size)
        a = a.view(B, T, E)

        # Linear projection...
        y = self.proj_linear(a)

        # Optional dropout...
        y = self.residual_dropout(y)

        return y




class Transcoder(nn.Module):

    @staticmethod
    def get_default_config():
        config = Configurator()
        with config.enable_auto_create():
            # Encoder
            config.encoder.tok_size            = None
            config.encoder.embd_size           = None
            config.encoder.context_length      = None
            config.encoder.num_blocks          = None
            config.encoder.num_heads           = None
            config.encoder.uses_causal_mask    = None
            config.encoder.attention_dropout   = None
            config.encoder.residual_dropout    = None
            config.encoder.feedforward_dropout = None

            # Decoder
            config.decoder.tok_size            = None
            config.decoder.embd_size           = None
            config.decoder.context_length      = None
            config.decoder.num_blocks          = None
            config.decoder.num_heads           = None
            config.decoder.uses_causal_mask    = None
            config.decoder.attention_dropout   = None
            config.decoder.residual_dropout    = None
            config.decoder.feedforward_dropout = None

        return config


    def __init__(self, config):
        super().__init__()

        self.encoder = Transformer(tok_size            = config.encoder.tok_size,
                                   embd_size           = config.encoder.embd_size,
                                   context_length      = config.encoder.context_length,
                                   num_blocks          = config.encoder.num_blocks,
                                   num_heads           = config.encoder.num_heads,
                                   uses_causal_mask    = config.encoder.uses_causal_mask,
                                   attention_dropout   = config.encoder.attention_dropout,
                                   residual_dropout    = config.encoder.residual_dropout,
                                   feedforward_dropout = config.encoder.feedforward_dropout

        self.decoder = Transformer(tok_size            = config.decoder.tok_size,
                                   embd_size           = config.decoder.embd_size,
                                   context_length      = config.decoder.context_length,
                                   num_blocks          = config.decoder.num_blocks,
                                   num_heads           = config.decoder.num_heads,
                                   uses_causal_mask    = config.decoder.uses_causal_mask,
                                   attention_dropout   = config.decoder.attention_dropout,
                                   residual_dropout    = config.decoder.residual_dropout,
                                   feedforward_dropout = config.decoder.feedforward_dropout


    def forward(self, x):
        """
        Cross attention between x and x_lookup.
        """
        # Encode input x...
        x_lookup = self.encoder(x)
