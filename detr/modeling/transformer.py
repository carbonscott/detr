import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embd_size,
                       head_size,
                       context_length,
                       memory_length     = None,
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
        if memory_length is None:
            memory_length = context_length

        # Self-attention layer to update each node by aggregating features from all other nodes...
        # The message-passing based communication happens in another vector space.
        self.proj_q = nn.Linear(self.embd_size, self.embd_size)    # What do I (this node) want?
        self.proj_k = nn.Linear(self.embd_size, self.embd_size)    # What do I have publicly?
        self.proj_v = nn.Linear(self.embd_size, self.embd_size)    # What do I provide to update the entire graph?

        # Store a mask to prevent it from gradient tracking...
        mask = torch.ones(context_length, memory_length).triu(diagonal=1).bool()
        self.register_buffer('mask', mask)

        # Linear projection...
        self.proj_linear = nn.Linear(embd_size, embd_size)

        # Use dropout after softmax...
        self.attention_dropout = nn.Dropout(attention_dropout)

        # Use dropout at the end...
        self.residual_dropout = nn.Dropout(residual_dropout)


    def forward(self, embd_q, embd_k = None, embd_v = None):
        B, T, E = embd_q.shape
        _, M, _ = embd_k.shape if embd_k is not None else (None, T, None)

        num_heads = self._num_heads
        head_size = self.head_size

        # Linearly project them to a vector space...
        q = self.proj_q(embd_q)   # B, T, E
        k = self.proj_k(embd_q if embd_k is None else embd_k)   # B, M, E
        v = self.proj_v(embd_q if embd_v is None else embd_v)   # B, M, E

        # Changes the view to facilitate scaled dot product within each head...
        q = q.view(B, T, num_heads, head_size).transpose(1, 2)   # (B, num_heads, T, head_size)
        k = k.view(B, M, num_heads, head_size).transpose(1, 2)   # (B, num_heads, M,  head_size)
        v = v.view(B, M, num_heads, head_size).transpose(1, 2)   # (B, num_heads, M,  head_size)

        # Scaled dot product...
        w = q @ k.transpose(-1, -2)    # (B, num_heads, T, head_size) @ (B, num_heads, head_size, M) ->
                                       # (B, num_heads, T, M)
        w /= torch.sqrt(torch.tensor(head_size))

        # Use causal mask???
        if self.uses_causal_mask:
            # Masking in the decoder to enable causal relation...
            w[:, :,self.mask[:T,:M]] = float('-inf')    # (B, num_heads, :T, :M)   `:M` means upto `M`

        # Obtain the softmax...
        w = w.softmax(dim = -1)    # (B, num_heads, T, M)

        # Aggregate information from all nodes...
        a = w @ v    # (B, num_heads, T, M) @ (B, num_heads, M, head_size) ->
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




class FeedForward(nn.Module):
    def __init__(self, embd_size, dropout = 0.0):
        super().__init__()

        self.ff_layer = nn.Sequential(
            nn.Linear(    embd_size, 4 * embd_size),
            nn.GELU(),
            nn.Linear(4 * embd_size,     embd_size),
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.ff_layer(x)




class TransformerEncoderBlock(nn.Module):
    def __init__(self, embd_size,
                       num_heads,
                       context_length,
                       uses_causal_mask    = False,
                       attention_dropout   = 0.0,
                       residual_dropout    = 0.0,
                       feedforward_dropout = 0.0):
        super().__init__()

        # Define the multi head attention layer to update node position in a sub space using an attention head...
        head_size = embd_size // num_heads
        self.multi_head_att_layer = MultiHeadAttention(embd_size,
                                                       head_size,
                                                       context_length,
                                                       uses_causal_mask  = uses_causal_mask,
                                                       attention_dropout = attention_dropout,
                                                       residual_dropout  = residual_dropout)

        # Define the feedforward layer to add non-linearity to the model...
        self.ff_layer = FeedForward(embd_size, dropout = feedforward_dropout)

        # Define layers to optimize model training...
        self.layer_norm_pre_multi_head  = nn.LayerNorm(embd_size)
        self.layer_norm_pre_feedforward = nn.LayerNorm(embd_size)


    def forward(self, embd_q):
        """
        Arguments:
            embd_q : (B, M, E), query token embedding

        Returns:
            out : (B, M, E)
        """
        # ___/ MULTI-HEAD ATTENTION BLOCK \___
        # Go through multi-head attention to update nodes in vector space...
        # ...Pre norm (Shared)
        embd_q_norm = self.layer_norm_pre_multi_head(embd_q)

        # ...Attention
        embd_q_update = self.multi_head_att_layer(embd_q_norm)    # (B, M, E)

        # ...Residual connection (out -> prenorm)
        embd_q_update += embd_q

        # Learn a better embedding representation by introducing non-linearity...
        # ...Pre norm
        embd_q_update_norm = self.layer_norm_pre_feedforward(embd_q_update)    # (B, M, E)

        # ...Feed forward
        embd_q_out = self.ff_layer(embd_q_update_norm)    # (B, M, E)

        # ...Residual connection (out -> prenorm)
        embd_q_out += embd_q_update

        return embd_q_out




class TransformerDecoderBlock(nn.Module):
    def __init__(self, embd_size,
                       num_heads,
                       context_length,
                       memory_length       = None,
                       uses_causal_mask    = False,
                       attention_dropout   = 0.0,
                       residual_dropout    = 0.0,
                       feedforward_dropout = 0.0):
        super().__init__()

        # Define the multi head attention layer to update node position in a sub space using an attention head...
        head_size = embd_size // num_heads
        self.multi_head_self_att_layer = MultiHeadAttention(embd_size,
                                                            head_size,
                                                            context_length,
                                                            uses_causal_mask  = uses_causal_mask,
                                                            attention_dropout = attention_dropout,
                                                            residual_dropout  = residual_dropout)

        self.multi_head_cross_att_layer = MultiHeadAttention(embd_size,
                                                             head_size,
                                                             context_length,
                                                             memory_length     = memory_length,
                                                             uses_causal_mask  = False,
                                                             attention_dropout = attention_dropout,
                                                             residual_dropout  = residual_dropout)

        # Define the feedforward layer to add non-linearity to the model...
        self.ff_layer = FeedForward(embd_size, dropout = feedforward_dropout)

        # Define layers to optimize model training...
        self.layer_norm_pre_self_att    = nn.LayerNorm(embd_size)
        self.layer_norm_pre_cross_att   = nn.LayerNorm(embd_size)
        self.layer_norm_pre_feedforward = nn.LayerNorm(embd_size)


    def forward(self, embd_q, embd_k, embd_v):
        """
        Arguments:
            embd_q : (B, M, E), query token embedding
            embd_k : (B, M, E), key   token embedding
            embd_v : (B, M, E), value token embedding

        In DETR's design, embd_k and embd_v are not identical as only embd_k
        accounts for positional encoding.

        Returns:
            out : (B, M, E)
        """
        # ___/ SELF-ATTENTION \___
        # Go through multi-head attention to update nodes in vector space...
        # ...Pre norm
        embd_q_norm = self.layer_norm_pre_self_att(embd_q)

        # ...Attention
        embd_q_self_att = self.multi_head_self_att_layer(embd_q_norm)    # (B, M, E)

        # ...Residual connection (out -> prenorm)
        embd_q_self_att += embd_q

        # ___/ CROSS-ATTENTION \___
        # ...Pre norm
        embd_q_self_att_norm = self.layer_norm_pre_cross_att(embd_q_self_att)
        embd_k_norm          = self.layer_norm_pre_cross_att(embd_k)
        embd_v_norm          = self.layer_norm_pre_cross_att(embd_v)

        # ...Attention
        embd_q_cross_att = self.multi_head_cross_att_layer(embd_q_self_att_norm, embd_k_norm, embd_v_norm)    # (B, M, E)

        # ...Residual connection (out -> prenorm)
        embd_q_cross_att += embd_q_self_att

        # Learn a better embedding representation by introducing non-linearity...
        # ...Pre norm
        embd_q_cross_att_norm = self.layer_norm_pre_feedforward(embd_q_cross_att)    # (B, M, E)

        # ...Feed forward
        embd_q_out = self.ff_layer(embd_q_cross_att_norm)    # (B, M, E)

        # ...Residual connection (out -> prenorm)
        embd_q_out += embd_q_cross_att

        return embd_q_out




class TransformerEncoderDecoder(nn.Module):
    def __init__(self, context_length,
                       memory_length,
                       tok_size,
                       embd_size,
                       num_blocks,
                       num_heads,
                       kv_tok_size         = None,
                       uses_causal_mask    = False,
                       attention_dropout   = 0.0,
                       residual_dropout    = 0.0,
                       feedforward_dropout = 0.0):
        super().__init__()

        # ___/ TOK EMBD \___
        # Embed a patch token...
        ## tok_size = Hp * Wp
        self.tok_embd_layer  = nn.Linear(tok_size,  embd_size)    # (B, M, N) -> (B, M, E)
        ## self.tok_embd_layer = nn.Conv2d(in_channels  = 1,
        ##                                 out_channels = embd_size,
        ##                                 kernel_size  = (Hp, Wp),
        ##                                 stride       = (Hp, Wp))    # (B, M, Hp, Wp) -> (B, M, E)

        if kv_tok_size is not None:
            self.kv_tok_embd_layer = nn.Linear(kv_tok_size, embd_size)    # (B, M, N') -> (B, M, E)

        # Define positional embedding layer to embed each position to a vector space...
        self.pos_embd_layer = nn.Embedding(memory_length, embd_size)

        # Store a positional tensor...
        self.register_buffer('pos_indices', torch.arange(memory_length))

        # ___/ Encoder \___
        # Define the multi head attention layer to update node position in a sub space using an attention head...
        head_size = embd_size // num_heads
        self.encoder = nn.Sequential(*tuple(
            TransformerEncoderBlock(embd_size,
                                    num_heads,
                                    memory_length,
                                    uses_causal_mask    = uses_causal_mask,
                                    attention_dropout   = attention_dropout,
                                    residual_dropout    = residual_dropout,
                                    feedforward_dropout = feedforward_dropout) for _ in range(num_blocks)
        ))

        # ___/ Decoder \___
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(embd_size,
                                    num_heads,
                                    context_length      = context_length,
                                    memory_length       = memory_length,
                                    uses_causal_mask    = uses_causal_mask,
                                    attention_dropout   = attention_dropout,
                                    residual_dropout    = residual_dropout,
                                    feedforward_dropout = feedforward_dropout) for _ in range(num_blocks)
        ])

        # Define object query...
        self.pos_embd_layer_object_query = nn.Embedding(context_length, embd_size)

        # Store a positional tensor for query objects...
        self.register_buffer('query_pos_indices', torch.arange(context_length))


    def forward(self, x):
        """
        N is number of tokens.
        Arguments:
            x : (B, M, Hp * Wp)
        """
        B, M, _ = x.shape

        # ___/ EMBED ALL NODES \___
        embd_x = self.tok_embd_layer(x)    # (B, M) -> (B, M, E)
        pos_embd = self.pos_embd_layer(self.pos_indices[:M])    # (M) -> (M, E)

        embd_x = embd_x + pos_embd    # (B, M, E) + (M, E)    =
                                      # (B, M, E) + (1, M, E) = (B, M, E)

        # ___/ Encoder \___
        # Derive the kv embedding for the decoder query...
        embd_kv = self.encoder(embd_x)

        # Add positional encoding to key only (DETR design)...
        embd_k = embd_kv + self.pos_embd_layer(self.pos_indices[:M])
        embd_v = embd_kv

        # ___/ Decoder \___
        # Go through multi-head attention to update nodes in vector space...
        T = len(self.query_pos_indices)
        embd_q = self.pos_embd_layer_object_query(self.query_pos_indices[:T])
        embd_q = embd_q.repeat(B, *((1,)*embd_q.ndim))
        for block in self.decoder:
            embd_q = block(embd_q, embd_k, embd_v)    # (B, T, E)

        return embd_q
