"""
File contains the following stuff:
1. RMSNorm class
    - RMS Normalization layer, a nn.Module
2. Attention class
    - Multi-head Grouped Multi-Query Attention (GQSA) layer, a nn.Module
3. FeedForward class
    - FeedForward layer, a nn.Module
4. LlamaLayer class
    - A Transformer block, a nn.Module; Uses the above modules.
5. Llama class
    - The overall model class. Uses the above modules.
6. load_pretrained function
    - Loads model weights
"""
from contextlib import nullcontext
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_llama import LlamaPreTrainedModel, LlamaConfig
from rope import apply_rotary_emb
from utils import *

# Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
# borrowed from the official Llama implementation:
# https://github.com/facebookresearch/llama/blob/main/llama/model.py
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the root mean square normalization. Use Equation 4 under
        Section 4 of https://arxiv.org/abs/1910.07467 as a reference. Add 
        the given epsilon value (self.eps) to the tensor's norm (i.e. inside
        the square root in Equation 4) before normalizing the tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        # Given a [batch, seq_len, hdim] tensor
        # First, take the mean along the embedding dimension [2], which gives us [batch, seq_len]
        # But we want to do a division of our original tensor by this rms tensor later, so we can either
        # unsqueeze manually to get [2,3,1] or we can do this keepdim=True
        # We add the eps for magical...stability... or something.
        rms = torch.sqrt(torch.mean((x**2), dim=-1, keepdim=True) + self.eps)
        # We just normalize our original tensor by the rms tensor; don't mess with scaling here
        return x / rms

    def forward(self, x):
        """
        Apply the root mean square normalizer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        # Normalize the tensor by the RMS (which is basically layer norm but without 0-centering iirc)
        output = self._norm(x.float()).type_as(x)
        # Apply this g_i scaler, which lets us learn a per-channel rescaling (just as in LayerNorm... wait, huh?). 
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len
        self.compute_query = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.compute_key = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_value = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_output = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def compute_query_key_value_scores(self,
                                       query: torch.Tensor,
                                       key: torch.Tensor,
                                       value: torch.Tensor) -> torch.Tensor:
        '''
        Jointly compute Scaled Dot Product Attention (see Section 3.2.1 in
        https://arxiv.org/abs/1706.03762 for details). The query, key, and
        value tensors each have shape (bs, n_local_heads, seqlen, head_dim).
        An optimal implemention will jointly computing attention for multiple
        heads (n_local_heads of them) at once using matrix/tensor operations.

        Make sure to use attention_dropout (self.attn_dropout) on the computed
        attention matrix before applying it to the value tensor.
        '''

        """
        Each of the incoming items are (batch_size, n_heads, max_batch_seqlen, head_dim)
        This is soemtimes referred to as (B,H,T,D)

        So we need to:
        1. Compute the raw attention scores (Q K^T)
        2. Compute the sqrt(d_k)
        3. Apply the pre-softmax causal attention mask
        4. Apply the softmax
        """
        # For each (batch sequence, head) pair, we want a tiny matmul between Q and K
        # So that we end up with a T x T score per matrix head.
        # But remember that we have to transpose our K matrix?
        # In this batch form, this really means swapping the last axis so its shape becomes
        # (B,H,D,T) instead of the usual (B,H,T,D)
        
        # First, we need to swap/transpose the lsat two dims of our key tensor.
        # We could do key = key.transpose(-1, -2)... but I prefer the below syntax. 
        key = key.swapaxes(2,3) # Change (B,H,T,D) -> (B,H,D,T)

        # Next, we do QK^T
        raw_scores = torch.matmul(query, key) # (B,H,T,D) @ (B,H,D,T) -> (B,H,T,T)

        # Let's also get the divisor for our "scaled" attention
        divisor = math.sqrt(self.head_dim) # We could alternatively select the appropriate D dim on our key tensor.
        
        # Now let's get our scaled raw scores by simply dividing our row scores by the divisor scalar.
        scaled_rawscores = raw_scores / divisor # Still (B,H,T,T)

        # Next we need to create and apply our attention mask.
        # this is going to be a TxT matrix of 0s with the upper triangle above the main diagonal being -inf
        T = query.shape[2]
        mask = torch.full(
            (T, T),
            float('-inf'),
            device=query.device
        )
        mask = torch.triu(mask, diagonal=1) # Diagonals and below become 0s
        # But to add this (T,T) mask to our (B,H,T,T) raw scores, we need to unsqueeze it so that it can be broadcasted
        mask = mask.unsqueeze(0).unsqueeze(0) # Now it's (1,1,T,T)

        # Let's now mask our scaled raw scores
        scaled_rawscores_masked = scaled_rawscores + mask # (B,H,T,T) still

        # Finally we can apply the Softmax function to this
        # We're applying the softmax over the "rows" of our scaled rawscores matrix
        # Such that for each row (source), all of the columns (target) attention scores sum to 1.
        attention_scores = torch.softmax(scaled_rawscores_masked, dim=-1) # (B,H,T,T) still

        # Next, we need to apply the attention dropout as per the docstring
        # This will zero out a random subset of our weights with prob=config.dropout during training
        # note that this will do nothing at inference time :)
        attention_scores_dropout = self.attn_dropout(attention_scores)

        # Now we have our attention matrix. 
        # We need to then multiple it by our value matrix! 
        # Remember, softmax(QK^T/sqrt(d_k))V   <--- Don't forget this v part!
        output = torch.matmul(attention_scores_dropout, value) # (B,H,T,T)@(B,H,T,D)->(BH,T,D)
        # Above, PyTorch treated the leading (B,H) as batch dims, and did a matmul over the last two dims
        # (T,T)@(T,D) -> (T,D), so our final one is (B,H,T,D), which is what we originally had too!

        return output

    def forward(
        self,
        x: torch.Tensor
    ):
        '''
        Llama2 uses Grouped-Query Attention. The details of GQA are actually
        not critical to solving this assignment; you are simply asked to
        compute Scaled Dot Product Attention (see above for details). GQA is
        a memory optimization to compute multi-head attention efficiently. See
        Section 2.2 in https://arxiv.org/abs/2305.13245 or
        https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7
        for details.
        '''
        batch_size, seqlen, _ = x.shape

        query = self.compute_query(x)
        key = self.compute_key(x)
        value = self.compute_value(x)
        query = query.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        query, key = apply_rotary_emb(query, key, self.head_dim, self.max_seq_len)

        # Grouped multiquery attention: expand out keys and values.
        # Convert both to:
        # (bs, seqlen, n_local_heads, head_dim)
        key = torch.repeat_interleave(key, dim=2, repeats=self.n_rep)
        value = torch.repeat_interleave(value, dim=2, repeats=self.n_rep)

        # make heads into a batch dimension
        query = query.transpose(1, 2)  #(bs, n_local _heads, seqlen, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = self.compute_query_key_value_scores(query, key, value)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # final projection into the residual stream
        output = self.resid_dropout(self.compute_output(output))
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the SwiGLU activation function (see Section 2 in
        https://arxiv.org/abs/2204.02311
        '''
        return F.silu(self.w1(x)) * self.w3(x)

    def forward(self, x):
        return self.dropout(self.w2(self.SwiGLU(x)))


class LlamaLayer(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            multiple_of=config.multiple_of,
            dropout=config.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.layer_norm_eps)

    def forward(self, x):
        '''
        This is the forward pass of the basic transformer building block. This is a
        modernized version of the block shown on the left of Figure 1 on
        https://arxiv.org/pdf/1706.03762.pdf.

        The transformer block should consist of:
        1) layer normalization of the input (via Root Mean Square layer normalization)
        2) self-attention on the layer-normalized input
        3) a residual connection (i.e., add the input to the output of the self-attention)
        3) layer normalization on the output of the self-attention
        4) a feed-forward network on the layer-normalized output of the self-attention
        5) add a residual connection from the unnormalized self-attention output to the
           output of the feed-forward network
        '''
        """
        Okay, this is a transformer block!
        In Llama 2, it's:
        1) Prenorm (RMSNorm)
        2) Attention
        3) Add to residual stream
        4) Prenorm (RMSNorm)
        5) FFNN
        6) Add to residual stream

        This function takes a tensor x of shape (bs, batch_maxseqlen, hdim)
        This will be referred to as (B,T,C)
        """ 
        h = x

        # Prenormalization via RMSNorm
        # Attention
        # Add to Residual Stream
        h = h + self.attention(self.attention_norm(x))

        # Prenorm via RMSNorm
        # FFNN
        # Add to Residual Stream
        h = h + self.feed_forward(self.ffn_norm(h))

        # Return
        return h

class Llama(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        '''
        You will probably never need to call this function, unless you decide
        to pretrain a Llama model from scratch.
        '''
        super().__init__(config)
        self.params = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(LlamaLayer(layer_id, config))
        self.norm = RMSNorm(config.dim, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('compute_output.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim

        return logits, h

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        We perform this generation using basic temperature sampling. Note that we are not using
        nucleus sampling (i.e. limiting ourselves to sampling from the top-k most probable tokens
        at each timestep), though this is often used in conjunction with temperature sampling,
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        """
        Sam Notes:
        - So idx is a sequence of token indices of shape (bs, seq_len)
            For a single prompt, this would be initialized as (1,T), where T is our prompt length
            The second dimension is the current length of the sequence, and grows by 1 every loop iteration. 
            - NOTE: Both idx and idx_cond are tensors of _token IDs_, not continuous embeddings.
        - Okay, the idX_cond thing is maintaining a SLIDING WINDOW of at most max_seq_len tokens.
            e.g. if our self.params.max_seq_len is 50, then as we generate the (say) 52nd token,
            then we'll keep only hte last 50 tokens as context (2:51, say.)
        - Every time we loop, we:
            - Take the lsat up-to-max-seq-len tokens from each batch row (bs, min(L, maxseqlen))
                - Where L is the current length of the sequence
            - Get the logits for position L [the next token, smapling a new token for each sequence (bs, 1)
            - Concatenate, such that idx becomes (batch_size, L+1)
        - Again, idx is (batch_size, current_length), whereas idx_cond is the CROPPEd version that we actually pass
            to the model at each step. This is going to be (batch_size, max(current_length, max_seq_len))
        """

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond) # We can use self here because we subclass nn.Module, and its __call__ is the thing to do.
            # logits is (bs, current_seq_len, vocab), unless in self.forward targets is None (which it usually is), then it's (bs, 1, vocab)
            logits = logits[:, -1, :] # crop to just the final time step; we only care about logits for the next token, here. 
            # logits is now (bs, vocab); we've removed that "time"/"position" axis; its' just logits for the next token for each sequence in batch, now.


            if temperature == 0.0:
                # select the single most likely index
                """
                Temperature of 0 is basically just greedy decoding.
                This just means that we have to... argmax over the logits (to get the vocab index with the highest logit)
                """
                # Note that we don't HAVE to softmax here, because argmax(softmax(vector)) and argmax(vector) will be the same thing, since softmax is monotonic.
                idx_next = torch.argmax(logits, dim=1) # (B,V) -> (B,) vector
                idx_next = idx_next.unsqueeze(1) # Add dim=1 to make it (B,1)
            else:
                '''
                Perform temperature sampling:
                1) identify  the logits at the final step.
                2) scale (divide) these probabilities by the given temperature.
                3) normalize the scaled logits with a softmax to obtain scaled probabilities.
                4) sample from the scaled probability distribution.

                Note that we are not using top-k sampling/nucleus sampling in this procedure.
                '''
                """
                Note that we have to actually softmax here because we're going to be really sampling, not just argmaxing.
                """
                temp_scaled = logits/temperature # Just scale the logits by temperature (B,V)
                probs = torch.softmax(temp_scaled, dim=1) # Softmax along the vocab dimension of our (B,V)
                idx_next = torch.multinomial(probs, num_samples=1) # Sample from the token probability dist for each seq: (B,1)
                # Above: Note that this indeed does return the INDEX that we'd want (rather than any of the token probs in our second dimension, here)
            
            # regardless of the temperature used... We take that vector of next indices for each sequence
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1) # Plop the new column onto the right side of our idx matrix


        return idx

def load_pretrained(checkpoint: str) -> Llama:
  """
  Loads a pretrained model from a checkpoint file.

  Args:
    checkpoint (str): The path to the checkpoint file.

  Returns:
    Llama: The loaded model.
  """
  device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
  #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
  dtype = "float32"

  torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
  torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
  device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
  ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
  ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

  # init from a model saved in a specific directory
  checkpoint_dict = torch.load(checkpoint, map_location=device)
  config = LlamaConfig(**checkpoint_dict['model_args'])
  model = Llama(config)
  state_dict = checkpoint_dict['model']
  unwanted_prefix = '_orig_mod.'
  for k,v in list(state_dict.items()):
      if k.startswith(unwanted_prefix):
          state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  model.load_state_dict(state_dict, strict=False)
  return model
