from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # -----------------------------------------------------------------------
    """ sam yapping
    Basically we're trying to inject positional information by rotating each PAIR of dimensions in our query/key
    vectors by a determined amount. 
    - We split each head-vector (q, k) into PAIRS of coordinates (D/2)
    - We assign each pair a unique frequency that depends on its index (lower-index=slower, higher=faster)
    - For each position p in the sequence, compute an angle for each pair
    - Convert that angle to a real-valued cos and sin
    - Multiply/rotate each complex coordinate by ^
    - Re-interleave the rotated pairs back into our D-dim tensor

    The goal is that attention scores become a function of the relative distance/positoin between our two
    rotated q and k vectors. It's nice and parameter free :)
    """
    
    # First, we build Build inverse frequency vector
    #   torch.arange(0, D, 2) → [0,2,4,…,D-2], length D/2
    #   inv_freq[i] = 1 / θ^(2i/D)
    # inv_freq: shape (D/2,)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=query.dtype) / head_dim)
    )

    # Next, we can compute angle matrix ω of shape (L, D/2)
    #   positions p = [0,1,...,L-1] shape (L,)
    #   freqs[p,i] = p * inv_freq[i]
    positions = torch.arange(seqlen, device=device, dtype=query.dtype)  # (L,)
    freqs = torch.outer(positions, inv_freq)                           # (L, D/2)

    # Next: Compute cos and sin of ω   (shapes: (L, D/2))
    cos = freqs.cos()
    sin = freqs.sin()

    # Then split each head‐vector into D/2 “complex” pairs
    #   query,key shape: (B, L, H, D)
    #   → (B, L, H, D/2, 2) → unbind → real, imag each (B, L, H, D/2)
    query_real, query_imag = query.float().reshape(*query.shape[:-1], -1, 2).unbind(-1)
    key_real,   key_imag   = key.float().reshape(*key.shape[:-1],   -1, 2).unbind(-1)

    # Broadcasting cos/sin to match that (B, L, H, D/2) real/imag shape
    #   reshape_for_broadcast will view (L, D/2) → (1, L, 1, D/2)
    cos_b = reshape_for_broadcast(cos, query_real)  # → (B, L, H, D/2)
    sin_b = reshape_for_broadcast(sin, query_real)  # → (B, L, H, D/2)

    # Perform ye ole rotation:
    #   (x_r + i x_i) * (cos + i sin)
    # -> real_out = x_r·cos - x_i·sin
    #    imag_out = x_r·sin + x_i·cos
    q_real_rot = query_real * cos_b - query_imag * sin_b   # (B, L, H, D/2)
    q_imag_rot = query_real * sin_b + query_imag * cos_b   # (B, L, H, D/2)
    k_real_rot = key_real * cos_b - key_imag * sin_b       # (B, L, H, D/2)
    k_imag_rot = key_real * sin_b + key_imag * cos_b       # (B, L, H, D/2)

    # Now re‐interleave real/imag into last dimension:
    #   stack -> (B,L,H,D/2,2) -> flatten dim -2 -> (B,L,H, D)
    query_out = torch.stack((q_real_rot, q_imag_rot), dim=-1).flatten(-2)
    key_out   = torch.stack((k_real_rot, k_imag_rot), dim=-1).flatten(-2)

    # And finally cast back to original dtype and return
    return query_out.to(query.dtype), key_out.to(key.dtype)