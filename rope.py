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

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

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
    bs, seq_len, n_heads, head_dim = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf.
    # You may also benefit from https://blog.eleuther.ai/rotary-embeddings/.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).
    second_col = _get_real_coef(bs, seq_len, n_heads, head_dim, theta)
    fourth_col = _get_img_coef(bs, seq_len, n_heads, head_dim, theta)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    query_third_col = torch.zeros(query.shape)
    query_third_col[:, :, :, ::2] = -query_imag
    query_third_col[:, :, :, 1::2] = query_real
    key_third_col = torch.zeros(key.shape)
    key_third_col[:, :, :, ::2] = -key_imag
    key_third_col[:, :, :, 1::2] = key_real

    query_out = query * second_col + query_third_col * fourth_col
    key_out = key * second_col + key_third_col * fourth_col
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out

def _get_real_coef(bs, seq_len, n_heads, head_dim, theta):
    """
    Return the second column described in slide 22
    """
    second_col = None
    for m in range(seq_len):
        col = []
        for ind in range(head_dim):
            i = ind // 2 + 1
            theta_i = theta ** (-2 * (i-1) / head_dim)
            theta_i = torch.Tensor([theta_i])
            col.append(torch.cos(m * theta_i))
        col = torch.Tensor(col)
        col = col[None, :]
        if second_col is None: second_col = col
        else: second_col = torch.cat([second_col, col])

    # repeat this matrix
    second_col = second_col[None, :, None, :]
    second_col = second_col.repeat(bs, 1, n_heads, 1)

    return second_col


def _get_img_coef(bs, seq_len, n_heads, head_dim, theta):
    """
    Return the second column described in slide 22
    """
    fourth_col = None
    for m in range(seq_len):
        col = []
        for ind in range(head_dim):
            i = ind // 2 + 1
            theta_i = theta ** (-2 * (i-1) / head_dim)
            theta_i = torch.Tensor([theta_i])
            col.append(torch.sin(m * theta_i))
        col = torch.Tensor(col)
        col = col[None, :]
        if fourth_col is None: fourth_col = col
        else: fourth_col = torch.cat([fourth_col, col])

    # repeat this matrix
    fourth_col = fourth_col[None, :, None, :]
    fourth_col = fourth_col.repeat(bs, 1, n_heads, 1)

    return fourth_col
