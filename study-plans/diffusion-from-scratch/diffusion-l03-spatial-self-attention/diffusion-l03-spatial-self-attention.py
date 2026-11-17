import math
import torch
import torch.nn.functional as F

def spatial_self_attention(x, params, num_groups, num_heads):
    """
    Returns: x plus the multi-head attention output, shape (B, C, H, W)
    """
    B, C, H, W = x.shape
    d = C // num_heads
    h = F.group_norm(x, num_groups, params["norm_weight"], params["norm_bias"], eps=1e-5)
    tokens = h.flatten(2).transpose(1, 2)

    def split(t):
        return t.view(B, H * W, num_heads, d).transpose(1, 2)

    q = split(F.linear(tokens, params["q_weight"], params["q_bias"]))
    k = split(F.linear(tokens, params["k_weight"], params["k_bias"]))
    v = split(F.linear(tokens, params["v_weight"], params["v_bias"]))
    attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), dim=-1)
    merged = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
    out = F.linear(merged, params["out_weight"], params["out_bias"])
    return x + out.transpose(1, 2).reshape(B, C, H, W)