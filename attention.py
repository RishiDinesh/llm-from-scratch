# %% [markdown]
# ### Simple attention mechanism without trainable weights

# %%
import torch

inputs = torch.tensor([
    [0.43, 0.15, 0.89], # your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55] # step
])

attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim = -1)
context_vectors = attn_weights @ inputs
context_vectors

# %% [markdown]
# ### Attention mechanism with trainable weights (Scaled dot product attention)

# %%
torch.manual_seed(123)

d_in = inputs.shape[1] # input embedding size
d_out = 2 # output embedding size, usually same as d_in

Wq = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
Wk = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
Wv = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

queries = inputs @ Wq
keys = inputs @ Wk
values = inputs @ Wv

d_k = keys.shape[1] # dimension of keys

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / (d_k ** 0.5) , dim = -1)
context_vectors = attn_weights @ values
context_vectors

# %% [markdown]
# ### Self-attention class

# %%
from torch import nn

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        self.Wq = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias = qkv_bias)
    
    def forward(self, x):
        queries = self.Wq(x)
        keys = self.Wk(x)
        k_d = keys.shape[-1]
        values = self.Wv(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / (k_d ** 0.5) , dim = -1)
        context_vec = attn_weights @ values
        return context_vec

# %% [markdown]
# ### Causal Attention class

# %%
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.Wq = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.mask = torch.triu(torch.ones(
            context_length, context_length
        ), diagonal = 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)
        attn_scores = queries @ keys.transpose(1, 2) # we transpose dims 1 and 2, keeping batch dim fixed
        mask = self.mask.bool()[:num_tokens, :num_tokens] # truncate to num_tokens
        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = self.dropout(
            torch.softmax(attn_scores /(keys.shape[-1] ** 0.5), dim = -1)
        )

        return attn_weights @ values

# %% [markdown]
# ### Multi-head Attention

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout, qkv_bias = False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
		# d_out is the dimension of the final concatenated context vector
		# head_dim is the dimension of the context vector given by each attention head
		# d_out must be divisible by num_heads
        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.Wk = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Wq = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # use a linear layer to combine head outputs
        self.register_buffer(
            name = "mask",
            tensor = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # linear projections: of shape (b, num_tokens, d_out)
        keys = self.Wk(x)
        queries = self.Wq(x)
        values = self.Wv(x)
        
        # d_out can be reshaped into num_heads * head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # calculate attn scores : keys is transposed to (b, num_heads, head_dim, num_tokens)
        # attn_scores shape : (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3) 
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1]**0.5), dim = -1)
        attn_weights = self.dropout(attn_weights)

        # this is of shape (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values
        # get to shape (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)
        # combine num_heads * head_dim to d_out
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # add an optional linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec

# %%
batch = torch.stack([inputs, inputs], dim = 0)

attention = MultiHeadAttention(
    d_in = 3, 
    d_out = 2,
    num_heads = 2,
    context_length = 6,
    dropout = 0.0
)

res = attention(batch)
print(res, res.shape, sep="\n")


