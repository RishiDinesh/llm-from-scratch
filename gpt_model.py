# %%
gpt_config = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# %% [markdown]
# ### Dummy GPT Model with placeholder

# %%
import torch
from torch import nn

class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5):
        super().__init__()
    
    def forward(self, x):
        return x

class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings = config["vocab_size"],
            embedding_dim = config["emb_dim"]
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings = config["context_length"],
            embedding_dim = config["emb_dim"]
        )
        self.dropout = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[
                DummyTransformerBlock(config)
                for _ in range(config["n_layers"])
            ]
        )
        self.norm = DummyLayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            in_features = config["emb_dim"],
            out_features = config["vocab_size"],
            bias = config["qkv_bias"]
        )
    
    def forward(self, x_input: torch.Tensor):
        b, seq_len = x_input.shape
        tok_emb = self.token_embedding(x_input)
        pos_emb = self.pos_embedding(
            torch.arange(seq_len, device = x_input.device)
        )
        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        logits = self.out_head(x)
        return logits

# %%
import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch = torch.stack([
    torch.Tensor(tokenizer.encode(txt1)),
    torch.Tensor(tokenizer.encode(txt2))
], dim = 0)
batch = batch.to(dtype = torch.int)

# %%
model = DummyGPTModel(gpt_config)
res = model(batch)
res.shape

# %% [markdown]
# ### Layer Normalization

# %%
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x: torch.Tensor):
        # x is of shape (b, num_tokens, emb_dim)
        # mean and variance is computed per token embedding vector
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# %% [markdown]
# ### GELU activation function and FeedForward

# %%
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

# %%
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_dim = config["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# %% [markdown]
# ### The Transformer Block

# %%
from attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in = config["emb_dim"],
            d_out = config["emb_dim"],
            num_heads = config["n_heads"],
            context_length = config["context_length"],
            dropout = config["drop_rate"],
            qkv_bias = config["qkv_bias"]
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.dropout(
            self.attention(
                self.norm1(x)
            )
        )
        x = x + shortcut
        shortcut = x
        x = self.dropout(
            self.ff(
                self.norm2(x)
            )
        )
        x = x + shortcut
        return x

# %% [markdown]
# ### GPT Model

# %%
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab, ndim = config["vocab_size"], config["emb_dim"]
        self.config = config
        self.token_embedding = nn.Embedding(vocab, ndim)
        self.pos_embedding = nn.Embedding(config["context_length"], ndim)
        self.dropout = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(config)
                for _ in range(config["n_layers"])
            ]
        )
        self.norm = LayerNorm(ndim)
        self.out_head = nn.Linear(ndim, vocab, bias = config["qkv_bias"])
    
    def forward(self, x_input: torch.Tensor):
        b, seq_len = x_input.shape
        tok_emb = self.token_embedding(x_input)
        pos_emb = self.pos_embedding(
            torch.arange(seq_len, device = x_input.device)
        )
        x = tok_emb + pos_emb
        logits = self.out_head(
            self.norm(
                self.transformer_blocks(
                    self.dropout(x)
                )
            )
        )
        return logits

# %%
model = GPTModel(gpt_config)
logits = model(batch)
logits.shape

# %%
total_params = sum(p.numel() for p in model.parameters())
total_params # 163 million parameters in smallest GPT-2

# %% [markdown]
# ### Generating Text

# %%
def generate_text(model, token_input, max_new_tokens, context_size):
    # token_input is of size (batch, n_tokens)
    for _ in range(max_new_tokens):
        # take the last context_size tokens
        token_input = token_input[:, -context_size:]
        with torch.no_grad():
            logits = model(token_input) # size: (b, n_tokens, vocab)
            # we only want the last output
            logits = logits[:, -1, :] # size: (b, vocab)
            next_token = torch.argmax( # size: (b, 1)
                torch.softmax(logits, dim = -1), dim = -1, keepdim = True
            )
            token_input = torch.cat([token_input, next_token], dim = -1)
    return token_input

# %%
start = "Hello, I am"
tokens = torch.tensor(tokenizer.encode(start))
token_input = torch.unsqueeze(tokens, dim = 0)
print(token_input, token_input.shape)

# %%
model.eval()
out = generate_text(
    model = model,
    token_input = token_input,
    max_new_tokens = 10,
    context_size = gpt_config["context_length"]
)
result = tokenizer.decode(out.squeeze(0).tolist())
print(result)


