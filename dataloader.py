# %% [markdown]
# ### Load Data

# %%
import os
import urllib.request

if not os.path.exists("./the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "./the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("./the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

# %%
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i + max_length + 1]))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# %%
import tiktoken
def create_dataloader(text, batch_size, max_length, stride, shuffle = True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset = dataset, 
        batch_size = batch_size,
        shuffle = shuffle, 
        drop_last = drop_last,
        num_workers = num_workers
    )

# %%
dataloader = create_dataloader(
    text = text,
    batch_size = 2,
    max_length = 4,
    stride = 1,
    shuffle = False
)


