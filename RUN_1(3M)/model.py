import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time
import sys
import math
import torch.optim as optim
import os
import numpy as np


torch.set_float32_matmul_precision("high")

torch.manual_seed(1337)
torch.autograd.set_detect_anomaly(False)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)



@dataclass
class ModelArgs:
    dim: int = 128
    n_layers: int = 8
    n_heads: int = 4
    n_kv_heads: int = 4
    ffn_dim: int = 384

    vocab_size: int = 8192
    max_seq_len: int = 256

    norm_eps: float = 1e-5

    micro_batch: int = 32
    accum_steps: int = 1  
    global_batch: int = 32
    

    device: str = "mps" if torch.backends.mps.is_available() else "cpu"



def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000.0):
    assert head_dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_embeddings(x, freqs_complex):
    orig_dtype = x.dtype
    # view_as_complex only supports float, double, half. Cast bfloat16 to float32 for this op
    if x.dtype == torch.bfloat16:
        x_float = x.float()
        x_complex = torch.view_as_complex(x_float.reshape(*x.shape[:-1], -1, 2))
    else:
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_complex * freqs_complex).reshape(*x.shape)
    return x_out.to(orig_dtype)



class GQA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads

        assert args.n_heads % args.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = self.dim // self.n_heads
        self.h_factor = self.n_heads // self.n_kv_heads

        self.c_attn = nn.Linear(
            self.dim,
            self.dim + 2 * self.n_kv_heads * self.head_dim,
            bias=False
        )

        self.c_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.c_proj.FLAG = True

    def forward(self, x, freqs_complex):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q = qkv[..., :self.dim]
        k = qkv[..., self.dim:self.dim + self.n_kv_heads * self.head_dim]
        v = qkv[..., self.dim + self.n_kv_heads * self.head_dim:]

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_embeddings(q, freqs_complex)
        k = apply_rotary_embeddings(k, freqs_complex)


        k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.h_factor, T, self.head_dim)
        k = k.reshape(B, self.n_heads, T, self.head_dim)

        v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.h_factor, T, self.head_dim)
        v = v.reshape(B, self.n_heads, T, self.head_dim)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.c_proj(y)



class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.ffn_dim, bias=False)
        self.w2 = nn.Linear(args.dim, args.ffn_dim, bias=False)
        self.w3 = nn.Linear(args.ffn_dim, args.dim, bias=False)
        self.w3.FLAG = True

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))



class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ln_1 = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.attn = GQA(args)
        self.ln_2 = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = MLP(args)

    def forward(self, x, freqs_complex):
        x = x + self.attn(self.ln_1(x), freqs_complex)
        x = x + self.mlp(self.ln_2(x))
        return x



class LLAMA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.tok_embedding = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])

        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.output.weight = self.tok_embedding.weight
        
        self.freqs_complex = precompute_theta_pos_frequencies(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.device
        )
        self.freqs_complex = self.freqs_complex.unsqueeze(0).unsqueeze(0)

        self.apply(lambda module: LLAMA.init_weights(module, args))
        
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        x = self.tok_embedding(idx)

        freqs = self.freqs_complex[:, :, :T].to(x.device)

        for layer in self.layers:
            x = layer(x, freqs)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss
    
    
    @staticmethod
    def init_weights(module, args):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "FLAG", False):
                std *= (2 * args.n_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)
    

class DataLoader:
    def __init__(self, B, T, file_path, device):
        self.B = B
        self.T = T
        self.device = device
        
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')
        
        self.num_batches = len(self.data) // (B * T)
        print(f"Loaded {file_path}: {len(self.data):,} tokens | {self.num_batches:,} batches")
        self.ptr = 0

    def get_batch(self):
        start = self.ptr * self.B * self.T
        end = start + (self.B * self.T) + 1
        
        chunk = self.data[start:end]
        
        if len(chunk) < (self.B * self.T) + 1:
            chunk = np.append(self.data[start:], self.data[:(self.B * self.T) + 1 - len(chunk)])
        chunk_tensor = torch.tensor(chunk.astype(np.int64), dtype=torch.long)
        
        x = chunk_tensor[:-1].view(self.B, self.T)
        y = chunk_tensor[1:].view(self.B, self.T)
        
        self.ptr += 1
        if self.ptr >= self.num_batches:
            self.ptr = 0
            
        return x, y



@torch.no_grad()
def evaluate_validation_set(model, val_data, eval_iters=20):
    model.eval()
    losses = []
    val_data.ptr = 0 
    
    for _ in range(eval_iters):
        x, y = val_data.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
        
    model.train()
    return sum(losses) / len(losses)


device = "mps" if torch.backends.mps.is_available() else "cpu"
log_file = "train_log.csv"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
save_interval = 2048

model = LLAMA(ModelArgs())
model.to(device)

print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = []
nodecay_params = []
for n, p in param_dict.items():
    if "tok_embedding" in n:
        nodecay_params.append(p)
    elif p.dim() >= 2:
        decay_params.append(p)
    else:
        nodecay_params.append(p)
optim_groups = [
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': nodecay_params, 'weight_decay': 0.0}
]

tokenizer_path = "../tokenizer.json"
train_path = "../DataSet/train.bin"
val_path = "../DataSet/val.bin"



max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 1024 
max_steps = 24414
accumulation_steps = model.args.global_batch // model.args.micro_batch

B ,T = model.args.micro_batch , model.args.max_seq_len

train_data = DataLoader(B, T, file_path=train_path, device=device)
val_data = DataLoader(B, T, file_path=val_path, device=device)

optimizer = optim.AdamW(optim_groups, lr=max_lr, foreach=True , betas=(0.9, 0.95) , eps=1e-8)
scaler = torch.amp.GradScaler("mps")


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


args = model.args
logs = []

optimizer_step = optimizer.step
args = model.args

torch.mps.synchronize()
t0 = time.time()

for i in range(max_steps):
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro_step in range(accumulation_steps):
        x, y = train_data.get_batch()
        x , y = x.to(device), y.to(device)

        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss = loss / accumulation_steps

        if not torch.isfinite(loss):
            print("Loss exploded")
            break
        loss_val = loss.detach()
        loss_accum += loss_val
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    scaler.step(optimizer)
    scaler.update()

    


    if i % 200 == 0 or i == max_steps - 1:
        
        torch.mps.synchronize()
        t1 = time.time()
        
        
        steps_passed = 1 if i == 0 else 200
        if i == max_steps - 1 and (max_steps - 1) % 200 != 0:
            steps_passed = (max_steps - 1) % 200
            
        
        dt = ((t1 - t0) * 1000) / steps_passed
        tokens_per_sec = (train_data.B * train_data.T * accumulation_steps * steps_passed) / (t1 - t0)
        
        val_loss = evaluate_validation_set(model, val_data)
        print(f"step {i:4d} | train loss {loss_accum.item():.4f} | val loss {val_loss:.4f} | lr {lr:.4e} | time {dt:.2f}ms | tok/sec {tokens_per_sec:.2f}")
        
        logs.append(f"{i},{loss_accum.item():.4f},{val_loss:.4f},{lr:.6e},{dt:.2f},{tokens_per_sec:.2f}\n")

        with open(log_file, "a") as f:
            f.writelines(logs)
        logs = []
        
        torch.mps.synchronize()
        t0 = time.time()
        
    if (i > 0 and i % save_interval == 0) or (i == max_steps - 1):
        checkpoint_path = os.path.join(checkpoint_dir, f"gpt_ckpt_step_{i:04d}.pt")
        checkpoint = {
            'step': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_accum.item(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"---> Saved checkpoint to {checkpoint_path}")

sys.exit(0)