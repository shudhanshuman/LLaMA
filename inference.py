import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time
import argparse
import sys
from tokenizers import Tokenizer

MODEL_PATH = "RUN_7(20M_GQA)/checkpoints/20_GQA.pt"

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_dim: int
    vocab_size: int
    max_seq_len: int
    device: str
    max_batch_size: int = 1
    norm_eps: float = 1e-5


def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000.0):
    assert head_dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_embeddings(x, freqs_complex):
    orig_dtype = x.dtype
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

        self.register_buffer("cache_k", torch.zeros(args.max_batch_size, self.n_kv_heads, args.max_seq_len, self.head_dim))
        self.register_buffer("cache_v", torch.zeros(args.max_batch_size, self.n_kv_heads, args.max_seq_len, self.head_dim))

    def forward(self, x, freqs_complex, start_pos=0):
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

        self.cache_k[:B, :, start_pos:start_pos+T] = k
        self.cache_v[:B, :, start_pos:start_pos+T] = v

        k = self.cache_k[:B, :, :start_pos+T]
        v = self.cache_v[:B, :, :start_pos+T]

        k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.h_factor, start_pos+T, self.head_dim)
        k = k.reshape(B, self.n_heads, start_pos+T, self.head_dim)

        v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.h_factor, start_pos+T, self.head_dim)
        v = v.reshape(B, self.n_heads, start_pos+T, self.head_dim)

        # Avoid non-contiguous tensor copy penalty on M2 as found in extended_analysis.md
        if self.h_factor > 1:
            k = k.contiguous()
            v = v.contiguous()

        is_causal = T > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

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

    def forward(self, x, freqs_complex, start_pos=0):
        x = x + self.attn(self.ln_1(x), freqs_complex, start_pos)
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
        ).unsqueeze(0).unsqueeze(0)

    def forward(self, idx, start_pos=0):
        B, T = idx.shape
        x = self.tok_embedding(idx)
        freqs = self.freqs_complex[:, :, start_pos:start_pos+T].to(x.device)

        for layer in self.layers:
            x = layer(x, freqs, start_pos)

        x = self.norm(x)
        logits = self.output(x)
        return logits


def load_model(checkpoint_path, max_seq_len):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading checkpoint from {checkpoint_path} on {device}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    sd = checkpoint['model_state_dict']

    vocab_size = sd['tok_embedding.weight'].shape[0]
    dim = sd['tok_embedding.weight'].shape[1]

    n_layers = 0
    while f"layers.{n_layers}.attn.c_proj.weight" in sd:
        n_layers += 1

    ffn_dim = sd['layers.0.mlp.w1.weight'].shape[0]
    c_attn_out = sd['layers.0.attn.c_attn.weight'].shape[0]
    kv_dim = (c_attn_out - dim) // 2

    # Standard head dimension inferred
    head_dim = 32
    n_heads = dim // head_dim
    n_kv_heads = kv_dim // head_dim
    
    # Identify model type
    model_type = "MHA" if n_heads == n_kv_heads else f"GQA ({n_heads//n_kv_heads}:1)"

    args = ModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_dim=ffn_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        device=device,
        max_batch_size=1
    )

    model = LLAMA(args)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Detected architecture: {model_type} | dim={dim} | n_layers={n_layers} | "
          f"n_heads={n_heads} | n_kv_heads={n_kv_heads} | ffn_dim={ffn_dim} | Params: {param_count/1e6:.2f}M")

    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model, device


def main():
    parser = argparse.ArgumentParser(description="LLAMA Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("tokenizer.json")
    prompt_tokens = tokenizer.encode(args.prompt).ids
    
    # Ensure cache length can handle the generated sequence
    max_seq_len = len(prompt_tokens) + args.max_tokens + 10
    model, device = load_model(MODEL_PATH, max_seq_len)

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    generated_tokens = []

    print("\nStarting generation...")
    print(args.prompt, end="", flush=True)

    with torch.inference_mode():
        start_time = time.time()
        
        # MPS bfloat16 context
        autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if device == "mps" else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        
        # Prefill phase
        with autocast_ctx:
            logits = model(input_ids, start_pos=0)
            
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token.item())
        
        # Decode the first generated token properly
        prompt_decoded = tokenizer.decode(prompt_tokens)
        first_token_decoded = tokenizer.decode(prompt_tokens + [next_token.item()])[len(prompt_decoded):]
        print(first_token_decoded, end="", flush=True)

        current_id = next_token.unsqueeze(1)
        start_pos = input_ids.size(1)

        gen_start_time = time.time()

        # Track the previous token for sliding window decoding (avoids O(N^2) overhead)
        prev_token = prompt_tokens[-1] if len(prompt_tokens) > 0 else generated_tokens[0]

        # Autoregressive generation phase
        for _ in range(args.max_tokens - 1):
            with autocast_ctx:
                logits = model(current_id, start_pos=start_pos)
            
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            token_id = next_token.item()
            generated_tokens.append(token_id)
            
            # Sliding window decode for O(1) string processing
            prefix_text = tokenizer.decode([prev_token])
            combined_text = tokenizer.decode([prev_token, token_id])
            new_text = combined_text[len(prefix_text):]
            
            print(new_text, end="", flush=True)

            current_id = next_token.unsqueeze(1)
            start_pos += 1
            prev_token = token_id

    end_time = time.time()
    
    print()
    total_time = end_time - start_time
    gen_time = end_time - gen_start_time
    tok_sec = (args.max_tokens - 1) / gen_time if args.max_tokens > 1 else 0
    
    print(f"\n--- Generation Stats ---")
    print(f"Tokens generated: {args.max_tokens}")
    print(f"Generation speed: {tok_sec:.2f} tok/sec (autoregressive phase)")
    print(f"Total time (including prefill): {total_time:.2f} s")

if __name__ == "__main__":
    main()
