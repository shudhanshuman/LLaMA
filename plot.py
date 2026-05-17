import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration & Data Setup ---
mha_paths = {
    "3M": "RUN_1(3M)/train_log.csv",
    "5M": "RUN_3(5M)/train_log.csv",
    "10M": "RUN_4(10M)/train_log.csv",
    "20M": "RUN_2(20M)/train_log.csv"
}

gqa_paths = {
    "3M": "RUN_8(3M_GQA)/train_log.csv", 
    "5M": "RUN_6(5M_GQA)/train_log.csv",
    "10M": "RUN_5(10M_GQA)/train_log.csv",
    "20M": "RUN_7(20M_GQA)/train_log.csv"
}

# Exact parameter counts for math operations
params_exact = {
    "3M": 2754688,
    "5M": 5115072,
    "10M": 10096896,
    "20M": 19667328
}
params_millions = {k: v / 1_000_000 for k, v in params_exact.items()}

col_names = ['Step', 'Train_Loss', 'Val_Loss', 'LR', 'Time_ms', 'Tok_Sec']
TOKENS_PER_STEP = 8192  # 32 micro_batch * 256 seq_len

os.makedirs("Visualizations", exist_ok=True)

# Helper function to load and process data
def load_and_process(paths_dict):
    data = {}
    for label, path in paths_dict.items():
        if os.path.exists(path):
            df = pd.read_csv(path, names=col_names)
            
            # Clean data: drop NaN validation losses and step 0 for log plots
            df_val = df[df['Val_Loss'].notna()].copy()
            df_val = df_val[df_val['Step'] > 0]
            
            # Feature Engineering: Tokens Seen and Compute (FLOPs)
            # FLOPs formula: C ≈ 6 * N * T
            df_val['Tokens_Seen'] = df_val['Step'] * TOKENS_PER_STEP
            df_val['FLOPs'] = 6 * params_exact[label] * df_val['Tokens_Seen']
            
            data[label] = df_val
        else:
            print(f"Could not find {path}")
    return data

mha_data = load_and_process(mha_paths)
gqa_data = load_and_process(gqa_paths)

colors = {"3M": "#4C72B0", "5M": "#DD8452", "10M": "#55A868", "20M": "#C44E52"}

# ==========================================
# GRAPH 1: Loss vs Tokens Seen
# ==========================================
plt.figure(figsize=(12, 7))
for label, df in mha_data.items():
    plt.plot(df['Tokens_Seen'], df['Val_Loss'], color=colors[label], linestyle='-', linewidth=2, label=f"MHA {label}")
for label, df in gqa_data.items():
    plt.plot(df['Tokens_Seen'], df['Val_Loss'], color=colors[label], linestyle='--', linewidth=2, label=f"GQA {label}")

plt.title('Validation Loss vs. Tokens Seen', fontsize=16, fontweight='bold')
plt.xlabel('Tokens Seen', fontsize=14)
plt.ylabel('Validation Loss', fontsize=14)
plt.legend(fontsize=10, ncol=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Visualizations/1_Loss_vs_Tokens.png", dpi=300)

# ==========================================
# GRAPH 2: Training Loss vs Compute (FLOPs)
# ==========================================
plt.figure(figsize=(12, 7))
for label, df in mha_data.items():
    plt.plot(df['FLOPs'], df['Val_Loss'], color=colors[label], linestyle='-', linewidth=2, label=f"MHA {label}")
for label, df in gqa_data.items():
    plt.plot(df['FLOPs'], df['Val_Loss'], color=colors[label], linestyle='--', linewidth=2, label=f"GQA {label}")

plt.title('Scaling Law: Loss vs Compute (FLOPs)', fontsize=16, fontweight='bold')
plt.xlabel('Compute (Total FLOPs)', fontsize=14)
plt.ylabel('Validation Loss', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(fontsize=10, ncol=2)
plt.tight_layout()
plt.savefig("Visualizations/2_Loss_vs_FLOPs.png", dpi=300)

# ==========================================
# GRAPH 3: Validation Loss vs Model Size (Pareto)
# ==========================================
plt.figure(figsize=(10, 6))
mha_x, mha_y = [], []
for label, df in mha_data.items():
    mha_x.append(params_millions[label])
    mha_y.append(df['Val_Loss'].iloc[-1])

gqa_x, gqa_y = [], []
for label, df in gqa_data.items():
    gqa_x.append(params_millions[label])
    gqa_y.append(df['Val_Loss'].iloc[-1])

# Plot Empirical Data
plt.plot(mha_x, mha_y, marker='o', markersize=10, linestyle='', linewidth=2, color='#4C72B0', label='MHA Baseline')
if gqa_x: # Ensure we have data to plot
    plt.plot(gqa_x, gqa_y, marker='*', markersize=14, linestyle='', color='#C44E52', label='GQA Ablation')

# Line of Best Fit (Math Magic) - using MHA for the fit
if len(mha_x) > 1:
    slope_mha, intercept_mha = np.polyfit(np.log10(mha_x), np.log10(mha_y), 1)
    a_mha = 10**intercept_mha
    x_fit_mha = np.linspace(min(mha_x) * 0.9, max(mha_x) * 1.1, 100)
    y_fit_mha = a_mha * (x_fit_mha ** slope_mha)

    plt.plot(x_fit_mha, y_fit_mha, color='#4C72B0', linestyle='--', linewidth=2, label='MHA Power-Law Fit')
    plt.text(0.95, 0.95, rf"MHA: $Loss = {a_mha:.3f} \cdot s^{{{slope_mha:.3f}}}$", transform=plt.gca().transAxes, 
             fontsize=13, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#4C72B0'))

# Line of Best Fit (Math Magic) - using GQA for the fit
if len(gqa_x) > 1:
    slope_gqa, intercept_gqa = np.polyfit(np.log10(gqa_x), np.log10(gqa_y), 1)
    a_gqa = 10**intercept_gqa
    x_fit_gqa = np.linspace(min(gqa_x) * 0.9, max(gqa_x) * 1.1, 100)
    y_fit_gqa = a_gqa * (x_fit_gqa ** slope_gqa)

    plt.plot(x_fit_gqa, y_fit_gqa, color='#C44E52', linestyle='--', linewidth=2, label='GQA Power-Law Fit')
    # Offset the text slightly below the MHA text box
    plt.text(0.95, 0.85, rf"GQA: $Loss = {a_gqa:.3f} \cdot s^{{{slope_gqa:.3f}}}$", transform=plt.gca().transAxes, 
             fontsize=13, va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#C44E52'))

plt.title('Pareto Frontier: Final Validation Loss vs. Model Size', fontsize=16, fontweight='bold')
plt.xlabel('Model Parameters (Millions)', fontsize=14)
plt.ylabel('Final Validation Loss', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=11, loc='lower left')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Visualizations/3_Loss_vs_Model_Size.png", dpi=300)

# ==========================================
# GRAPH 4: Perplexity Comparison
# ==========================================
plt.figure(figsize=(10, 6))
mha_ppl = [np.exp(y) for y in mha_y]
gqa_ppl = [np.exp(y) for y in gqa_y]

plt.plot(mha_x, mha_ppl, marker='o', markersize=10, linestyle='-', linewidth=2, color='#4C72B0', label='MHA Perplexity')
if gqa_x: # Ensure we have data to plot
    plt.plot(gqa_x, gqa_ppl, marker='*', markersize=14, linestyle='--', linewidth=2, color='#C44E52', label='GQA Perplexity')

# Annotate points
for i, txt in enumerate(mha_ppl):
    plt.annotate(f"{txt:.2f}", (mha_x[i], mha_ppl[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#4C72B0')
for i, txt in enumerate(gqa_ppl):
    plt.annotate(f"{txt:.2f}", (gqa_x[i], gqa_ppl[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='#C44E52')

plt.title('Language Fluency: Perplexity vs. Model Size', fontsize=16, fontweight='bold')
plt.xlabel('Model Parameters (Millions)', fontsize=14)
plt.ylabel('Final Validation Perplexity (Lower is Better)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Visualizations/4_Perplexity_Comparison.png", dpi=300)

# ==========================================
# GRAPH 5: Hardware Efficiency (Tokens/Sec)
# ==========================================
plt.figure(figsize=(10, 6))
labels = list(params_millions.keys())
x = np.arange(len(labels))
width = 0.35

mha_tps = []
gqa_tps = []

# Safely extract TPS handling missing files gracefully
for lbl in labels:
    if lbl in mha_paths and os.path.exists(mha_paths[lbl]):
        mha_tps.append(pd.read_csv(mha_paths[lbl], names=col_names)['Tok_Sec'].iloc[10:].mean())
    else:
        mha_tps.append(0)  # Default to 0 if data isn't collected yet
        
    if lbl in gqa_paths and os.path.exists(gqa_paths[lbl]):
        gqa_tps.append(pd.read_csv(gqa_paths[lbl], names=col_names)['Tok_Sec'].iloc[10:].mean())
    else:
        gqa_tps.append(0)  # Default to 0 if data isn't collected yet

bars1 = plt.bar(x - width/2, mha_tps, width, label='MHA', color='#4C72B0')
bars2 = plt.bar(x + width/2, gqa_tps, width, label='GQA', color='#DD8452')

plt.title('Hardware Efficiency: Tokens Per Second (Apple M2)', fontsize=16, fontweight='bold')
plt.xlabel('Model Size', fontsize=14)
plt.ylabel('Average Tokens per Second', fontsize=14)
plt.xticks(x, labels)
plt.legend()

# Add text labels on bars, skipping 0 heights
for bar in bars1 + bars2:
    height = bar.get_height()
    if height > 0:  # Only annotate bars that exist
        plt.text(bar.get_x() + bar.get_width()/2, height + 100, f"{int(height)}", ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Visualizations/5_Hardware_Throughput.png", dpi=300)

print("\n🚀 All 5 comprehensive graphs generated successfully in the 'Visualizations' folder!")