import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

run_paths = {
    "3M Params": "RUN_1(3M)/train_log.csv",
    "5M Params": "RUN_3(5M)/train_log.csv",
    "10M Params": "RUN_4(10M)/train_log.csv",
    "20M Params": "RUN_2(20M)/train_log.csv"
}

param_counts = {
    "3M Params": 3.1,
    "5M Params": 5.12,
    "10M Params": 10.09,
    "20M Params": 19.67
}

col_names = ['Step', 'Train_Loss', 'Val_Loss', 'LR', 'Time_ms', 'Tok_Sec']

os.makedirs("Visualizations", exist_ok=True)

dataframes = {}
for label, path in run_paths.items():
    if os.path.exists(path):
        dataframes[label] = pd.read_csv(path, names=col_names)
    else:
        print(f"⚠️ Warning: Could not find {path}")

if not dataframes:
    print("❌ No data found. Exiting.")
    exit()

# --- Plot 1: Validation Loss Learning Curves ---
plt.figure(figsize=(12, 7))
for label, df in dataframes.items():
    # Only plot the rows where we actually ran validation (every 200 steps)
    val_data = df[df['Val_Loss'].notna()]
    plt.plot(val_data['Step'], val_data['Val_Loss'], label=label, linewidth=2)

plt.title('Validation Loss Scaling Law (200M Tokens)', fontsize=16, fontweight='bold')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Validation Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Visualizations/1_Learning_Curves.png", dpi=300)
print("✅ Saved 1_Learning_Curves.png")


# --- Plot 2: Hardware Throughput (M2 Efficiency) ---
plt.figure(figsize=(10, 6))
labels = []
avg_tps = []

for label, df in dataframes.items():
    labels.append(label)
    # Ignore the first few steps to let throughput stabilize
    stable_tps = df['Tok_Sec'].iloc[10:].mean()
    avg_tps.append(stable_tps)

bars = plt.bar(labels, avg_tps, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
plt.title('Apple M2 Throughput by Model Size', fontsize=16, fontweight='bold')
plt.xlabel('Model Architecture', fontsize=14)
plt.ylabel('Average Tokens per Second', fontsize=14)

# Add the numbers on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 200, f"{int(yval):,}", ha='center', va='bottom', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Visualizations/2_Hardware_Throughput.png", dpi=300)
print("✅ Saved 2_Hardware_Throughput.png")




# --- Plot 3: The Pareto Frontier (Final Loss vs Scale) ---
plt.figure(figsize=(10, 6))
x_params = []
y_final_loss = []

# Gather the data
for label, df in dataframes.items():
    x_params.append(param_counts[label])
    # Grab the very last validation loss recorded
    final_loss = df['Val_Loss'].iloc[-1]
    y_final_loss.append(final_loss)
    
    # Annotate the specific points
    plt.annotate(f"{final_loss:.3f}", (param_counts[label], final_loss), 
                 textcoords="offset points", xytext=(0,10), ha='center')

# 1. Plot the actual empirical data
plt.plot(x_params, y_final_loss, marker='o', markersize=10, linestyle='-', linewidth=2, color='#4C72B0', label='Empirical Loss')

# 2. THE ONE-LINE MATH FIT: Calculate slope and intercept in log-log space
slope, intercept = np.polyfit(np.log10(x_params), np.log10(y_final_loss), 1)

# Calculate 'a' (the constant) from the log intercept
a = 10**intercept

# 3. Generate points for the smooth theoretical dotted line
x_fit = np.linspace(min(x_params) * 0.9, max(x_params) * 1.1, 100)
y_fit = a * (x_fit ** slope)

# 4. Plot the dotted power-law fit line
plt.plot(x_fit, y_fit, color='#DD8452', linestyle='--', linewidth=2, label='Power-Law Fit')

# 5. Add the formula text box in the top right corner
formula_text = rf"$\mathrm{{Loss}} = {a:.3f} \cdot s^{{{slope:.3f}}}$"
plt.text(0.95, 0.95, formula_text, transform=plt.gca().transAxes, 
         fontsize=14, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

# Chart formatting
plt.title('Pareto Frontier: Final Loss vs. Parameter Count', fontsize=16, fontweight='bold')
plt.xlabel('Model Parameters / Scale ($s$) in Millions', fontsize=14)
plt.ylabel('Final Validation Loss', fontsize=14)
plt.legend(fontsize=12, loc='lower left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the updated plot
plt.savefig("Visualizations/3_Pareto_Frontier.png", dpi=300)
print("✅ Saved 3_Pareto_Frontier.png (with Power-Law Fit)")

# --- Plot 4: Log-Log Scaling Law (Power Law Emergence) ---
plt.figure(figsize=(12, 7))

for label, df in dataframes.items():
    val_data = df[df['Val_Loss'].notna()]
    
    # CRITICAL MATH FIX: We must filter out Step 0. 
    # The logarithm of 0 is mathematically undefined and will crash the plot.
    val_data = val_data[val_data['Step'] > 0]
    
    plt.plot(val_data['Step'], val_data['Val_Loss'], label=label, linewidth=2)

plt.title('Log-Log Validation Loss Scaling Law', fontsize=16, fontweight='bold')
plt.xlabel('Training Steps (Log Scale)', fontsize=14)
plt.ylabel('Validation Loss (Log Scale)', fontsize=14)

# This is where the magic happens: converting axes to logarithmic scale
plt.xscale('log')
plt.yscale('log')

# Adding minor gridlines helps visualize the log scale intervals
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("Visualizations/4_LogLog_Learning_Curves.png", dpi=300)
print("✅ Saved 4_LogLog_Learning_Curves.png")

print("\n🚀 All visualizations successfully generated in the 'Visualizations' folder!")