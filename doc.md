### 1. What are we trying to find out?

You are testing two distinct, foundational theories of modern AI architecture at the exact same time:

* **Objective A: Proving the Scaling Law.** We wanted to mathematically prove that as a neural network gets wider (more parameters), its fundamental understanding of language improves in a predictable, exponential curve, even if it reads the exact same amount of data.
* **Objective B: The GQA "Free Lunch" Hypothesis.** Grouped Query Attention (GQA) saves massive amounts of VRAM by deleting the model's short-term memory (Key/Value heads). The industry hypothesis is that if you take those deleted memory parameters and inject them into the model's reasoning engine (the Feed-Forward Network), the model will perform just as well while running much faster. We wanted to prove if this "free lunch" holds true across different scales, or if tiny models collapse under the pressure.

---

### 2. What are the 8 models?

You built a perfect $4 \times 2$ experimental grid. You trained four sizes of brains, and for each size, you tested two different architectures.

**The Control Group: Multi-Head Attention (MHA)**
These models had a standard 1:1 ratio of memory heads to query heads.

1. **3M MHA:** The absolute baseline.
2. **5M MHA:** The first scaling step.
3. **10M MHA:** The middle-ground curve setter.
4. **20M MHA:** The heaviest, smartest control model.

**The Experimental Group: Grouped Query Attention (GQA)**
These models were intentionally "crippled" by deleting 50% to 75% of their memory heads.

5.  **3M GQA:** The micro-ablation (4:1 ratio).
6.  **5M GQA:** The awkward-math ablation (2:1 ratio).
7.  **10M GQA:** The industry-standard ablation (4:1 ratio).
8.  **20M GQA:** The heavy ablation (4:1 ratio).

---

### 3. What varied between them? (The Variables)

To make this a mathematically bulletproof study, we had to be incredibly strict about what changed and what didn't.

#### **The Independent Variables (What you changed)**

* **Parameter Count:** You scaled the brain size from ~2.75 million up to ~19.67 million. You achieved this by expanding the model's hidden dimension (`dim`), the number of attention heads, and the baseline size of the SwiGLU network.
* **Attention Routing (MHA vs. GQA):** You altered how the model looks up information. You reduced the `n_kv_heads` to force multiple Query heads to share a single Key/Value memory pathway.
* **FFN Dimension (`ffn_dim`):** *This was your masterstroke.* Whenever you deleted KV heads for the GQA models, you precisely expanded the `ffn_dim` to absorb the exact number of lost parameters. This is called **Iso-Parameter Scaling**. It guaranteed that the 10M GQA model had the exact same number of parameters (10,092,544) as the 10M MHA model.

#### **The Control Variables (What you locked down completely)**

If any of these had changed, your entire experiment would be invalid:

* **The Data Diet:** Every single model read exactly **200,000,000 tokens**. None of them looped the dataset, and none of them stopped early.
* **Network Depth:** Every model was strictly locked to **8 layers**. You only scaled the *width* of the network, never the depth.
* **Context Window:** Every model could only "see" **256 tokens** at a time (`max_seq_len`).
* **The Dictionary:** The tokenizer vocabulary size was permanently locked at **8192**.

