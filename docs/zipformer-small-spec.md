# Zipformer-Small (Zipformer-S) Architecture Specification

## 1. Overview and Model Topology
Zipformer is a U-Net-like encoder structure designed for Automatic Speech Recognition (ASR). Unlike the Conformer, which processes sequences at a fixed frame rate, Zipformer downsamples the sequence in the middle stacks to learn temporal representations at varying resolutions efficiently.

- **Total Parameters:** ~30 Million
- **Input Features:** 80-dimensional Mel filter-bank features (extracted on 25ms frames with a 10ms frame shift, representing 100Hz).
- **Optimizer Context:** Uses `ScaledAdam` without standard LayerNorm (relies on proposed `BiasNorm` and learned parameter scales).
- **Activations:** `SwooshR` and `SwooshL` (replacing Swish).

## 2. Global Downsampling & Upsampling Structure
The model passes the input through an initial convolutional embedding layer and then through 6 cascaded encoder stacks operating at different frame rates.

1. **Conv-Embed:** Reduces the initial 100Hz temporal sequence length by a factor of 2, resulting in a **50Hz** embedding sequence.
2. **Stack 1:** Operates at **50Hz**
3. **Stack 2:** Operates at **25Hz** (Downsampled)
4. **Stack 3:** Operates at **12.5Hz** (Downsampled)
5. **Stack 4:** Operates at **6.25Hz** (Downsampled - strongest temporal compression)
6. **Stack 5:** Operates at **12.5Hz** (Upsampled)
7. **Stack 6:** Operates at **25Hz** (Upsampled)

*Note: The output of each stack is padded with zeros or truncated to match the embedding dimension of the next stack. The final encoder output dimension takes the maximum dimension across all stacks (256 for the Small variant).*

## 3. Zipformer-S Stack Configurations
The 6 encoder stacks are configured with the following layer counts and dimensionalities:

| Stack | Layers (Blocks) | Frame Rate | Embedding Dimension | Middle Feed-Forward Hidden Dim |
|-------|-----------------|------------|---------------------|--------------------------------|
| 1     | 2               | 50Hz       | 192                 | 512                            |
| 2     | 2               | 25Hz       | 256                 | 768                            |
| 3     | 2               | 12.5Hz     | 256                 | 768                            |
| 4     | 2               | 6.25Hz     | 256                 | 768                            |
| 5     | 2               | 12.5Hz     | 256                 | 768                            |
| 6     | 2               | 25Hz       | 256                 | 768                            |

## 4. Attention & Convolution Hyperparameters
For the 6 encoder stacks, the parameters governing the Attention and Convolution modules are specific per stack:

- **Number of Attention Heads:** `{4, 4, 4, 8, 4, 4}` (Stack 1 to 6)
- **Convolution Kernel Sizes:** `{31, 31, 15, 15, 15, 31}` (Stack 1 to 6)
- **Attention Query Dimension:** `32` per head (constant across all stacks)
- **Attention Value Dimension:** `12` per head (constant across all stacks)

## 5. Zipformer Block Internal Architecture
To improve efficiency, standard Multi-Head Self-Attention (MHSA) is decoupled into Multi-Head Attention Weight (MHAW) and Self-Attention (SA) components. A single Zipformer block performs the following data-flow:

1. **Attention Weight Calculation (MHAW):** - Computes attention weights once per block to save memory and compute.
2. **Parallel Branching:** - The input goes through a **Feed-Forward Module 1**, followed by a **Non-Linear Attention (NLA)** module (which uses the shared weights from MHAW).
3. **Dual Module Groups:** - The sequence passes through two consecutive groups. Each group consists of:
     - `Self-Attention (SA)` *(re-uses weights from MHAW)* - `1D Convolution`
     - `Feed-Forward Module`
4. **Feed-Forward Dimension Scaling:**
   - There are 3 Feed-Forward (FF) modules inside a single block.
   - **FF 1 (First):** Hidden dim is **3/4** of the Middle FF. (e.g., in Stack 2, 3/4 of 768 = 576)
   - **FF 2 (Middle):** Hidden dim is defined in the table above (e.g., 768).
   - **FF 3 (Last):** Hidden dim is **5/4** of the Middle FF. (e.g., in Stack 2, 5/4 of 768 = 960)
5. **Bypass & Normalization:**
   - Instead of standard residual additions, two learned **Bypass** modules are applied at the middle and the end of the block. They use channel-wise scalar weights to combine input and output dynamically.
   - Normalized at the output using **BiasNorm** to retain absolute sequence length information.
