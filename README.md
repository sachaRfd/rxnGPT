# Mambo GPT: Decoder-only model for chemical reaction experiments

This is a full-reimplementation of a decoder architecture for machine-translation. It includes the following NON-Optimised features:
   - Seperate Positional Encodings for SRC/TGT.
   - SRC/TGT Masked encoding.
   - Bidirectional Attention on SRC tokens - Prefix-LM style.


Please be-aware the inference is non-optimised, so pretty slow when compared to other implementations.

This was a toy-architecture for me to assess how accurate decoders can be for chemical translation.


## Setting up the Environment

To set up the environment required for running the RXN GPT model, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sachaRfd/rxnGPT.git
   cd rxngpt
   ```

2. **Create Conda Environment:**

   Create a conda environment with python 3.12:

   ```bash
   conda create -n rxngpt python=3.12.2 -y
   ```

   Then install dependencies and code:

   ```bash
   conda activate rxngpt
   pip install -e .
   ```

## Training a model:


3. **Training Script:**
   Change model configurations within the `configs directory` script and run:

   ```bash
   python src/rxngpt/train_decoder.py --configs PATH/TO/CONFIG
   ```

4. **Best way of training**

   1. First its best to pretrain the model on all tokens. This means the model learns to predict all next-tokens, regardless of they are src or tgt
   2. Next, once converged, the model can finetuned only on tgt tokens. Meaning the model attends all src tokens, and the loss is only calculated on tgt tokens. This is similar to what would happen at inference, where you would give the model the src and require the tgt output.


## Sampling models:

   Add the path to your trained checkpoint to the config used for training and then run:

   ```bash
   python src/rxngpt/sample_decoder.py --config PATH/TO/CONFIG
   ```




