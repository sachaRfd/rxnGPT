# Mambo GPT: Decoder-only model for chemical reaction experiments

This is a full-reimplementation of a decoder architecture for machine-translation. It includes the following NON-Optimised features: 
   - Seperate Positional Encodings for SRC/TGT.
   - SRC/TGT Masked encoding.
   - Bidirectional Attention on SRC tokens - Prefix-LM style.


Please be-aware the inference is non-optimised, so pretty slow when compared to HF implementations. This was a toy-architecture for me to assess how accurate decoders can be for chemical translation.


## Setting up the Environment

To set up the environment required for running the Mambo GPT model, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/mambo-gpt/mambo-gpt.git
   cd rxngpt
   ```

2. **Create Conda Environment:**
   Use the provided `env.yml` file to create a Conda environment with all the necessary dependencies.
   ```bash
   conda env create -f env.yml
   ```

3. **Activate the Environment:**
   Activate the newly created Conda environment.
   ```bash
   conda activate rxngpt
   pip install -e .
   ```

4. **Verify Installation:**
   Ensure that the environment is set up correctly by running a simple test.
   ```bash
   python test.py
   ```

5. **Usage:**
   Now you're ready to use the RXN GPT model for chemical reaction experiments.

