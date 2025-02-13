import os
import torch
from tqdm import tqdm
from rxngpt.decoder import GPTConfig, GPT
from rxngpt.dataset import get_token2label
import argparse
import yaml


def load_yaml(file_path: str):
    """Load YAML file with automatic type handling."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Your script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config YAML file"
    )
    return parser.parse_args()


def generate_samples(
    model,
    input,
    greedy_decode,
    encoder,
    label2token,
    max_tokens,
    temperature,
    top_k,
    device,
    use_sep_pos_enc=False,
    use_attention_mask=False,
):
    model.eval().to(device)
    start_ids = encoder(input)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    src_tgt_mask = torch.ones_like(x)
    src_tgt_mask[:, -1] = (
        2  # start prod token should have a chemical embeddings of PRODUCT == 2
    )

    src_tgt_mask = src_tgt_mask.to(device)

    if greedy_decode == "beam":
        raise NotImplementedError

    else:
        with torch.no_grad():
            predicted_product, finished = model.generate(
                x,
                src_tgt_mask,
                max_tokens,
                greedy_decode=greedy_decode,
                temperature=temperature,
                top_k=top_k,
                label2token=label2token,
                use_sep_pos_enc=use_sep_pos_enc,
                use_attention_mask=use_attention_mask,
            )
        return predicted_product, finished


def setup_test_data(test_data):
    return [data.split(" ") + ["[BEGP]"] for data in test_data]


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    return model


def main():
    print("Running Sampling Script\n")

    args = parse_args()
    config = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert (
        config["trained_model_path"] is not None
    ), "Please input a model into the 'trained_model_path' variable."

    # Print sampling parameters
    print("-" * 50)
    print(
        f"\nSampling Parameters: \n"
        f"Checkpoint Path: {config['trained_model_path']}\n"
        f"Max New Tokens: {config['max_new_tokens']}\n"
        f"Temperature: {config['temperature']}\n"
        f"Greedy Decode: {config['decoding_algo']}\n"
        f"Top K: {config['top_k']}\n"
        f"Seed: {config['seed']}\n"
        f"Device: {device}\n"
        f"Use Mini: {config['use_mini']}\n"
        f"Use Sep Pos Enc: {config['use_seperate_pos_enc']}\n"
        f"Use Attention Mask: {config['use_attention_mask']}\n"
    )
    print("-" * 50)

    # Output paths
    path_to_save = os.path.dirname(config["trained_model_path"])
    path_to_save_samples = os.path.join(path_to_save, "predicted-tgt.txt")

    if os.path.exists(path_to_save_samples):
        raise FileExistsError(
            f"The file already exists at {path_to_save_samples}. Choose a different path."
        )

    # Load test data
    with open(os.path.join(config["test_data_path"], "src-test.txt"), "r") as test_file:
        test_src = [line.strip() for line in test_file.readlines()]

    test_src = setup_test_data(test_src)

    token2label = get_token2label(path=config["test_data_path"])
    label2token = {v: k for k, v in token2label.items()}
    encode = lambda s: [token2label[c] for c in s]

    # Load model
    model = load_model(config["trained_model_path"], device)

    if config["use_mini"]:
        test_src = test_src[:10]

    # Generate samples
    generated_samples = []
    failed_to_finish = []
    for idx, test_sample in enumerate(tqdm(test_src)):
        predicted_product, finished = generate_samples(
            model=model,
            greedy_decode=config["decoding_algo"],
            input=test_sample,
            encoder=encode,
            label2token=label2token,
            max_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_k=config["top_k"],
            device=device,
            use_sep_pos_enc=["use_seperate_pos_enc"],
            use_attention_mask=config["use_attention_mask"],
        )
        if config["decoding_algo"] == "beam":
            generated_samples.extend(predicted_product)
        else:
            generated_samples.append(predicted_product)

        if finished == 0:
            print("Failed to finish prediction at idx:", idx)
            failed_to_finish.append(idx)

    # Save generated samples
    print("Failed reaction are at following indices:", failed_to_finish)
    print("Saving generated samples...")

    with open(path_to_save_samples, "w") as file:
        for sample in generated_samples:
            line = " ".join(map(str, sample))
            file.write(line + "\n")

    print("Saved Samples at", path_to_save_samples)

    # Log the failed reactions
    log_file_path = os.path.join(path_to_save, "failed_to_finish_reactions.log")
    with open(log_file_path, "w") as log_file:
        log_file.write("Failed reaction indices are at the following indices:\n")
        log_file.write("\n".join(map(str, failed_to_finish)))


if __name__ == "__main__":
    main()
