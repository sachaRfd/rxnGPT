import os
import torch
from tqdm import tqdm
from src.decoder import GPTConfig, GPT
from src.dataset import get_token2label


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
    src_tgt_mask[:, -1] = 2  # start prod token should

    # src_tgt_mask[:, 0] = 1   # First  10 paddings.

    src_tgt_mask = src_tgt_mask.to(device)

    if greedy_decode == "beam":
        predicted_product, finished = model.beam_search_decode(
            x,
            src_tgt_mask,
            label2token=label2token,
            max_new_tokens=max_tokens,
            top_k=top_k,
            use_sep_pos_enc=use_sep_pos_enc,
            use_attention_mask=use_attention_mask,
        )
        assert (
            len(predicted_product) == top_k
        ), f"There was an issue in generating {top_k} samples."
        return predicted_product, None
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
    return [data + ["[BEGP]"] for data in [data.split(" ") for data in test_data]]
    # return [["[PAD]"] * 100 + data + ["[BEGP]"] for data in [data.split(" ") for data in test_data]]

    # return [["[PAD]"] * 100 + data + ["[BEGP]"] for data in [data.split(" ") for data in test_data]]

    # return [["[BEGR]"] + data + ["[ENDR]"] + ["[BEGP]"] for data in [data.split(" ") for data in test_data]]


if __name__ == "__main__":
    print("Running Sampling Script\n")

    # Variables:
    ckpt_path = (
        "exp_prefixLM/prefixLM_MIT_mixed_test_continued/epoch_9_val_loss_0.0420_ckpt.pt"
    )

    max_new_tokens = 300
    temperature = 1.0
    greedy_decode = "greedy"
    top_k = 1
    seed = 1337
    device = "cuda"
    use_mini = False
    use_sep_pos_enc = True
    use_attention_mask = True

    # Print sampling parameters:
    print("-" * 50)
    print(
        "\nSampling Parameters: \n"
        f"Checkpoint Path: {ckpt_path}\n"
        f"Max New Tokens: {max_new_tokens}\n"
        f"Temperature: {temperature}\n"
        f"Greedy Decode: {greedy_decode}\n"
        f"Top K: {top_k}\n"
        f"Seed: {seed}\n"
        f"Device: {device}\n"
        f"Use Mini: {use_mini}\n"
        f"Use Sep Pos Enc: {use_sep_pos_enc}\n"
        f"Use Attention Mask: {use_attention_mask}\n"
    )
    print("-" * 50)

    path_to_save = os.path.dirname(ckpt_path)
    path_to_save_samples = os.path.join(
        path_to_save, "predicted-tgt-with_bidirectinal_src.txt"
    )

    if os.path.exists(path_to_save_samples):
        raise FileExistsError(
            f"The file already exists at {path_to_save_samples}. Choose a different path."
        )

    assert greedy_decode in ["greedy", "beam"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"

    test_path = "data/USPTO/MIT_mixed/"
    # test_path = "data/USPTO/MIT_mixed/"
    with open(os.path.join(test_path, "src-test.txt"), "r") as test_file:
        test_src = [line.strip() for line in test_file.readlines()]

    test_src = setup_test_data(test_src)

    token2label = get_token2label(path=test_path)
    label2token = {v: k for k, v in token2label.items()}
    encode = lambda s: [token2label[c] for c in s]

    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    if use_mini:
        test_src = test_src[:50]

    generated_samples = []
    failed_to_finish = []
    for idx, test_sample in enumerate(tqdm(test_src)):
        predicted_product, finished = generate_samples(
            model=model,
            greedy_decode=greedy_decode,
            input=test_sample,
            encoder=encode,
            label2token=label2token,
            max_tokens=max_new_tokens,
            temperature=1.0,
            top_k=top_k,
            device=device,
            use_sep_pos_enc=use_sep_pos_enc,
            use_attention_mask=use_attention_mask,
        )
        if greedy_decode == "beam":
            generated_samples.extend(predicted_product)
        else:
            generated_samples.append(predicted_product)
            # print("".join(predicted_product))
        if finished == 0:
            print("Failed to finish prediction at idx: ", idx)
            failed_to_finish.append(idx)

    print("Failed reaction are at following indices: ", failed_to_finish)
    print("Saving generated samples...")

    with open(path_to_save_samples, "w") as file:
        for sample in generated_samples:
            line = " ".join(map(str, sample))
            file.write(line + "\n")

    print("Saved Samples at", path_to_save_samples)

    # Log the failed reactions:
    log_file_path = os.path.join(path_to_save, "failed_to_finish_reactions.log")

    with open(log_file_path, "w") as log_file:
        log_file.write("Failed reaction indices are at the following indices:\n")
        log_file.write("\n".join(map(str, failed_to_finish)))
