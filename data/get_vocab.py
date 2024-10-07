"""

Script to get Vocab from files: 

"""
from typing import Dict, List
import os
import json
import matplotlib.pyplot as plt


def learn_label_encoding(tokenized_inputs: List[List[str]]) -> Dict[str, int]:
    """Learn a label encoding from a tokenized dataset. The padding token,
    `"[PAD]"` is always assigned the label 0.

    Parameters
    ----------
    tokenized_inputs : List[List[str]]
        SMILES of the molecules in the dataset, tokenized into a list
        of tokens.

    Returns
    -------
    Dict[str, int]
        A dictionary mapping SMILES tokens to integer labels.
    """
    token2label = dict()
    token2label["[PAD]"] = len(token2label)
    for inp in tokenized_inputs:
        for token in inp:
            if token not in token2label:
                token2label[token] = len(token2label)

    tokens_to_add = ["[BEGR]", "[BEGP]", "[ENDR]", "[ENDP]"]
    for token in tokens_to_add:
        if token not in token2label:
            token2label[token] = len(token2label)

    return token2label


def count_tokens(tokenized_inputs):
    token_count = dict()
    for inp in tokenized_inputs:
        # print(inp)
        # break
        for token in inp.split(" "):
            if token not in token_count:
                token_count[token] = 1
            else:
                token_count[token] += 1
    return token_count


if __name__ == "__main__":
    print("Running Vocab script")

    # Load data from path:
    path_to_data = "data/one_to_infinite_step_with_reaction_classes_fixed"
    assert os.path.isdir(path_to_data), "Path to data is not a directory"

    files = ["train", "val", "test"]

    files_format = [f"{path_to_data}/src-{file}.txt" for file in files] + [
        f"{path_to_data}/tgt-{file}.txt" for file in files
    ]

    for file in files_format:
        assert os.path.isfile(file), f"File: {file} does not exist"

    # Load data from files:
    data = []
    for file in files_format:
        with open(file, "r") as f:
            data += [line.strip() for line in f.readlines()]

        print(f"Loaded data from file: {file}: \t length: {len(data)}")

    print(f"Total data loaded: {len(data)}")

    # Create the vocab dict:
    token2label = learn_label_encoding([d.split() for d in data])

    print(f"Vocab created: {len(token2label)} tokens found.")

    # Save dict as token2label.json:
    path_to_save = os.path.join(path_to_data, "token2label.json")
    with open(path_to_save, "w") as f:
        json.dump(token2label, f)

    # Get a count of each token in data:
    token_count = count_tokens(data)

    # Convert token count values to a list
    token_frequencies = list(token_count.values())

    # Calculate total count of tokens
    total_tokens = sum(token_frequencies)

    # Histogram where x-axis is the token type and y-axis is the frequency
    plt.bar(token_count.keys(), [freq / total_tokens for freq in token_frequencies])
    plt.xlabel("Token type")
    plt.ylabel("Frequency")
    plt.title("Histogram of token frequencies")

    # Save:
    plt.savefig(os.path.join(path_to_data, "token_frequency_histogram.png"))
