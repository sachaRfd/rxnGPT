"""
Script to generate a vocabulary from a dataset.

This script reads tokenized SMILES data from a specified dataset directory,
processes the data to extract unique tokens, assigns labels to them,
and saves the vocabulary mapping as a JSON file.

Usage:
    python get_vocab.py /path/to/dataset

Arguments:
    dataset_path (str): Path to the dataset directory containing
                        src-train.txt, src-val.txt, src-test.txt,
                        tgt-train.txt, tgt-val.txt, and tgt-test.txt.

Output:
    A JSON file named 'token2label.json' is saved in the dataset directory,
    containing the vocabulary mapping.
"""

import argparse
import os
import json
from typing import Dict, List


def learn_label_encoding(tokenized_inputs: List[List[str]]) -> Dict[str, int]:
    special_tokens = ["[PAD]", "[BEGR]", "[BEGP]", "[ENDR]", "[ENDP]"]
    unique_tokens = set(token for inp in tokenized_inputs for token in inp)
    sorted_tokens = sorted(unique_tokens)
    token2label = {token: idx for idx, token in enumerate(special_tokens)}
    for token in sorted_tokens:
        if token not in token2label:
            token2label[token] = len(token2label)
    return token2label


def count_tokens(tokenized_inputs):
    token_count = {}
    for inp in tokenized_inputs:
        for token in inp.split(" "):
            token_count[token] = token_count.get(token, 0) + 1
    return token_count


def main():
    parser = argparse.ArgumentParser(description="Generate vocabulary from dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory.")
    args = parser.parse_args()

    path_to_data = args.dataset_path
    assert os.path.isdir(path_to_data), f"Error: {path_to_data} is not a directory."

    files = ["train", "val", "test"]
    files_format = [f"{path_to_data}/src-{file}.txt" for file in files] + [
        f"{path_to_data}/tgt-{file}.txt" for file in files
    ]

    for file in files_format:
        assert os.path.isfile(file), f"Error: {file} does not exist."

    data = []
    for file in files_format:
        with open(file, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            data.extend(lines)
            print(f"Loaded {len(lines)} lines from {file}.")

    print(f"Total data loaded: {len(data)}")
    token2label = learn_label_encoding([d.split() for d in data])
    print(f"Vocab created: {len(token2label)} tokens found.")

    path_to_save = os.path.join(path_to_data, "token2label.json")
    with open(path_to_save, "w") as f:
        json.dump(token2label, f)

    print(f"Vocabulary saved to {path_to_save}")


if __name__ == "__main__":
    main()
