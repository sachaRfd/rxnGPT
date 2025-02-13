"""
Adaptation of Karpathy's code for mol generation and chemical reaction prediciton:
"""

import os
import json
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader

from rxngpt.utils import collate_fn_right_padding


def get_token2label(
    path,
    include_eor=False,
):
    """
    Get vocabulary of tokens.

    Can include end of reaction token if necessary.

    """
    out = {}
    token_path = os.path.join(path, "token2label.json")

    with open(token_path, "r") as jsonfile:
        out = json.load(jsonfile)

    if include_eor:
        out["<EOR>"] = 300
    return out


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

    return token2label


class ReactDataset(Dataset):
    def __init__(
        self,
        split="train",
        dataset_path="data/USPTO/MIT_mixed",
        token2label_path: str = "data/USPTO/MIT_mixed",
        use_mini_data: bool = False,
        keep_small_rxn: bool = False,
    ):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}. Use 'train' or 'val', or 'test'."

        self.split = split
        self.token2label = get_token2label(token2label_path)

        if split == "train":
            data_path_src = os.path.join(dataset_path, "src-train.txt")
            data_path_tgt = os.path.join(dataset_path, "tgt-train.txt")

        elif split == "val":
            data_path_src = os.path.join(dataset_path, "src-val.txt")
            data_path_tgt = os.path.join(dataset_path, "tgt-val.txt")
        else:
            data_path_src = os.path.join(dataset_path, "src-test.txt")
            data_path_tgt = os.path.join(dataset_path, "tgt-test.txt")

        with open(data_path_src, "r") as file:
            self.data_src = [line.strip() for line in file.readlines()]
        with open(data_path_tgt, "r") as file:
            self.data_tgt = [line.strip() for line in file.readlines()]

        if keep_small_rxn:
            smaller_than = 15
            idx_to_keep = []
            for r, p in zip(self.data_src, self.data_tgt):
                if (
                    len(r.split(" ")) < smaller_than
                    and len(p.split(" ")) < smaller_than
                ):
                    idx_to_keep.append(True)

            self.data_src = [self.data_src[i] for i in idx_to_keep]
            self.data_tgt = [self.data_tgt[i] for i in idx_to_keep]

        if use_mini_data:
            self.data_src = self.data_src[:10]
            self.data_tgt = self.data_tgt[:10]

        # Run Methods:
        self.add_start_end_tokens()
        self.tokenise_data()

        self.reactions = [r + p for r, p in zip(self.data_src, self.data_tgt)]

        print("Done Processing dataset")

    def add_start_end_tokens(
        self,
    ):
        print("\nAdding start and end tokens to Reactants and Products...")
        # Add start and end tokens:
        self.data_src = [
            smiles.split(" ")  # ["[BEGR]"] + smiles.split(" ") + ["[ENDR]"]
            for smiles in self.data_src  # noqa
        ]
        self.data_tgt = [
            ["[BEGP]"] + smiles.split(" ") + ["[ENDP]"]
            for smiles in self.data_tgt  # noqa
        ]

    def tokenise_data(
        self,
    ):
        print("Tokenising Reactants and Products...")
        self.data_src = [
            [self.token2label[token] for token in tokens]
            for tokens in self.data_src  # noqa
        ]
        self.data_tgt = [
            [self.token2label[token] for token in tokens]
            for tokens in self.data_tgt  # noqa
        ]

    def __len__(self):
        return len(self.data_src)

    def __getitem__(self, index):
        # First let's try predicting the Reactant again after the
        # Mid token is predicted:
        reactants = self.data_src[index]
        reactants = self.data_src[index]
        products = self.data_tgt[index]
        reaction = self.reactions[index]

        # Mask the reactants:
        src_tgt_mask = [1] * len(reactants)
        src_tgt_mask += [2] * len(products)
        src_mask = [1] * len(reactants)
        tgt_mask = [2] * len(products)

        reaction_X = reaction[:-1]
        reaction_y = reaction[1:]
        src_tgt_mask = src_tgt_mask[:-1]

        # Create attention masks:
        full_att_mask = [1] * len(reaction)
        src_att_mask = [1] * len(reactants)
        tgt_att_mask = [1] * len(products)

        return (
            reaction_X,
            reaction_y,
            reactants,
            products,
            src_tgt_mask,
            src_mask,
            tgt_mask,
            full_att_mask,
            src_att_mask,
            tgt_att_mask,
        )
