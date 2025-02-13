"""
Evaluation script of the generated data
"""

from rdkit import Chem
import pandas as pd
import os


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ""


def get_rank(row, base, max_rank):
    for i in range(1, max_rank + 1):
        if row["canonical_target"] == row["{}{}".format(base, i)]:
            return i
    return 0


def main(opt):
    with open(opt.targets, "r") as f:
        targets = ["".join(line.strip().split(" ")) for line in f.readlines()]

    predictions = [[] for i in range(opt.beam_size)]

    test_df = pd.DataFrame(targets)
    test_df.columns = ["target"]
    total = len(test_df)

    with open(opt.predictions, "r") as f:
        for i, line in enumerate(f.readlines()):
            predictions[i % opt.beam_size].append(
                "".join(line.strip().split(" ")),
            )

    for i, preds in enumerate(predictions):
        # Get canonical Target too:
        test_df["canonical_target"] = test_df["target"].apply(
            lambda x: canonicalize_smiles(x)
        )
        test_df["prediction_{}".format(i + 1)] = preds
        test_df["canonical_prediction_{}".format(i + 1)] = test_df[
            "prediction_{}".format(i + 1)
        ].apply(lambda x: canonicalize_smiles(x))

    test_df["rank"] = test_df.apply(
        lambda row: get_rank(row, "canonical_prediction_", opt.beam_size),
        axis=1,
    )

    print(test_df.head(10))

    correct = 0
    invalid_smiles_count = 0

    for i in range(1, opt.beam_size + 1):
        correct += (test_df["rank"] == i).sum()
        invalid_smiles = (
            test_df["canonical_prediction_{}".format(i)] == ""
        ).sum()  # noqa
        invalid_smiles_count += invalid_smiles

        if opt.invalid_smiles:
            print(
                "Top-{}: {:.1f}% || Invalid SMILES {:.2f}%".format(
                    i, correct / total * 100, invalid_smiles / total * 100
                )
            )
        else:
            print("Top-{}: {:.1f}%".format(i, correct / total * 100))

    return test_df


class Options:
    def __init__(self, targets, predictions, beam_size, invalid_smiles):
        self.targets = targets
        self.predictions = predictions
        self.beam_size = beam_size
        self.invalid_smiles = invalid_smiles


if __name__ == "__main__":
    print("Running Evaluation Script")
    target_path = "data/USPTO/MIT_mixed/tgt-test.txt"
    predictions_path = "exp_prefixLM/prefixLM_MIT_mixed_test_continued/predicted-tgt-epoch-epoch-testing123.txt"
    opt = Options(
        targets=target_path,
        predictions=predictions_path,
        beam_size=1,
        invalid_smiles=True,
    )
    test_df = main(opt=opt)

    # Check where the predictions are different:
    test_df["diff"] = test_df["canonical_prediction_1"] != test_df["target"]

    # Only save those rows:
    test_df = test_df[test_df["diff"] == True]

    # remove unnecessary columns:
    test_df = test_df.drop(columns=["diff", "rank", "target", "prediction_1"])

    # Save the dataframe to the same directory as the predictions:
    directory = os.path.dirname(predictions_path)
    save_path = os.path.join(directory, "diff_predictions.csv")

    # Seperate by . to easily read in CDK-depict :)
    # test_df.to_csv(save_path, index=True, sep=".")

    # Get the indices of the bad predictions:
    indices = test_df.index.tolist()
    # print(indices)
    # print(len(indices))

    # Save the indices to a file:
    with open(os.path.join(directory, "bad_indices.txt"), "w") as f:
        for idx in indices:
            f.write(str(idx) + "\n")
