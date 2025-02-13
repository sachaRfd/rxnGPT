"""

File for utility functions such as collating functions and more

"""

import torch
import math
from torch.nn.utils.rnn import pad_sequence


def pad_sequences(sequences, batch_first=True, padding_value=0):
    """
    Pad a list of sequences to the same length.
    """
    return pad_sequence(
        [torch.tensor(seq) for seq in sequences],
        batch_first=batch_first,
        padding_value=padding_value,
    )


def reverse_sequences(sequences):
    """
    Reverse a list of sequences.
    """
    return [torch.tensor(seq[::-1]) for seq in sequences]


def collate_fn_pre_training(r_and_p_list):
    r, p, r_pos, p_pos, src_mask, tgt_mask, src_att_mask, tgt_att_mask = zip(
        *r_and_p_list
    )

    # Combine r, p and src_mask and tgt_mask:
    reaction = r + p
    src_tgt_mask = src_mask + tgt_mask
    attention_mask = src_att_mask + tgt_att_mask

    # Create positional embeddings:
    src_tgt_pos = r_pos + p_pos

    # Pad sequences:
    reaction = pad_sequences(reaction)
    src_tgt_mask = pad_sequences(src_tgt_mask)
    attention_mask = pad_sequences(attention_mask)
    src_tgt_pos = pad_sequences(src_tgt_pos, padding_value=-1)

    return reaction, src_tgt_mask, src_tgt_pos, attention_mask


def collate_fn_finetuning(r_and_p_list):
    r, p, src_mask, tgt_mask, src_att_mask, tgt_att_mask = zip(*r_and_p_list)

    # Pad sequences to the right:
    r = pad_sequences(r)
    p = pad_sequences(p)
    src_mask = pad_sequences(src_mask)
    tgt_mask = pad_sequences(tgt_mask)
    src_att_mask = pad_sequences(src_att_mask)
    tgt_att_mask = pad_sequences(tgt_att_mask)

    # Combine r, p and src_mask and tgt_mask:
    reaction = torch.cat((r, p), dim=-1)
    src_tgt_mask = torch.cat((src_mask, tgt_mask), dim=-1)
    attention_mask = torch.cat((src_att_mask, tgt_att_mask), dim=-1)

    # Create positional embeddings:
    src_pos = torch.arange(
        0,
    )
    tgt_pos = torch.arange(0, len(r))
    src_tgt_pos = torch.cat((src_pos, tgt_pos), dim=-1)

    return reaction, src_tgt_mask, src_tgt_pos, attention_mask


def collate_fn_right_padding(r_and_p_list):
    (
        X,
        Y,
        r,
        p,
        src_tgt_mask,
        src_mask,
        tgt_mask,
        full_att_mask,
        src_att_mask,
        tgt_att_mask,
    ) = zip(*r_and_p_list)

    # Pad sequences:
    X = pad_sequences(X, padding_value=0)
    Y = pad_sequences(Y, padding_value=0)
    r = pad_sequences(r, padding_value=0)
    p = pad_sequences(p, padding_value=0)
    src_tgt_mask = pad_sequences(
        src_tgt_mask, padding_value=2
    )  # Pad the rest of the src_tgt language mask as if its products (token 2)
    src_mask = pad_sequences(src_mask, padding_value=1)
    tgt_mask = pad_sequences(tgt_mask, padding_value=2)

    # Pad the attention masks:
    full_att_mask = pad_sequences(full_att_mask, padding_value=0)
    src_att_mask = pad_sequences(src_att_mask, padding_value=0)
    tgt_att_mask = pad_sequences(tgt_att_mask, padding_value=0)

    return (
        X,
        Y,
        r,
        p,
        src_tgt_mask,
        src_mask,
        tgt_mask,
        full_att_mask,
        src_att_mask,
        tgt_att_mask,
    )


def collate_fn_right_padding_with_srctgt_emb(r_and_p_list):
    X, Y, r, p, src_tgt_mask, src_mask, tgt_mask = zip(*r_and_p_list)

    # Pad sequences:
    X = pad_sequences(X)
    Y = pad_sequences(Y)
    r = pad_sequences(r)
    p = pad_sequences(p)
    src_tgt_mask = pad_sequences(src_tgt_mask)
    src_mask = pad_sequences(src_mask, padding_value=1)
    tgt_mask = pad_sequences(tgt_mask, padding_value=2)

    return X, Y, r, p, src_tgt_mask, src_mask, tgt_mask


def collate_fn_right_padding_with_src_mask_right_seperate_padding_emb(r_and_p_list):
    X, Y, r, p, src_tgt_mask, src_mask, tgt_mask = zip(*r_and_p_list)

    # Pad sequences:
    X = pad_sequence(
        [torch.tensor(seq) for seq in X],
        batch_first=True,
        padding_value=0,
    )
    Y = pad_sequence(
        [torch.tensor(seq) for seq in Y],
        batch_first=True,
        padding_value=0,
    )

    # pad mask:
    src_tgt_mask = pad_sequence(
        [torch.tensor(seq) for seq in src_tgt_mask],
        batch_first=True,
        padding_value=0,
    )

    tgt_mask = pad_sequence(
        [torch.tensor(seq) for seq in tgt_mask],
        batch_first=True,
        padding_value=0,
    )
    src_mask = pad_sequence(
        [torch.tensor(seq) for seq in src_mask],
        batch_first=True,
        padding_value=0,
    )

    # reverse r and p:
    r = [torch.tensor(seq[::-1]) for seq in r]
    # src_mask = [torch.tensor(seq[::-1]) for seq in src_mask]
    # r = [torch.tensor(seq) for seq in r]
    p = [torch.tensor(seq) for seq in p]  # Do not reverse the P

    padded_r = pad_sequence(
        [seq for seq in r],
        batch_first=True,
        padding_value=0,
    )
    padded_p = pad_sequence(
        [seq for seq in p],
        batch_first=True,
        padding_value=0,
    )

    # Pad the sequence with value 2
    # padded_src_mask = pad_sequence(
    #     [seq for seq in src_mask],
    #     batch_first=True,
    #     padding_value=0,
    # )

    # Re-reverse:
    padded_r = torch.flip(padded_r, [1])
    # src_mask = torch.flip(padded_src_mask, [1])
    # padded_p = torch.flip(padded_p, [1])  # Dont reverse the P

    return X, Y, padded_r, padded_p, src_tgt_mask, src_mask, tgt_mask


def collate_fn_right_padding_with_src_mask_left(r_and_p_list):
    """
    Collate function that pads the src mask to the left with the same value as the src mask.

    Pads the tgt mask to the right with the same value.


    """
    X, Y, r, p, mask, src_mask, tgt_mask = zip(*r_and_p_list)

    # Pad sequences:
    X = pad_sequence(
        [torch.tensor(seq) for seq in X],
        batch_first=True,
        padding_value=0,
    )
    Y = pad_sequence(
        [torch.tensor(seq) for seq in Y],
        batch_first=True,
        padding_value=0,
    )

    # pad mask:
    mask = pad_sequence(
        [torch.tensor(seq) for seq in mask],
        batch_first=True,
        padding_value=0,
    )

    tgt_mask = pad_sequence(
        [torch.tensor(seq) for seq in tgt_mask],
        batch_first=True,
        padding_value=2,
    )

    # reverse r and p:
    r = [torch.tensor(seq[::-1]) for seq in r]
    src_mask = [torch.tensor(seq[::-1]) for seq in src_mask]
    # r = [torch.tensor(seq) for seq in r]
    p = [torch.tensor(seq) for seq in p]  # Do not reverse the P

    padded_r = pad_sequence(
        [seq for seq in r],
        batch_first=True,
        padding_value=0,
    )
    padded_p = pad_sequence(
        [seq for seq in p],
        batch_first=True,
        padding_value=0,
    )

    # Pad the sequence with value 2
    padded_src_mask = pad_sequence(
        [seq for seq in src_mask],
        batch_first=True,
        padding_value=1,
    )

    # Re-reverse:
    padded_r = torch.flip(padded_r, [1])
    src_mask = torch.flip(padded_src_mask, [1])
    # padded_p = torch.flip(padded_p, [1])  # Dont reverse the P

    return X, Y, padded_r, padded_p, mask, src_mask, tgt_mask


def get_lr(iteration, warmup_iters, max_lr, lr_decay_iters, min_lr):
    if iteration < warmup_iters:
        return (max_lr / warmup_iters) * iteration
    else:
        return min_lr + (max_lr - min_lr) * math.exp(
            -1.0 * (iteration - warmup_iters) / lr_decay_iters
        )
