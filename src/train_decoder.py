"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""  # noqa

from tqdm import tqdm
import os
import time
import math
import wandb
import json

import torch

from src.decoder import GPTConfig, GPT
from src.dataset import ReactDataset, DataLoader
from src.utils import (
    collate_fn_right_padding,
)

from torch.optim.lr_scheduler import LambdaLR


def get_lr(iteration, warmup_iters, max_lr, lr_decay_iters, min_lr):
    if iteration < warmup_iters:
        return (max_lr / warmup_iters) * iteration
    else:
        return min_lr + (max_lr - min_lr) * math.exp(
            -1.0 * (iteration - warmup_iters) / lr_decay_iters
        )


def main():
    print("Running Training script")
    # Output params:
    # name = "large_model_2_no_src_tgt_mask"
    name = "decoder_testing_bidirectional_src_mask"

    out_dir = f"exp_MIT_decoder/{name}"
    trained_model_path = "exp_MIT_decoder/decoder_testing_bidirectional_src_mask/epoch_9_val_loss_0.0552_ckpt.pt"  # (
    # "exp_tree_decoder/large_model_1/epoch_39_val_loss_0.3318_ckpt.pt"
    # )

    # wandb params:
    wandb_log = True
    # wandb_project = "decoder_infinite_tree_prediction"
    wandb_project = "decoder_experiments_MIT_continued "
    wandb_run_name = f"decoder_{name}"  # 'run' + str(time.time())

    # Dataset params:
    block_size = 560  # 1024 + 512  # 560  # 911  # 560
    batch_size = 64  # 8  # 32  # 64 * 4  # 6
    epochs = 10

    # model params:
    init_from = "resume"
    if init_from == "resume":
        assert trained_model_path is not None

    # Model parameters:
    n_layer = 8
    n_head = 8
    n_embd = 128 * 4
    dropout = 0.1
    bias = True

    # Optimizer + lr scheduler paramters:
    learning_rate = 1e-4  # 1e-4  # max learning rate
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    warmup_iters = 2_000
    lr_decay_iters = 20_000
    min_lr = 5e-5

    # system stuff:
    device = "cuda"
    os.makedirs(out_dir, exist_ok=True)  # Create out directory
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # Miscelaneous:
    best_val_loss = 1e9

    use_mini = False
    use_src_tgt_mask = True
    use_bidirectional_src_mask = True
    use_seperate_pos_enc = True
    use_augm = False
    finetune = True
    use_scheduler = False

    accumulation_steps = 8  # Accumulate gradients over 4 batches

    if use_bidirectional_src_mask:
        assert finetune is True, "Cannot use bidirectional src mask without finetune."
    if use_seperate_pos_enc:
        assert (
            use_src_tgt_mask is True
        ), "Cannot use seperate pos enc without src tgt mask."

    data_path = "data/USPTO/MIT_mixed"
    data_path_train = "data/USPTO/MIT_mixed"
    data_path_test = "data/USPTO/MIT_mixed"

    if use_augm:
        data_path_train += "_augm"

    # data_path = "data/one_to_infinite_step_with_reaction_classes_fixed"
    # data_path_train = "data/one_to_infinite_step_with_reaction_classes_fixed"
    # data_path_test = "data/one_to_infinite_step_with_reaction_classes_fixed"

    # Load the datasets:
    train_dataset = ReactDataset(
        split="train",
        dataset_path=data_path_train,
        token2label_path=data_path,
        use_mini_data=use_mini,
    )
    val_dataset = ReactDataset(
        split="val",
        dataset_path=data_path_test,
        token2label_path=data_path,
        use_mini_data=use_mini,
    )

    # uses [src, tgt, PAD]
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_right_padding,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size * 4,
        shuffle=True,
        collate_fn=collate_fn_right_padding,
    )

    vocab_size = len(train_dataset.token2label)

    # model init
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,  # 2225,  # 256,
        dropout=dropout,
    )
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch...")
        # determine the vocab size we'll use for from-scratch training
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model = model.to(device)
        print("Model arguments used:", model_args)

    elif init_from == "resume":
        print("Loading model from trained path...")

        checkpoint = torch.load(trained_model_path, map_location=device)

        # exit()
        checkpoint_model_args = checkpoint["model_args"]
        # Refresh the model args with the ones used in previous training:
        for k in [
            "n_layer",
            "n_head",
            "n_embd",
            "block_size",
            "bias",
            "vocab_size",
        ]:
            model_args[k] = checkpoint_model_args[k]

        # Create model:
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)  # noqa
        model.load_state_dict(state_dict)
        model = model.to(device)
        print("Model arguments used:", model_args)

        # Refresh the best_val_loss:
        best_val_loss = checkpoint["best_val_loss"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        # weight_decay=weight_decay,
    )

    if use_scheduler:
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda iteration: get_lr(
                iteration,
                warmup_iters,
                learning_rate,
                lr_decay_iters,
                min_lr,
            ),
        )

    if init_from == "resume":
        # Load optimiser state dict:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Save hyperparameters to a config file
    config_dict = {
        "resume": init_from,
        "batch_size": batch_size,
        "layers": model_args["n_layer"],
        "embedding_size": model_args["n_embd"],
        "head_count": model_args["n_head"],
        "block_size": block_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
        "grad_clip": grad_clip,
        "warmup_iters": warmup_iters,
        "lr_decay_iters": lr_decay_iters,
        "min_lr": min_lr,
        "epochs": epochs,
        "use_seperate_pos_enc": use_seperate_pos_enc,
        "use_augm": use_augm,
        "use_scheduler": use_scheduler,
        "accumulation_steps": accumulation_steps,
    }

    config_filename = os.path.join(out_dir, "config.json")
    with open(config_filename, "w") as config_file:
        json.dump(config_dict, config_file)

    if wandb_log:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config_dict,
        )

    # Training Setup:
    losses_dict = {"train": [], "val": [], "train_per": [], "val_per": []}
    for e in range(epochs):
        train_loss = 0
        train_perplexity = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        model.train()
        for batch_idx, (
            x,
            y,
            react,
            prod,
            src_tgt_language_mask,
            src_mask,
            tgt_mask,
            full_att_mask,
            src_att_mask,
            tgt_att_mask,
        ) in progress_bar:
            # (src, PAD, tgt, PAD)

            if finetune:
                src_tgt_language_mask = torch.cat([src_mask, tgt_mask], dim=-1)
                src_tgt_language_mask = src_tgt_language_mask[:, :-1]

                # Create new att mask:
                full_att_mask = torch.cat([src_att_mask, tgt_att_mask], dim=-1)
                reaction = torch.cat([react, prod], dim=-1)
                x = reaction[:, :-1]
                y = reaction[:, 1:]

            # (src, tgt, PAD)
            else:
                prod = None

            # To match X_input
            full_att_mask = full_att_mask[:, :-1]

            x = x.to(device)
            y = y.to(device)
            src_tgt_language_mask = src_tgt_language_mask.to(device)
            full_att_mask = full_att_mask.to(device)

            if use_src_tgt_mask is False:
                src_tgt_language_mask = None

            _, loss = model(
                idx=x,
                targets=y,
                src_tgt_language_mask=src_tgt_language_mask,
                Products=prod,
                attention_mask=full_att_mask,
                use_seperate_pos_enc=use_seperate_pos_enc,
                use_bidirectional_src_mask=use_bidirectional_src_mask,
            )

            train_loss += loss.item()
            loss = loss / accumulation_steps  # Divide loss by accumulation steps

            loss.backward()

            if batch_idx % accumulation_steps == 0 or batch_idx == len(train_loader):
                # Update the model weights after accumulation_steps or at the end of epoch
                optimizer.step()
                optimizer.zero_grad()

                # MAY NEED TO NORMALISE THE GRADIENTS - as they might be big due to accum

            # Update progress message
            progress_bar.set_description(
                f"Epoch {e+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}"
            )

            # optimizer.step()
            if use_scheduler:
                scheduler.step()  # Step scheduler

        train_loss /= len(train_loader)  # * accumulation_steps)
        losses_dict["train"].append(train_loss)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for (
                x_val,
                y_val,
                react,
                prod,
                src_tgt_language_mask,
                src_mask,
                tgt_mask,
                full_att_mask,
                src_att_mask,
                tgt_att_mask,
            ) in tqdm(
                val_loader,
                desc=f"Epoch {e+1}/{epochs} - Validation",
            ):
                # (src, PAD, tgt, PAD)
                if finetune:
                    src_tgt_language_mask = torch.cat([src_mask, tgt_mask], dim=-1)
                    src_tgt_language_mask = src_tgt_language_mask[:, :-1]

                    # Create new att mask:
                    full_att_mask = torch.cat([src_att_mask, tgt_att_mask], dim=-1)

                    reaction = torch.cat([react, prod], dim=-1)
                    x_val = reaction[:, :-1]
                    y_val = reaction[:, 1:]

                # (src, tgt, PAD)
                else:
                    prod = None

                # To match X_input
                full_att_mask = full_att_mask[:, :-1]

                x_val = x_val.to(device)
                y_val = y_val.to(device)
                src_tgt_language_mask = src_tgt_language_mask.to(device)
                full_att_mask = full_att_mask.to(device)

                if use_src_tgt_mask is False:
                    src_tgt_language_mask = None

                _, loss_val = model(
                    idx=x_val,
                    targets=y_val,
                    src_tgt_language_mask=src_tgt_language_mask,
                    Products=prod,
                    attention_mask=full_att_mask,
                    use_seperate_pos_enc=use_seperate_pos_enc,
                    use_bidirectional_src_mask=use_bidirectional_src_mask,
                )
                val_loss += loss_val.item()

        val_loss /= len(val_loader)
        losses_dict["val"].append(val_loss)
        train_perplexity = 2**train_loss
        val_perplexity = 2**val_loss
        losses_dict["train_per"].append(train_perplexity)
        losses_dict["val_per"].append(val_perplexity)

        print(
            f"Epoch {e+1}/{epochs} Train: {train_loss:.4f} - Val: {val_loss:.4f} - Train Perplexity {train_perplexity:.4f} - Val Perplexity {val_perplexity:.4f}"  # noqa
        )

        # Log stuff:
        if wandb_log:
            wandb.log(
                {
                    "epoch": e,
                    "train_loss": losses_dict["train"][-1],
                    "train_perplex": losses_dict["train_per"][-1],
                    "val_loss": losses_dict["val"][-1],
                    "val_perplex": losses_dict["val_per"][-1],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        # Save the Model checkpoint:
        # if losses_dict["val"][-1] < best_val_loss:
        best_val_loss = losses_dict["val"][-1]
        # if e > 0:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": model_args,
            "epoch": e,
            "best_val_loss": best_val_loss,
        }
        path_to_save = os.path.join(
            out_dir, f"epoch_{e}_val_loss_{best_val_loss:.4f}_ckpt.pt"
        )
        print(f"saving checkpoint to {path_to_save}")
        torch.save(checkpoint, path_to_save)


if __name__ == "__main__":
    main()
