"""

Training Script.

"""

from tqdm import tqdm
import os
import wandb
import json

import torch

from src.decoder import GPTConfig, GPT
from src.dataset import ReactDataset, DataLoader
from src.utils import (
    collate_fn_right_padding, get_lr
)


class CFG:  
    """
    Class with all hyperparameters.
    """
    # Run params
    run_name = "decoder_testing_bidirectional_src_mask"
    output_dir = f"exp_MIT_decoder/{run_name}"
    wandb_log = False
    wandb_project = "decoder_experiments"
    wandb_run_name = run_name


    # Whether or not to load from trained model:
    trained_model_path = None
    init_from = "scratch"               # use "resume" to load from previous checkpoint

    # Dataset params: 
    data_path = "data/MIT_mixed"        # Path to src/tgt files.
    use_augm = False                    # Whether to use the augmented datasets.

    # Model Params: 
    block_size = 560                    # Max size of rxn sequences
    n_layer = 8
    n_head = 8
    n_embd = 128 * 4
    dropout = 0.1
    bias = False

    # training Params: 
    epochs = 10
    batch_size = 64
    accumulation_steps = 1             # Simulate larger batchsize
    learning_rate = 1e-4                # max learning rate
    use_scheduler = False
    weight_decay = 0.0
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0                     # clip gradients at this value, or disable if == 0.0
    warmup_iters = 2_000
    lr_decay_iters = 20_000
    min_lr = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging Params: 
    use_mini = True                    # Whether to use tiny part of dataset

    # Model Architecture changes: 
    use_src_tgt_language_mask = False            # Whether an extra mask telling the model which tokens are SRC/TGT/PAD should be included.
    use_bidirectional_src_mask = False  # Whether the attention on the SRC tokens should be bidirectional or not - PREFIX-LM
    use_seperate_pos_enc = False        # Whether there should be a seperate positional encoding for the src and tgt.
    finetune = True                    # Whether to train the model on TGT tokens only.





def main():
    print("Running Training script")
    if CFG.init_from == "resume":
        assert CFG.trained_model_path is not None

    # system stuff:
    print(f"USING {CFG.device} as device :)")
    os.makedirs(CFG.output_dir, exist_ok=True)  # Create out directory

    # Miscelaneous:
    best_val_loss = 1e9

    if CFG.use_bidirectional_src_mask:
        assert CFG.finetune is True, "Cannot use bidirectional src mask without finetune mode."
        # Model wont know how where to seperate the src and tgt.
    if CFG.use_seperate_pos_enc:
        assert (
            CFG.use_src_tgt_mask is True
        ), "Cannot use seperate pos enc without src tgt mask."
        # Need to use the src-tgt mask when using sepeate positional encoding - otherwise model would be invariant to positions.


    if CFG.use_augm:
        CFG.data_path += "_augm"

    # Load the datasets + Vocab size:
    train_dataset = ReactDataset(
        split="train",
        dataset_path=CFG.data_path,
        token2label_path=CFG.data_path,
        use_mini_data=CFG.use_mini,
    )
    val_dataset = ReactDataset(
        split="val",
        dataset_path=CFG.data_path,
        token2label_path=CFG.data_path,
        use_mini_data=CFG.use_mini,
    )
    vocab_size = len(train_dataset.token2label)


    # uses [src, tgt, PAD]
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_fn_right_padding,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=CFG.batch_size * 4,
        shuffle=True,
        collate_fn=collate_fn_right_padding,
    )

    # model init
    model_args = dict(
        n_layer=CFG.n_layer,
        n_head=CFG.n_head,
        n_embd=CFG.n_embd,
        block_size=CFG.block_size,
        bias=CFG.bias,
        vocab_size=vocab_size,  # 2225,  # 256,
        dropout=CFG.dropout,
    )
    if CFG.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch...")
        # determine the vocab size we'll use for from-scratch training
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model = model.to(CFG.device)
        print("Model arguments used:", model_args)

    elif CFG.init_from == "resume":
        print("Loading model from trained path...")

        checkpoint = torch.load(CFG.trained_model_path, map_location=CFG.device)
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
        model = model.to(CFG.device)
        print("Model arguments used:", model_args)

        # Refresh the best_val_loss:
        best_val_loss = checkpoint["best_val_loss"]

    # Setup optim:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        betas=(CFG.beta1, CFG.beta2),
        weight_decay=CFG.weight_decay,
    )

    if CFG.use_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda iteration: get_lr(
                iteration,
                CFG.warmup_iters,
                CFG.learning_rate,
                CFG.lr_decay_iters,
                CFG.min_lr,
            ),
        )

    if CFG.init_from == "resume":
        # Load optimiser state dict:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Save hyperparameters to a config file
    config_dict = {
        "resume": CFG.init_from,
        "batch_size": CFG.batch_size,
        "layers": model_args["n_layer"],
        "embedding_size": model_args["n_embd"],
        "head_count": model_args["n_head"],
        "block_size": CFG.block_size,
        "learning_rate": CFG.learning_rate,
        "weight_decay": CFG.weight_decay,
        "beta1": CFG.beta1,
        "beta2": CFG.beta2,
        "grad_clip": CFG.grad_clip,
        "warmup_iters": CFG.warmup_iters,
        "lr_decay_iters": CFG.lr_decay_iters,
        "min_lr": CFG.min_lr,
        "epochs": CFG.epochs,
        "use_seperate_pos_enc": CFG.use_seperate_pos_enc,
        "use_src_tgt_mask" : CFG.use_src_tgt_language_mask,
        "use_bidirectional_src_mask" : CFG.use_bidirectional_src_mask,
        "finetune": CFG.finetune,
        "use_augm": CFG.use_augm,
        "use_scheduler": CFG.use_scheduler,
        "accumulation_steps": CFG.accumulation_steps,
    }

    config_filename = os.path.join(CFG.output_dir, "config.json")
    with open(config_filename, "w") as config_file:
        json.dump(config_dict, config_file)

    if CFG.wandb_log:
        wandb.init(
            project=CFG.wandb_project,
            name=CFG.wandb_run_name,
            config=config_dict,
        )

    # Training Setup:
    losses_dict = {"train": [], "val": [], "train_per": [], "val_per": []}
    for e in range(CFG.epochs):
        train_loss = 0
        train_perplexity = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        model.train()
        for batch_idx, (
            x,          # Concatenation of SRC/TGT/PADDING without last token
            y,          # Without first token
            src,      
            tgt,
            src_tgt_language_mask,  # Which tokens are src, tgt or padding
            src_mask,               
            tgt_mask,
            full_att_mask,
            src_att_mask,
            tgt_att_mask,
        ) in progress_bar:
            # (src, PAD, tgt, PAD)
            if CFG.finetune:
                src_tgt_language_mask = torch.cat([src_mask, tgt_mask], dim=-1)
                src_tgt_language_mask = src_tgt_language_mask[:, :-1]

                # Create new att mask:
                full_att_mask = torch.cat([src_att_mask, tgt_att_mask], dim=-1)
                reaction = torch.cat([src, tgt], dim=-1)
                x = reaction[:, :-1]
                y = reaction[:, 1:]

            # (src, tgt, PAD)
            else:
                tgt = None

            # To match X_input
            full_att_mask = full_att_mask[:, :-1]

            x = x.to(CFG.device)
            y = y.to(CFG.device)
            src_tgt_language_mask = src_tgt_language_mask.to(CFG.device)
            full_att_mask = full_att_mask.to(CFG.device)

            if CFG.use_src_tgt_language_mask is False:
                src_tgt_language_mask = None


            _, loss = model(
                idx=x,
                targets=y,
                src_tgt_language_mask=src_tgt_language_mask,
                Products=tgt,
                attention_mask=full_att_mask,
                use_seperate_pos_enc=CFG.use_seperate_pos_enc,
                use_bidirectional_src_mask=CFG.use_bidirectional_src_mask,
            )

            train_loss += loss.item()
            loss = loss / CFG.accumulation_steps  # Divide loss by accumulation steps
            loss.backward()

            if batch_idx % CFG.accumulation_steps == 0 or batch_idx == len(train_loader):
                # Update the model weights after accumulation_steps or at the end of epoch
                optimizer.step()
                optimizer.zero_grad()

                # MAY NEED TO NORMALISE THE GRADIENTS - as they might be big due to accum

            # Update progress message
            progress_bar.set_description(
                f"Epoch {e+1}/{CFG.epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}"
            )

            # optimizer.step()
            if CFG.use_scheduler:
                scheduler.step()  # Step scheduler

        train_loss /= len(train_loader)  # * accumulation_steps)
        losses_dict["train"].append(train_loss)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for (
                x_val,
                y_val,
                src,
                tgt,
                src_tgt_language_mask,
                src_mask,
                tgt_mask,
                full_att_mask,
                src_att_mask,
                tgt_att_mask,
            ) in tqdm(
                val_loader,
                desc=f"Epoch {e+1}/{CFG.epochs} - Validation",
            ):
                # (src, PAD, tgt, PAD)
                if CFG.finetune:
                    src_tgt_language_mask = torch.cat([src_mask, tgt_mask], dim=-1)
                    src_tgt_language_mask = src_tgt_language_mask[:, :-1]

                    # Create new att mask:
                    full_att_mask = torch.cat([src_att_mask, tgt_att_mask], dim=-1)

                    reaction = torch.cat([src, tgt], dim=-1)
                    x_val = reaction[:, :-1]
                    y_val = reaction[:, 1:]

                # (src, tgt, PAD)
                else:
                    tgt = None

                # To match X_input
                full_att_mask = full_att_mask[:, :-1]

                x_val = x_val.to(CFG.device)
                y_val = y_val.to(CFG.device)
                src_tgt_language_mask = src_tgt_language_mask.to(CFG.device)
                full_att_mask = full_att_mask.to(CFG.device)

                if CFG.use_src_tgt_language_mask is False:
                    src_tgt_language_mask = None

                _, loss_val = model(
                    idx=x_val,
                    targets=y_val,
                    src_tgt_language_mask=src_tgt_language_mask,
                    Products=tgt,
                    attention_mask=full_att_mask,
                    use_seperate_pos_enc=CFG.use_seperate_pos_enc,
                    use_bidirectional_src_mask=CFG.use_bidirectional_src_mask,
                )
                val_loss += loss_val.item()

        val_loss /= len(val_loader)
        losses_dict["val"].append(val_loss)
        train_perplexity = 2**train_loss
        val_perplexity = 2**val_loss
        losses_dict["train_per"].append(train_perplexity)
        losses_dict["val_per"].append(val_perplexity)

        print(
            f"Epoch {e+1}/{CFG.epochs} Train: {train_loss:.4f} - Val: {val_loss:.4f} - Train Perplexity {train_perplexity:.4f} - Val Perplexity {val_perplexity:.4f}"  # noqa
        )

        # Log stuff:
        if CFG.wandb_log:
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
        if losses_dict["val"][-1] < best_val_loss:
            best_val_loss = losses_dict["val"][-1]
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "epoch": e,
                "best_val_loss": best_val_loss,
            }
            path_to_save = os.path.join(
                CFG.output_dir, f"epoch_{e}_val_loss_{best_val_loss:.4f}_ckpt.pt"
            )
            print(f"saving checkpoint to {path_to_save}")
            torch.save(checkpoint, path_to_save)


if __name__ == "__main__":
    main()
