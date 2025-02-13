import os
import json
import torch
import yaml
import argparse
from tqdm import tqdm
import wandb
from rxngpt.decoder import GPTConfig, GPT
from rxngpt.dataset import ReactDataset, DataLoader
from rxngpt.utils import collate_fn_right_padding, get_lr


def setup_datasets(config: dict) -> tuple[ReactDataset, ReactDataset, int]:
    """Initialize training and validation datasets."""
    data_path = config["data_path"] + ("_augm" if config["use_augm"] else "")

    train_data = ReactDataset(
        split="train",
        dataset_path=data_path,
        token2label_path=data_path,
        use_mini_data=config["use_mini"],
    )

    val_data = ReactDataset(
        split="val",
        dataset_path=data_path,
        token2label_path=data_path,
        use_mini_data=config["use_mini"],
    )

    return train_data, val_data, len(train_data.token2label)


def create_data_loaders(
    train_data: ReactDataset, val_data: ReactDataset, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_right_padding,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size * 4,
        shuffle=False,
        collate_fn=collate_fn_right_padding,
    )
    return train_loader, val_loader


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


def main():
    args = parse_args()
    config = load_yaml(args.config)

    if config["init_from"] == "resume":
        assert config["trained_model_path"] is not None

    # System setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING {device} as device :)")
    os.makedirs(config["output_dir"], exist_ok=True)  # Create out directory

    # Initialize Datasets and DataLoaders
    train_dataset, val_dataset, vocab_size = setup_datasets(config)
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, config["batch_size"]
    )

    # Model init
    model_args = {
        "n_layer": config["n_layer"],
        "n_head": config["n_head"],
        "n_embd": config["n_embd"],
        "block_size": config["block_size"],
        "bias": config["bias"],
        "vocab_size": vocab_size,
        "dropout": config["dropout"],
    }

    if config["init_from"] == "scratch":
        print("Initializing a new model from scratch...")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf).to(device)
    elif config["init_from"] == "resume":
        print("Loading model from trained path...")
        checkpoint = torch.load(config["trained_model_path"], map_location=device)
        checkpoint_model_args = checkpoint["model_args"]

        # Refresh model args from checkpoint
        model_args.update(checkpoint_model_args)
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf).to(device)
        model.load_state_dict(checkpoint["model"])
        best_val_loss = checkpoint["best_val_loss"]

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )

    if config["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda iteration: get_lr(
                iteration,
                config["warmup_iters"],
                config["learning_rate"],
                config["lr_decay_iters"],
                config["min_lr"],
            ),
        )

    config_filename = os.path.join(config["output_dir"], "config.json")
    with open(config_filename, "w") as config_file:
        json.dump(config, config_file)

    if config["wandb_log"]:
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            config=config,
        )
    # Training Setup:
    losses_dict = {"train": [], "val": [], "train_per": [], "val_per": []}
    best_val_loss = 1e9
    for e in range(config["epochs"]):
        train_loss = 0
        train_perplexity = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        model.train()
        for batch_idx, (
            x,  # Concatenation of SRC/TGT/PADDING without last token
            y,  # Without first token
            src,
            tgt,
            src_tgt_language_mask,  # Which tokens are src, tgt or padding
            src_mask,
            tgt_mask,
            full_att_mask,
            src_att_mask,
            tgt_att_mask,
        ) in progress_bar:
            if config["finetune"]:  # Train model on TGT tokens only
                src_tgt_language_mask = torch.cat([src_mask, tgt_mask], dim=-1)
                src_tgt_language_mask = src_tgt_language_mask[:, :-1]

                # Create new att mask:
                full_att_mask = torch.cat([src_att_mask, tgt_att_mask], dim=-1)
                reaction = torch.cat([src, tgt], dim=-1)
                x = reaction[:, :-1]
                y = reaction[:, 1:]

            else:  # Train on both SRC and TGT tokens
                tgt = None

            # To match X_input
            full_att_mask = full_att_mask[:, :-1]

            x = x.to(device)
            y = y.to(device)
            src_tgt_language_mask = src_tgt_language_mask.to(device)
            full_att_mask = full_att_mask.to(device)

            if config["use_src_tgt_language_mask"] is False:
                src_tgt_language_mask = None

            _, loss = model(
                idx=x,
                targets=y,
                src_tgt_language_mask=src_tgt_language_mask,
                Products=tgt,
                attention_mask=full_att_mask,
                use_seperate_pos_enc=config["use_seperate_pos_enc"],
                use_bidirectional_src_mask=["use_bidirectional_src_mask"],
            )

            train_loss += loss.item()
            loss = (
                loss / config["accumulation_steps"]
            )  # Divide loss by accumulation steps
            loss.backward()

            if batch_idx % config["accumulation_steps"] == 0 or batch_idx == len(
                train_loader
            ):
                # Update the model weights after accumulation_steps or at the end of epoch
                optimizer.step()
                optimizer.zero_grad()

                # MAY NEED TO NORMALISE THE GRADIENTS - as they might be big due to accum

            # Update progress message
            progress_bar.set_description(
                f"Epoch {e + 1}/{config['epochs']} - Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}"
            )

            # optimizer.step()
            if config["use_scheduler"]:
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
                desc=f"Epoch {e + 1}/{config['epochs']} - Validation",
            ):
                # (src, PAD, tgt, PAD)
                if config["finetune"]:
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

                x_val = x_val.to(device)
                y_val = y_val.to(device)
                src_tgt_language_mask = src_tgt_language_mask.to(device)
                full_att_mask = full_att_mask.to(device)

                if config["use_src_tgt_language_mask"] is False:
                    src_tgt_language_mask = None

                _, loss_val = model(
                    idx=x_val,
                    targets=y_val,
                    src_tgt_language_mask=src_tgt_language_mask,
                    Products=tgt,
                    attention_mask=full_att_mask,
                    use_seperate_pos_enc=config["use_seperate_pos_enc"],
                    use_bidirectional_src_mask=config["use_bidirectional_src_mask"],
                )
                val_loss += loss_val.item()

        val_loss /= len(val_loader)
        losses_dict["val"].append(val_loss)
        train_perplexity = 2**train_loss
        val_perplexity = 2**val_loss
        losses_dict["train_per"].append(train_perplexity)
        losses_dict["val_per"].append(val_perplexity)

        print(
            f"Epoch {e + 1}/{config['epochs']} Train: {train_loss:.4f} - Val: {val_loss:.4f} - Train Perplexity {train_perplexity:.4f} - Val Perplexity {val_perplexity:.4f}"  # noqa
        )

        # Log stuff:
        if config["wandb_log"]:
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
                config["output_dir"], f"epoch_{e}_val_loss_{best_val_loss:.4f}_ckpt.pt"
            )
            print(f"saving checkpoint to {path_to_save}")
            torch.save(checkpoint, path_to_save)


if __name__ == "__main__":
    main()
