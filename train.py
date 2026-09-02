from vision_transformer import (
    TransformerEncoderBlock,
    VisionTransformer,
    get_config
)

from datasets import load_dataset
from tqdm import tqdm

import torch

def get_ds(config):
    ds = load_dataset("ethz/food101")




def train_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config = get_config()

    encoder  = TransformerEncoderBlock(
        config["embedding_dim"],
        config["num_of_heads"],
        config["mlp_dim"],
    )

    model = VisionTransformer(
        config["num_of_layers"],
        config["embedding_dim"],
        config["num_of_classes"],
        encoder,
        config["channels"],
        config["patch_size"],
    )
    model.train()


    train_dataloader, validation_dataloader = get_ds(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss(
        reduction="batchmean",
        label_smoothing=config["label_smoothing"],
    )

    for epoch in range(config["epochs"]):
        pbar = tqdm(train_dataloader, desc=f"Epochs {epoch}/{config["epochs"]}")

        for batch in pbar:
            inputs = batch["inputs"].to(device)
            targets = batch["labels"].to(device)
            logits = model(inputs)

            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss})


        torch.save(model.state_dict(), config["file_name"].format(epoch))


if __name__ == "__main__":
    train_model()