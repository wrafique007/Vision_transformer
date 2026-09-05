from vision_transformer import (
    TransformerEncoderBlock,
    VisionTransformer,
    get_config
)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

import torch

class FoodDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset= dataset
        self.transforms = transforms.Compose([
                transforms.Resize(256), # Resize shorter edge to 256, keep aspect ratio
                transforms.Resize((224, 224)), # Then crop to 224*224
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image = self.dataset[index]["image"]
        label = self.dataset[index]["label"]

        transformed_image = self.transforms(image)

        return transformed_image, label


def get_ds(config):
    ds = load_dataset("ethz/food101")

    train_dataset = FoodDataset(ds["train"])
    validation_dataset = FoodDataset(ds["validation"])

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_dataloader, validation_dataloader






def train_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"device is {device}")# waleed

    config = get_config()

    # waleed
    # encoder  = TransformerEncoderBlock(
    #     config["embedding_dim"],
    #     config["num_of_heads"],
    #     config["mlp_dim"],
    # )

    model = VisionTransformer(
        config["num_of_layers"],
        config["embedding_dim"],
        config["num_of_heads"],
        config["mlp_dim"],
        config["num_of_classes"],
        config["channels"],
        config["patch_size"],
    )
    model.to(device)
    model.train()


    train_dataloader, validation_dataloader = get_ds(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss(
        reduction="mean",
        label_smoothing=config["label_smoothing"],
    )

    for epoch in range(config["epochs"]):
        pbar = tqdm(train_dataloader, desc=f"Epochs {epoch}/{config["epochs"]}")

        for inputs, targets in pbar:
            # waleed
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})


        torch.save(model.state_dict(), config["file_name"].format(epoch))


if __name__ == "__main__":
    config = get_config()
    train_model()