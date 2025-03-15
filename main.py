import torch
import albumentations as A
import albumentations.pytorch as AP
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import torch.utils.data.dataloader as DataLoader
from src.utils import Dataset
from src.utils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs)

from src.model import My_Unet
from tqdm import tqdm


def main():

    # HYPERPARAMETERS
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    NUM_EPOCHS = 3
    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 160
    PIN_MEMORY = True
    LOAD_MODEL = True
    TRAIN_IMG_DIR = "data/images"
    TRAIN_MASK_DIR = "data/masks"
    VAL_IMG_DIR = "data/val_images"
    VAL_MASK_DIR = "data/val_masks"

    def train_fn(loader, model, optimizer, loss_fn):
        loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(DEVICE)
            targets = targets.float().unsqueeze(1).to(DEVICE)

            # forward
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

    def create_transforms():
        train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                AP.ToTensorV2(),
            ]
        )

        val_transforms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                AP.ToTensorV2(),
            ]
        )

        return train_transform, val_transforms
    
    # Load the model
    model = My_Unet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    train_transform, val_transforms = create_transforms()
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        PIN_MEMORY,
    )

    check_accuracy(val_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == '__main__':
    mp.freeze_support() # For Windows support
    main()