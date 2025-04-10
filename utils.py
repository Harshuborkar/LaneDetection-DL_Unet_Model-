import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torchvision
import re

class LaneDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        label_name = img_name.rsplit(".",1)[0] + ".txt"  # Or your label extension [VAMSHEE] Added rsplit to get the name of the file without extension and then added  .txt     
        label_path = os.path.join(self.label_dir, label_name)

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = self.create_mask_from_text(label_path, image.shape[1], image.shape[0])
            
            ##[VAMSHEE] commented the below code as it is getting into infinite loop

            # if np.all(mask == 0):  # Check for empty masks
            #     return self.__getitem__(np.random.randint(0, self.__len__()))  # Return a different image

            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            return image, mask

        except FileNotFoundError:
            print(f"Warning: File not found for image: {img_name} or label: {label_name}. Skipping.")
            return self.__getitem__(np.random.randint(0, self.__len__()))  # Return a different image
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            return self.__getitem__(np.random.randint(0, self.__len__()))  # Return a different image

    def create_mask_from_text(self, label_path, width, height):
        mask = np.zeros((height, width), dtype=np.uint8)
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    # *** PARSE YOUR TEXT LABEL FILE FORMAT HERE ***
                    # Example (assuming x1, y1, x2, y2 coordinates):
                    match = re.findall(r'\d+', line)  # Find all numbers in the line
                    if len(match) == 4: # If we have 4 numbers
                        x1, y1, x2, y2 = map(int, match)
                        mask[y1:y2, x1:x2] = 1 # Set pixels within the rectangle to 1

                    # Example (assuming pixel coordinates and class label):
                    # parts = line.split()
                    # if len(parts) >= 3:
                    #     x = int(parts[0])
                    #     y = int(parts[1])
                    #     label = int(parts[2])
                    #     if label == 1:  # If it's a lane pixel (adjust as needed)
                    #         mask[y, x] = 1

        except FileNotFoundError:
            print(f"Warning: Label file not found: {label_path}")
            return mask
        except ValueError:
            print(f"Warning: Error parsing label file: {label_path}")
            return mask
        except Exception as e:
            print(f"Error processing label file {label_path}: {e}")
            return mask
        return mask


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

# Function had extra arguments which are label_dirs for val and train, removed them [VAMSHEE]
def get_loaders(
    train_dir,  val_dir,  batch_size, 
    train_transform, val_transform, num_workers=4, pin_memory=True
):
    train_ds = LaneDataset(
        image_dir=os.path.join(train_dir, "images"),  # Correct path[VAMSHEE] ADDED images instead of image
        label_dir=os.path.join(train_dir, "labels"),  # Corrected path[VAMSHEE] ADDED labels instead of label
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=16,  #hardcoded batch  size as 16, earlier it was batch_size[VAMSHEE]
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = LaneDataset(
        image_dir=os.path.join(val_dir, "images"),  # Correct path [VAMSHEE] ADDED images instead of image
        label_dir=os.path.join(val_dir, "labels"), # Correct path [VAMSHEE] ADDED labels instead of label
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1).float()  # Ensure y is float and has channel dim
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1).float(), f"{folder}/gt_{idx}.png")  # Save ground truth

    model.train()