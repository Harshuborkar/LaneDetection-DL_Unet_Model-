import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import re  # For regular expressions (if needed for parsing)

class LaneDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None): # Changed mask_dir to label_dir
        self.image_dir = image_dir
        self.label_dir = label_dir # Changed mask_dir to label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        label_name = img_name.split(".")[0] + ".txt" # Or whatever your label file extension is
        label_path = os.path.join(self.label_dir, label_name)

        try:
            image = np.array(Image.open(img_path).convert("RGB"))

            # ***LABEL PROCESSING (Crucial Change)***
            mask = self.create_mask_from_text(label_path, image.shape[1], image.shape[0]) # Create mask from text

            if np.all(mask == 0): # Check if the mask is all zeros.
              return self.__getitem__(np.random.randint(0, self.__len__())) # Return a different image-mask pair.

            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            return image, mask

        except FileNotFoundError:
            print(f"Warning: File not found for image: {img_name} or label: {label_name}. Skipping.")
            return self.__getitem__(np.random.randint(0, self.__len__())) # Return a different image-mask pair.
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            return self.__getitem__(np.random.randint(0, self.__len__())) # Return a different image-mask pair.


    def create_mask_from_text(self, label_path, width, height):
        """Creates a binary mask from the text label file."""
        mask = np.zeros((height, width), dtype=np.uint8)  # Initialize an empty mask

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    # ***PARSE EACH LINE OF YOUR TEXT FILE***
                    # This is the most dataset-specific part.  You need to
                    # extract the segmentation information (e.g., coordinates,
                    # pixel values) from each line of the text file.

                    # Example (assuming each line has x1, y1, x2, y2 coordinates):
                    match = re.findall(r'\d+', line) # Find all numbers in the line
                    if len(match) == 4: # If we have 4 numbers
                        x1, y1, x2, y2 = map(int, match)
                        # Draw a rectangle in the mask
                        mask[y1:y2, x1:x2] = 1 # Set pixels within the rectangle to 1
                    # Example (assuming each line has pixel coordinates and a class label):
                    # parts = line.split()
                    # if len(parts) >= 3:
                    #     x = int(parts[0])
                    #     y = int(parts[1])
                    #     label = int(parts[2])
                    #     if label == 1:  # If it's a lane pixel (adjust as needed)
                    #         mask[y, x] = 1


        except FileNotFoundError:
            print(f"Warning: Label file not found: {label_path}")
            return mask # Return an empty mask if the file is not found
        except ValueError:
            print(f"Warning: Error parsing label file: {label_path}")
            return mask # Return an empty mask if there's a parsing error
        except Exception as e:
            print(f"Error processing label file {label_path}: {e}")
            return mask

        return mask