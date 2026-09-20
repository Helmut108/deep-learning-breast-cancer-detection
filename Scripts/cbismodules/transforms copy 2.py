from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import torch
from torchvision.transforms.v2 import InterpolationMode

class ResizeAndPad:
    #constructor
    def __init__(self, size):
        self.size = size

    # runs when caalled
    def __call__(self, image):
        # channel, height, width = image.shape
        _, height, width = image.shape

        scale = min(
            self.size / height,
            self.size / width
        )

        # calc the scale factor
        new_height = round(height * scale)
        new_width = round(width * scale)

        # resize
        image = TF.resize(
            image,
            size=[new_height, new_width],
            antialias=True
        )
        # print(f"Resized shape before padding: {image.shape}")
        # plt.imshow(image.squeeze(0), cmap="gray")
        # plt.title(f"Before padding: {tuple(image.shape)}")
        # plt.axis("off")
        # plt.show()

        # calculate necessary paddig
        pad_height = self.size - new_height
        pad_width = self.size - new_width

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # pad the image
        image = TF.pad(
            image,
            padding=[
                pad_left,
                pad_top,
                pad_right,
                pad_bottom
            ],
            fill=0
        )
        # print(f"Resized shape after padding: {image.shape}")
        return image


def create_transforms(use_augmentation=False):
    base_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeAndPad(512),
    ]

    train_transforms = base_transforms.copy()

    if use_augmentation:
        train_transforms.append(
            v2.RandomAffine(
                degrees=2,
                translate=(0.02, 0.02),
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            )
        )

    train_transform = v2.Compose(train_transforms)
    val_transform = v2.Compose(base_transforms)

    return train_transform, val_transform
