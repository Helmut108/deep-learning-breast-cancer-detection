import torchvision.transforms.functional as TF

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