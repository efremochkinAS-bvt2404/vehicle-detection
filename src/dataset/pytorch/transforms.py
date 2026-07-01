from torchvision.transforms import functional as F


class DetectionToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_detection_transforms():
    return DetectionToTensor()