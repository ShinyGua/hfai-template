from typing import Tuple, Any

from torchvision.datasets import ImageFolder


class ObjectImageWithIndex(ImageFolder):
    def __init__(self, root, transform, target_transform):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index
