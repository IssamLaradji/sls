import torchvision
import torchvision.transforms as transforms


# ------------------------------------------
# Base class
class MainDataset:
    def __init__(self, split, dataset):
        if dataset == "CIFAR10":
            # dataset options
            data_dir = '/mnt/projects/counting/tmp/CIFAR10/'
            self.n_classes = 10

            if split == "train":
                transform_function = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])

                self.dataset = torchvision.datasets.CIFAR10(
                    root=data_dir,
                    train=True,
                    download=False,
                    transform=transform_function)

            if split == "val":
                transform_function = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ])

                self.dataset = torchvision.datasets.CIFAR10(
                    root=data_dir,
                    train=False,
                    download=False,
                    transform=transform_function)
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, y = self.dataset[index]

        return {
            "images": X,
            "targets": y,
            "meta": {
                "index": index,
                "image_id": index,
                "split": self.split
            }
        }


#