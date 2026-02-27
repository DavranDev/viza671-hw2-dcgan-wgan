import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/HW2'
    HF_CACHE   = os.path.join(DRIVE_ROOT, 'hf_cache')
except ImportError:
    IN_COLAB  = False
    DRIVE_ROOT = '.'
    HF_CACHE   = None

os.makedirs(DRIVE_ROOT, exist_ok=True)

IMAGE_SIZE = 64


class HFLFWDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform  = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item  = self.hf_dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label


class LFWDataLoader:
    def __init__(self, batch_size_train, batch_size_test):
        data_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        print("Loading LFW dataset from HuggingFace...")
        hf_data = load_dataset('logasja/lfw', split='train',
                               cache_dir=HF_CACHE)
        dataset = HFLFWDataset(hf_data, transform=data_transform)

        n_total = len(dataset)
        n_train = int(0.9 * n_total)
        n_test  = n_total - n_train
        train_data, test_data = random_split(
            dataset, [n_train, n_test],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(
            train_data, batch_size=batch_size_train, shuffle=True,
            num_workers=2, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_data, batch_size=batch_size_test, shuffle=False,
            num_workers=2, pin_memory=True
        )


def main():
    batch_size_train = 64
    batch_size_test  = 10

    torch.manual_seed(1)
    lfw_loaders = LFWDataLoader(batch_size_train, batch_size_test)

    data_iter  = iter(lfw_loaders.test_loader)
    test_batch = next(data_iter)

    test_batch_size = 5
    samples = test_batch[0][:test_batch_size]
    y_true  = test_batch[1][:test_batch_size]

    print(f'Batch shape: {samples.shape}')
    print(f'min={samples.min().item():.4f}')
    print(f'max={samples.max().item():.4f}')

    for i, sample in enumerate(samples):
        plt.subplot(1, test_batch_size, i + 1)
        plt.title(f'ID: {int(y_true[i])}')
        img = (sample.permute(1, 2, 0).numpy() + 1) / 2
        plt.imshow(img.clip(0, 1))
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
