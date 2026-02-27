import os
import torch
import torchvision
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/HW2'
except ImportError:
    DRIVE_ROOT = '.'

from GAN_Networks import Generator


def run_generate(model='dcgan'):
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    num_images = 16

    if model == 'dcgan':
        checkpoint_path = os.path.join(DRIVE_ROOT, 'dcgan_checkpoints',
                                        'dcgan_latest.pth.tar')
        title = 'DCGAN Generated Faces (Inference)'
    else:
        checkpoint_path = os.path.join(DRIVE_ROOT, 'wgan_checkpoints',
                                        'wgan_latest.pth.tar')
        title = 'WGAN-GP Generated Faces (Inference)'

    gen = Generator(latent_dim).to(device)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                map_location=device, weights_only=False)
        gen.load_state_dict(checkpoint['netG'])
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Error: No checkpoint found at {checkpoint_path}! "
              f"Train the model first.")
        return

    gen.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        z = torch.randn(num_images, latent_dim, 1, 1).to(device)
        generated = gen(z).cpu()
        generated = (generated + 1) / 2

    grid = torchvision.utils.make_grid(generated, nrow=4).permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.clip(0, 1))
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_real_samples(num_samples=16):
    from LFW_DataLoader import LFWDataLoader
    lfw_loaders = LFWDataLoader(batch_size_train=num_samples,
                                 batch_size_test=num_samples)

    data_iter      = iter(lfw_loaders.test_loader)
    images, labels = next(data_iter)

    images = (images[:16] + 1) / 2
    grid   = torchvision.utils.make_grid(images, nrow=4)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy().clip(0, 1))
    plt.title("Real LFW Faces (Ground Truth)")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    run_generate(model='dcgan')
