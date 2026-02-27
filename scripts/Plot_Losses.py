import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/HW2'
except ImportError:
    DRIVE_ROOT = '.'

LOSS_DIR = os.path.join(DRIVE_ROOT, 'loss_data')


def plot_dcgan_losses():
    path = os.path.join(LOSS_DIR, 'dcgan_losses.npy')
    if not os.path.isfile(path):
        print(f"Error: {path} not found. Run DCGAN_Train.py first.")
        return

    data     = np.load(path, allow_pickle=True).item()
    G_losses = data['G_losses']
    D_losses = data['D_losses']
    epochs   = range(1, len(G_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, D_losses, color='red', linewidth=1.5)
    ax1.set_title('DCGAN: Discriminator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, G_losses, color='blue', linewidth=1.5)
    ax2.set_title('DCGAN: Generator Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('DCGAN Training Losses on LFW', fontsize=14)
    plt.tight_layout()

    out = os.path.join(DRIVE_ROOT, 'dcgan_loss_curve.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


def plot_wgan_losses():
    path = os.path.join(LOSS_DIR, 'wgan_losses.npy')
    if not os.path.isfile(path):
        print(f"Error: {path} not found. Run WGAN_GP_Train.py first.")
        return

    data        = np.load(path, allow_pickle=True).item()
    W_distances = data['W_distances']
    epochs      = range(1, len(W_distances) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, W_distances, color='green', linewidth=1.5)
    ax.set_title('WGAN-GP: Wasserstein Distance over Training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Wasserstein Distance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = os.path.join(DRIVE_ROOT, 'wgan_loss_curve.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


if __name__ == '__main__':
    plot_dcgan_losses()
    plot_wgan_losses()
