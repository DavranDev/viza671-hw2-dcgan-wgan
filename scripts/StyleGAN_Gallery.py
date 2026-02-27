import os
import sys
import subprocess
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT   = '/content/drive/MyDrive/HW2'
    STYLEGAN_DIR = '/content/stylegan2-ada-pytorch'
except ImportError:
    DRIVE_ROOT   = '.'
    STYLEGAN_DIR = './stylegan2-ada-pytorch'

MODEL_PATH  = os.path.join(DRIVE_ROOT, 'models', 'ffhq.pkl')
GALLERY_DIR = os.path.join(DRIVE_ROOT, 'gallery')
PYTHON      = sys.executable

os.makedirs(os.path.join(DRIVE_ROOT, 'models'), exist_ok=True)
os.makedirs(GALLERY_DIR, exist_ok=True)


def setup_stylegan():
    if not os.path.isdir(STYLEGAN_DIR):
        print("Cloning StyleGAN2-ADA repository...")
        subprocess.run([
            'git', 'clone',
            'https://github.com/NVlabs/stylegan2-ada-pytorch.git',
            STYLEGAN_DIR
        ], check=True)
        print("Installing dependencies...")
        subprocess.run([
            PYTHON, '-m', 'pip', 'install', '-q',
            'click', 'requests', 'tqdm', 'pyspng', 'ninja', 'imageio-ffmpeg'
        ], check=True)
    else:
        print("StyleGAN2-ADA repo already present.")


def download_model():
    if os.path.isfile(MODEL_PATH):
        print(f"FFHQ model already in Drive: {MODEL_PATH}")
        return
    print("Downloading FFHQ pretrained model (~330 MB) to Drive...")
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
    subprocess.run(['wget', '-q', '-O', MODEL_PATH, url], check=True)
    print(f"Model saved to {MODEL_PATH}")


def generate_gallery(trunc=0.7):
    print(f"\nGenerating 50 faces → {GALLERY_DIR}")
    cmd = [
        PYTHON,
        os.path.join(STYLEGAN_DIR, 'generate.py'),
        '--outdir',  GALLERY_DIR,
        '--seeds',   '1-50',
        '--network', MODEL_PATH,
        '--trunc',   str(trunc),
    ]
    subprocess.run(cmd, check=True)
    print("Gallery generation complete.")


def show_gallery(cols=10):
    imgs = sorted([
        f for f in os.listdir(GALLERY_DIR)
        if f.endswith('.png') and 'seed' in f
    ])
    if not imgs:
        print("No images found. Run generate_gallery() first.")
        return

    rows  = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, fname in enumerate(imgs):
        img = Image.open(os.path.join(GALLERY_DIR, fname))
        axes[i].imshow(img)
        axes[i].set_title(fname.replace('seed', '').replace('.png', ''),
                          fontsize=7)
        axes[i].axis('off')

    for j in range(len(imgs), len(axes)):
        axes[j].axis('off')

    plt.suptitle('StyleGAN2-ADA — Gallery of 50 Generated Faces (FFHQ)',
                 fontsize=13)
    plt.tight_layout()

    out = os.path.join(DRIVE_ROOT, 'gallery_grid.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Gallery grid saved to {out}")
    plt.show()


if __name__ == '__main__':
    setup_stylegan()
    download_model()
    generate_gallery(trunc=0.7)
    show_gallery(cols=10)
    print("\nInspect the gallery above, then note which seed numbers")
    print("match each morph theme before running StyleGAN_Morph.py.")
