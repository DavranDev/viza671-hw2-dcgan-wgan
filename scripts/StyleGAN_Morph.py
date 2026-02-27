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

MODEL_PATH = os.path.join(DRIVE_ROOT, 'models', 'ffhq.pkl')
MORPHS_DIR = os.path.join(DRIVE_ROOT, 'morphs')
PYTHON     = sys.executable

os.makedirs(MORPHS_DIR, exist_ok=True)

MORPH_PAIRS = [
    (3,  42, 'young_to_old'),
    (7,  28, 'left_to_right'),
    (15, 37, 'casual_to_formal'),
]

N_STEPS    = 10
FRAME_SIZE = 256
TRUNC_PSI  = 0.7


def ensure_stylegan():
    if not os.path.isdir(STYLEGAN_DIR):
        print("Cloning StyleGAN2-ADA repository...")
        subprocess.run([
            'git', 'clone',
            'https://github.com/NVlabs/stylegan2-ada-pytorch.git',
            STYLEGAN_DIR
        ], check=True)
        subprocess.run([
            PYTHON, '-m', 'pip', 'install', '-q',
            'click', 'requests', 'tqdm', 'pyspng', 'ninja', 'imageio-ffmpeg'
        ], check=True)


def load_generator(device):
    if STYLEGAN_DIR not in sys.path:
        sys.path.insert(0, STYLEGAN_DIR)
    import legacy
    print(f"Loading FFHQ model from {MODEL_PATH} ...")
    with open(MODEL_PATH, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()
    print("Model loaded.")
    return G


def seed_to_w(G, seed, device, trunc=TRUNC_PSI):
    z = torch.from_numpy(
        np.random.RandomState(seed).randn(1, G.z_dim)
    ).float().to(device)
    w = G.mapping(z, c=None, truncation_psi=trunc)
    return w


def w_to_image(G, w, device):
    with torch.no_grad():
        img = G.synthesis(w, noise_mode='const')
    img = (img.clamp(-1, 1) + 1) / 2
    img = img[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def lerp_w(w_start, w_end, n_steps=N_STEPS):
    ts = np.linspace(0.0, 1.0, n_steps)
    return [(1 - t) * w_start + t * w_end for t in ts]


def make_strip(G, seed_start, seed_end, label, device):
    print(f"\nMorphing: seed {seed_start} → seed {seed_end}  [{label}]")

    w_start  = seed_to_w(G, seed_start, device)
    w_end    = seed_to_w(G, seed_end,   device)
    w_frames = lerp_w(w_start, w_end, N_STEPS)

    frames = []
    for step, w in enumerate(w_frames):
        img = w_to_image(G, w, device).resize((FRAME_SIZE, FRAME_SIZE))
        frames.append(img)
        print(f"  step {step + 1}/{N_STEPS}", end='\r')
    print()

    strip = Image.new('RGB', (FRAME_SIZE * N_STEPS, FRAME_SIZE))
    for i, frame in enumerate(frames):
        strip.paste(frame, (i * FRAME_SIZE, 0))

    out_path = os.path.join(MORPHS_DIR, f'morph_{label}.png')
    strip.save(out_path)
    print(f"  Saved: {out_path}")
    return strip


def show_strips():
    strip_files = [
        os.path.join(MORPHS_DIR, f'morph_{label}.png')
        for _, _, label in MORPH_PAIRS
    ]
    labels = [label for _, _, label in MORPH_PAIRS]

    fig, axes = plt.subplots(len(strip_files), 1,
                             figsize=(N_STEPS * 2.5, len(strip_files) * 2.8))
    if len(strip_files) == 1:
        axes = [axes]

    for ax, path, lbl in zip(axes, strip_files, labels):
        if os.path.isfile(path):
            ax.imshow(Image.open(path))
            ax.set_title(lbl.replace('_', ' → ').title(), fontsize=11)
        else:
            ax.set_title(f'{lbl} — not found', fontsize=11)
        ax.axis('off')

    plt.suptitle('StyleGAN2-ADA — W-Space Morphing Strips', fontsize=13)
    plt.tight_layout()

    out = os.path.join(DRIVE_ROOT, 'morph_summary.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\nMorph summary saved to {out}")
    plt.show()


if __name__ == '__main__':
    ensure_stylegan()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    G = load_generator(device)

    for seed_start, seed_end, label in MORPH_PAIRS:
        make_strip(G, seed_start, seed_end, label, device)

    show_strips()

    print("\nDone! Check MORPH_PAIRS at the top of the file if the")
    print("transitions don't match the intended themes — update seed numbers")
    print("after inspecting the gallery from StyleGAN_Gallery.py.")
