import os
import torch
import torch.optim as optim
import torch.autograd as autograd
import torchvision
import numpy as np

try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/HW2'
except ImportError:
    IN_COLAB  = False
    DRIVE_ROOT = '.'

from GAN_Networks import Generator, Critic, weights_init
from LFW_DataLoader import LFWDataLoader

NUM_EPOCHS       = 100
BATCH_SIZE       = 64
MILESTONE_EPOCHS = [5, 35, 65, 100]
N_CRITIC         = 5

LATENT_DIM    = 100

LEARNING_RATE = 0.0001
BETA1         = 0.0
BETA2         = 0.9
LAMBDA_GP     = 10


def compute_gradient_penalty(critic, real_imgs, fake_imgs, device):
    B     = real_imgs.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real_imgs +
                    (1 - alpha) * fake_imgs.detach()).requires_grad_(True)
    d_interp = critic(interpolated)
    gradients = autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(B, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


class WGANGP_LFW:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.netG = Generator(LATENT_DIM).to(self.device)
        self.netC = Critic().to(self.device)
        self.netG.apply(weights_init)
        self.netC.apply(weights_init)

        self.data_loader = LFWDataLoader(batch_size_train=BATCH_SIZE,
                                         batch_size_test=BATCH_SIZE)

        self.optimizerG = optim.Adam(self.netG.parameters(),
                                     lr=LEARNING_RATE, betas=(BETA1, BETA2))
        self.optimizerC = optim.Adam(self.netC.parameters(),
                                     lr=LEARNING_RATE, betas=(BETA1, BETA2))

        torch.manual_seed(42)
        self.fixed_noise = torch.randn(16, LATENT_DIM, 1, 1).to(self.device)

        self.W_distances = []
        self.G_losses    = []

        self.checkpoint_path = os.path.join(DRIVE_ROOT, 'wgan_checkpoints',
                                             'wgan_latest.pth.tar')
        self.results_dir     = os.path.join(DRIVE_ROOT, 'wgan_results')
        self.loss_dir        = os.path.join(DRIVE_ROOT, 'loss_data')
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.loss_dir,    exist_ok=True)

        self.curr_epoch = 0
        self.load_model()

    def save_model(self):
        checkpoint = {
            'epoch':       self.curr_epoch,
            'netG':        self.netG.state_dict(),
            'netC':        self.netC.state_dict(),
            'optimizerG':  self.optimizerG.state_dict(),
            'optimizerC':  self.optimizerC.state_dict(),
            'W_distances': self.W_distances,
            'G_losses':    self.G_losses,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_model(self):
        if os.path.isfile(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path,
                                    map_location=self.device, weights_only=False)
            self.curr_epoch  = checkpoint['epoch']
            self.netG.load_state_dict(checkpoint['netG'])
            self.netC.load_state_dict(checkpoint['netC'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG'])
            self.optimizerC.load_state_dict(checkpoint['optimizerC'])
            self.W_distances = checkpoint.get('W_distances', [])
            self.G_losses    = checkpoint.get('G_losses',    [])
            print(f"Resuming from epoch {self.curr_epoch}")

    def train(self):
        print(f"Training WGAN-GP on {self.device}. "
              f"Milestone saves at epochs {MILESTONE_EPOCHS}.")

        for epoch in range(self.curr_epoch, NUM_EPOCHS):
            self.curr_epoch = epoch + 1

            epoch_W_dist = 0.0
            epoch_G_loss = 0.0
            num_batches  = 0

            for i, (real_imgs, _) in enumerate(self.data_loader.train_loader):
                batch_size = real_imgs.size(0)
                real_imgs  = real_imgs.to(self.device)

                for _ in range(N_CRITIC):
                    self.optimizerC.zero_grad()

                    z         = torch.randn(batch_size, LATENT_DIM, 1, 1).to(self.device)
                    fake_imgs = self.netG(z).detach()

                    loss_real = -self.netC(real_imgs).mean()
                    loss_fake =  self.netC(fake_imgs).mean()
                    gp        = compute_gradient_penalty(
                                    self.netC, real_imgs, fake_imgs, self.device)

                    lossC = loss_real + loss_fake + LAMBDA_GP * gp
                    lossC.backward()
                    self.optimizerC.step()

                wasserstein_d = (-loss_real - loss_fake).item()
                epoch_W_dist += wasserstein_d

                self.optimizerG.zero_grad()

                z     = torch.randn(batch_size, LATENT_DIM, 1, 1).to(self.device)
                lossG = -self.netC(self.netG(z)).mean()

                lossG.backward()
                self.optimizerG.step()

                epoch_G_loss += lossG.item()
                num_batches  += 1

            avg_W = epoch_W_dist / num_batches
            avg_G = epoch_G_loss / num_batches
            self.W_distances.append(avg_W)
            self.G_losses.append(avg_G)

            print(f"Epoch [{self.curr_epoch}/{NUM_EPOCHS}] | "
                  f"W_dist: {avg_W:.4f} | "
                  f"G_loss: {avg_G:.4f}")

            self.save_model()

            np.save(os.path.join(self.loss_dir, 'wgan_losses.npy'),
                    {'W_distances': self.W_distances, 'G_losses': self.G_losses})

            if self.curr_epoch in MILESTONE_EPOCHS:
                self.save_visual_results()

        print("Training complete. Losses saved to loss_data/wgan_losses.npy")

    def save_visual_results(self):
        self.netG.eval()
        with torch.no_grad():
            gen_imgs = self.netG(self.fixed_noise).cpu()
            gen_imgs = (gen_imgs + 1) / 2

            file_path = os.path.join(self.results_dir,
                                     f"epoch_{self.curr_epoch:03d}.png")
            torchvision.utils.save_image(gen_imgs, file_path, nrow=4)
            print(f"-> Sample images saved to {file_path}")
        self.netG.train()


if __name__ == '__main__':
    wgan = WGANGP_LFW()
    wgan.train()
