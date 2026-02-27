import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
from torch.optim.lr_scheduler import StepLR

try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/HW2'
except ImportError:
    IN_COLAB  = False
    DRIVE_ROOT = '.'

from GAN_Networks import Generator, Discriminator, weights_init
from LFW_DataLoader import LFWDataLoader

NUM_EPOCHS       = 100
BATCH_SIZE       = 64
MILESTONE_EPOCHS = [5, 35, 65, 100]

LATENT_DIM = 100

LEARNING_RATE   = 0.0002
BETA1           = 0.5
BETA2           = 0.999
SCHEDULER_STEP  = 20
SCHEDULER_GAMMA = 0.5


class GAN_LFW:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.netG = Generator(LATENT_DIM).to(self.device)
        self.netD = Discriminator().to(self.device)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        self.data_loader = LFWDataLoader(batch_size_train=BATCH_SIZE,
                                         batch_size_test=BATCH_SIZE)

        self.optimizerG = optim.Adam(self.netG.parameters(),
                                     lr=LEARNING_RATE, betas=(BETA1, BETA2))
        self.optimizerD = optim.Adam(self.netD.parameters(),
                                     lr=LEARNING_RATE, betas=(BETA1, BETA2))
        self.schedulerG = StepLR(self.optimizerG,
                                  step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
        self.schedulerD = StepLR(self.optimizerD,
                                  step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
        self.criterion = nn.BCELoss()

        torch.manual_seed(42)
        self.fixed_noise = torch.randn(16, LATENT_DIM, 1, 1).to(self.device)

        self.G_losses = []
        self.D_losses = []

        self.checkpoint_path = os.path.join(DRIVE_ROOT, 'dcgan_checkpoints',
                                             'dcgan_latest.pth.tar')
        self.results_dir     = os.path.join(DRIVE_ROOT, 'dcgan_results')
        self.loss_dir        = os.path.join(DRIVE_ROOT, 'loss_data')
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.loss_dir,    exist_ok=True)

        self.curr_epoch = 0
        self.load_model()

    def save_model(self):
        checkpoint = {
            'epoch':      self.curr_epoch,
            'netG':       self.netG.state_dict(),
            'netD':       self.netD.state_dict(),
            'optimizerG': self.optimizerG.state_dict(),
            'optimizerD': self.optimizerD.state_dict(),
            'G_losses':   self.G_losses,
            'D_losses':   self.D_losses,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_model(self):
        if os.path.isfile(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path,
                                    map_location=self.device, weights_only=False)
            self.curr_epoch = checkpoint['epoch']
            self.netG.load_state_dict(checkpoint['netG'])
            self.netD.load_state_dict(checkpoint['netD'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG'])
            self.optimizerD.load_state_dict(checkpoint['optimizerD'])
            self.G_losses = checkpoint.get('G_losses', [])
            self.D_losses = checkpoint.get('D_losses', [])
            print(f"Resuming from epoch {self.curr_epoch}")

    def train(self):
        print(f"Training DCGAN on {self.device}. "
              f"Milestone saves at epochs {MILESTONE_EPOCHS}.")

        for epoch in range(self.curr_epoch, NUM_EPOCHS):
            self.curr_epoch = epoch + 1

            epoch_G_loss = 0.0
            epoch_D_loss = 0.0
            num_batches  = 0

            for i, (real_imgs, _) in enumerate(self.data_loader.train_loader):
                batch_size = real_imgs.size(0)
                real_imgs  = real_imgs.to(self.device)

                real_label = torch.ones(batch_size,  1).to(self.device)
                fake_label = torch.zeros(batch_size, 1).to(self.device)

                self.optimizerD.zero_grad()

                output_real  = self.netD(real_imgs)
                lossD_real   = self.criterion(output_real, real_label)

                z            = torch.randn(batch_size, LATENT_DIM, 1, 1).to(self.device)
                fake_imgs    = self.netG(z)
                output_fake  = self.netD(fake_imgs.detach())
                lossD_fake   = self.criterion(output_fake, fake_label)

                lossD = lossD_real + lossD_fake
                lossD.backward()
                self.optimizerD.step()

                self.optimizerG.zero_grad()

                output_g = self.netD(fake_imgs)
                lossG    = self.criterion(output_g, real_label)

                lossG.backward()
                self.optimizerG.step()

                epoch_G_loss += lossG.item()
                epoch_D_loss += lossD.item()
                num_batches  += 1

            self.schedulerG.step()
            self.schedulerD.step()

            avg_G = epoch_G_loss / num_batches
            avg_D = epoch_D_loss / num_batches
            self.G_losses.append(avg_G)
            self.D_losses.append(avg_D)

            print(f"Epoch [{self.curr_epoch}/{NUM_EPOCHS}] | "
                  f"G_loss: {avg_G:.4f} | "
                  f"D_loss: {avg_D:.4f}")

            self.save_model()

            np.save(os.path.join(self.loss_dir, 'dcgan_losses.npy'),
                    {'G_losses': self.G_losses, 'D_losses': self.D_losses})

            if self.curr_epoch in MILESTONE_EPOCHS:
                self.save_visual_results()

        print("Training complete. Losses saved to loss_data/dcgan_losses.npy")

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
    gan = GAN_LFW()
    gan.train()
