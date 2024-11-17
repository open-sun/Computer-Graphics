
from PIL import Image
import jittor as jt
from jittor import init
import argparse
import os
import numpy as np
import math
from jittor import nn
from jittor.dataset.cifar import CIFAR10
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
if jt.has_cuda:
    jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
opt = parser.parse_args()
print(opt)
# data
transform = transform.Compose([
    transform.Resize(opt.img_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
dataloader =MNIST(train=True, transform=transform,download=False).set_attrs(batch_size=opt.batch_size, shuffle=True)
# train_dataset = CIFAR10()
# dataloader = jt.dataset.DataLoader(train_dataset, batch_size=8)
# print(len(train_dataset))
# print(len(dataloader))




def trans(insize, outsize, normalize=True):
            layer = [nn.Linear(insize, outsize)]
            if normalize:
                layer.append(nn.BatchNorm1d(outsize, 0.8))
            layer.append(nn.LeakyReLU(0.2))
            return layer
def distrans(insize, outsize):
            layer = [nn.Linear(insize, outsize)]
            layer.append(nn.Dropout(0.4))
            layer.append(nn.LeakyReLU(0.2))
            return layer
img_shape = (opt.channels, opt.img_size, opt.img_size)




# sheng cheng qi he jian bie qi
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        self.model = nn.Sequential(*trans((opt.latent_dim + opt.n_classes), 128, normalize=False), *trans(128, 256), *trans(256, 512), *trans(512, 1024), nn.Linear(1024, int(np.prod(img_shape))), nn.Tanh())

    def execute(self, noise, labels):
        gen_input = jt.contrib.concat((self.label_emb(labels), noise), dim=1)
        img = self.model(gen_input)
        img = img.view((img.shape[0], *img_shape))
        return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.model = nn.Sequential(*trans((opt.n_classes + int(np.prod(img_shape))), 512,normalize=False),*distrans(512,512),*distrans(512,512), nn.Linear(512, 1))

    def execute(self, img, labels):
        d_in = jt.contrib.concat((img.view((img.shape[0], (- 1))), self.label_embedding(labels)), dim=1)
        validity = self.model(d_in)
        return validity


def save_image(img, path, nrow=10, padding=5):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("N%nrow!=0")
        return
    ncol=int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            img_.append(np.zeros((C,W,padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    elif C==1:
        img = img[:,:,0]
    Image.fromarray(np.uint8(img)).save(path)

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = jt.array(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))).float32().stop_grad()
    labels = jt.array(np.array([num for _ in range(n_row) for num in range(n_row)])).float32().stop_grad()
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.numpy(), "images/%d.png" % batches_done, nrow=n_row)

# ----------
#  Training
# ----------


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    adversarial_loss = nn.MSELoss()

    generator = Generator()
    discriminator = Discriminator()

    # Configure data loader

    optimizer_G = nn.Adam(generator.parameters(), lr=.0002,betas=(0.5, 0.999))
    optimizer_D = nn.Adam(discriminator.parameters(), lr=.0002,betas=(0.5, 0.999))

    print(f"Using CUDA: {bool(jt.flags.use_cuda)}")
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = jt.ones([batch_size, 1]).float32().stop_grad()
            fake = jt.zeros([batch_size, 1]).float32().stop_grad()

            # Configure input
            real_imgs = jt.array(imgs)
            labels = jt.array(labels)

            # -----------------
            #  Train Generator
            # -----------------

            # Sample noise and labels as generator input
            z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32()
            gen_labels = jt.array(np.random.randint(0, opt.n_classes, batch_size)).float32()

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            g_loss.sync()
            optimizer_G.step(g_loss)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.stop_grad(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.sync()
            optimizer_D.step(d_loss)
            if i  % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader)/64, d_loss.data, g_loss.data)
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)

        if epoch % 2 == 0:
            generator.save("saved_models/generator_last.pkl")
            discriminator.save("saved_models/discriminator_last.pkl")