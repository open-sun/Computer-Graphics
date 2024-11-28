
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
parser.add_argument('--n_epochs', type=int, default=2, help='number of epochs of training')
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
dataloader =MNIST(train=True, transform=transform,download=True).set_attrs(batch_size=opt.batch_size, shuffle=True)
# train_dataset = CIFAR10()
# dataloader = jt.dataset.DataLoader(train_dataset, batch_size=8)
# print(len(train_dataset))
# print(len(dataloader))




img_shape = (opt.channels, opt.img_size, opt.img_size)




# sheng cheng qi he jian bie qi
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embing = nn.Embedding(opt.n_classes,10)
        self.linear1 = nn.Linear((opt.latent_dim + opt.n_classes), 128)
        self.linear2 = nn.Linear(128, 256)
        self.batch_normal1 = nn.BatchNorm1d(256, 0.8)
        self.linear3 = nn.Linear(256, 512)
        self.batch_normal2 = nn.BatchNorm1d(512, 0.8)
        self.linear4 = nn.Linear(512, 1024)
        self.batch_normal3 = nn.BatchNorm1d(1024, 0.8)
        self.linear5 = nn.Linear(1024, int(np.prod(img_shape)))
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def execute(self, noise, labels):
        x= jt.contrib.concat((self.embing(labels), noise), dim=1)
        
        x = self.linear1(x)
        x=self.relu(x)

        x = self.linear2(x)
        x=self.batch_normal1(x)
        x=self.relu(x)

        x=self.linear3(x)
        x=self.batch_normal2(x)
        x=self.relu(x)

        x=self.linear4(x)
        x=self.batch_normal3(x)
        x=self.relu(x)

        x=self.linear5(x)

        x = self.tanh(x)
        x = x.view((x.shape[0], *img_shape))
        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embing = nn.Embedding(opt.n_classes,10)
        self.linear1 = nn.Linear((opt.n_classes + int(np.prod(img_shape))), 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 1)
        self.drop_out = nn.Dropout(0.4)
        self.relu = nn.LeakyReLU(0.2)


    def execute(self, img, labels):
        x = jt.contrib.concat((img.view((img.shape[0], (- 1))), self.label_embing(labels)), dim=1)

        x=self.linear1(x)
        x=self.relu(x)

        x=self.linear2(x)
        x=self.drop_out(x)
        x=self.relu(x)

        x=self.linear2(x)
        x=self.drop_out(x)
        x=self.relu(x)

        x=self.linear3(x)
        return x


def train():
    
    generator = Generator()
    discriminator = Discriminator()

    Gen_optimizer = nn.Adam(generator.parameters(), lr=.0001,betas=(0.5, 0.999))
    Dis_optimizer = nn.Adam(discriminator.parameters(), lr=.0001,betas=(0.5, 0.999))


    loos_func = nn.MSELoss() #loss

    g_losses = []
    d_losses = []
    with open("output.txt", "w") as output_file:  
        for epoch in range(opt.n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):

                batch_size = imgs.shape[0]

            
                valid = jt.ones([batch_size, 1]).float32().stop_grad() #compare with labels
                fake = jt.zeros([batch_size, 1]).float32().stop_grad()

            
                real_imgs = jt.array(imgs)
                labels = jt.array(labels)

            
                noise = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32()
                gen_labels = jt.array(np.random.randint(0, opt.n_classes, batch_size)).float32()
                gen_imgs = generator(noise, gen_labels)
            


                validity = discriminator(gen_imgs, gen_labels)#gen youhua 
                g_loss = loos_func(validity, valid)
                g_loss.sync()
                Gen_optimizer.step(g_loss)

                validity_real = discriminator(real_imgs, labels)
                d_real_loss = loos_func(validity_real, valid)

            
                validity_fake = discriminator(gen_imgs.stop_grad(), gen_labels)
                d_fake_loss = loos_func(validity_fake, fake)

            
                d_loss = (d_real_loss + d_fake_loss) / 2#pan bie qi you hua
                d_loss.sync()
                Dis_optimizer.step(d_loss)
                if i  % 50 == 0:
                    log_info = (
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data, g_loss.data)
                    )
                    print(log_info)
                    output_file.write(log_info + "\n")

                batches_done = epoch * len(dataloader) + i
                if batches_done % opt.sample_interval == 0:
                    sample_image(n_row=10, batches_done=batches_done)

            if epoch % 2 == 0:
                generator.save("saved_models/generator_last.pkl")
                discriminator.save("saved_models/discriminator_last.pkl")




# def save_image(img, path, nrow=10, padding=5):
#     N,C,W,H = img.shape
#     if (N%nrow!=0):
#         print("N%nrow!=0")
#         return
#     ncol=int(N/nrow)
#     img_all = []
#     for i in range(ncol):
#         img_ = []
#         for j in range(nrow):
#             img_.append(img[i*nrow+j])
#             img_.append(np.zeros((C,W,padding)))
#         img_all.append(np.concatenate(img_, 2))
#         img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
#     img = np.concatenate(img_all, 1)
#     img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
#     img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
#     min_=img.min()
#     max_=img.max()
#     img=(img-min_)/(max_-min_)*255
#     img=img.transpose((1,2,0))
#     if C==3:
#         img = img[:,:,::-1]
#     elif C==1:
#         img = img[:,:,0]
#     Image.fromarray(np.uint8(img)).save(path)

# def sample_image(n_row, batches_done):
#     """Saves a grid of generated digits"""
#     # Sample noise
#     z = jt.array(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))).float32().stop_grad()
#     labels = jt.array(np.array([num for _ in range(n_row) for num in range(n_row)])).float32().stop_grad()
#     gen_imgs = generator(z, labels)
#     save_image(gen_imgs.numpy(), "images/%d.png" % batches_done, nrow=n_row)




if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    adversarial_loss = nn.MSELoss()

    generator = Generator()
    discriminator = Discriminator()

    

    optimizer_G = nn.Adam(generator.parameters(), lr=.0001,betas=(0.5, 0.999))
    optimizer_D = nn.Adam(discriminator.parameters(), lr=.0001,betas=(0.5, 0.999))


    print(f"Using CUDA: {bool(jt.flags.use_cuda)}")
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

         
            valid = jt.ones([batch_size, 1]).float32().stop_grad()
            fake = jt.zeros([batch_size, 1]).float32().stop_grad()

           
            real_imgs = jt.array(imgs)
            labels = jt.array(labels)

          
            z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32()
            gen_labels = jt.array(np.random.randint(0, opt.n_classes, batch_size)).float32()

            
            gen_imgs = generator(z, gen_labels)
        
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            g_loss.sync()
            optimizer_G.step(g_loss)

            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

          
            validity_fake = discriminator(gen_imgs.stop_grad(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

         
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.sync()
            optimizer_D.step(d_loss)
            if i  % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data, g_loss.data)
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)

        if epoch % 2 == 0:
            generator.save("saved_models/generator_last.pkl")
            discriminator.save("saved_models/discriminator_last.pkl")