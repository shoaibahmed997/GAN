import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


# hyper parameters 

BATCH_SIZE = 64 # this reffers to  "m" samples according to paper
LR = 0.01 
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
train_loader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
# utility functions 
def generate_noise():
    # this function will generate the z distribution
    z = torch.randn((BATCH_SIZE,100))
    return z

# create the generator 
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(100, 128),
            nn.Linear(128,256),
            nn.Linear(256,512),
            nn.Linear(512,1024),
            nn.Linear(1024,1*28*28),
            nn.Tanh()
        )

    def forward(self,z):
        z = self.layers(z)
        return z
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(784,512),
            nn.Linear(512,256),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.layers(x.view(BATCH_SIZE,-1))
        return x
    
# initilise the generator and discriminator
    
generator = Generator().to(device=DEVICE)
discriminator = Discriminator().to(device=DEVICE)


loss_fn = nn.BCELoss()

optimizer_g = torch.optim.SGD(generator.parameters(),lr=LR)
optimizer_d = torch.optim.SGD(discriminator.parameters(),lr=LR)

def training():
    for epoch in range(EPOCHS):
        z = generate_noise()
        z = z.to(device=DEVICE)
        for _, (img,label) in enumerate(train_loader):
            img = img.to(device=DEVICE)
            optimizer_d.zero_grad()
            # training disc for fake images
            fake_img = generator(z)
            fake_y = discriminator(fake_img)
            loss_fake = loss_fn(fake_y,torch.zeros(BATCH_SIZE,1).to(device=DEVICE))
            # training disc for real images
            real_y = discriminator(img)
            loss_real = loss_fn(real_y,torch.ones(BATCH_SIZE,1).to(device=DEVICE))

            total_disc_loss = loss_fake+loss_real

            total_disc_loss.backward()
            optimizer_d.step()

            # training the generator
            optimizer_g.zero_grad()
            z = generate_noise()
            z = z.to(device=DEVICE)
            fake_y = discriminator(generator(z))
            gen_loss = loss_fn(fake_y,torch.ones(BATCH_SIZE,1).to(device=DEVICE))
            gen_loss.backward()
            optimizer_g.step()

            print(f'EPOCH:{epoch} | discriminator loss:{total_disc_loss.item()} | generator loss:{gen_loss.item()}')

training()


            





