import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch import nn

batch_size = 256
trans = transforms.Compose([transforms.ToTensor()])

train_set = dataset.MNIST(root="./data", train=True, transform=trans, download=True)
test_set = dataset.MNIST(root="./data", train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=batch_size,
                                          shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.lin1 = nn.Linear(784, 512)
        
        self.logvar = nn.Linear(512, 100)
        self.mu = nn.Linear(512, 100)
        
        self.lin3 = nn.Linear(100,512)
        self.lin4 = nn.Linear(512, 784)
        
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def encoder(self, X):
        x = self.lrelu(self.lin1(X))
        mean = self.mu(x)
        logstd = self.logvar(x)
        
        return mean, logstd
    
    def z(self, mean, logstd):
        std = torch.exp(0.5*logstd)
        noise = torch.randn_like(std)
        
        return mean + std*noise
    
    def decoder(self, X):
        x = self.lrelu(self.lin3(X))
        x = self.lin4(x)
        
        return x
    
    def forward(self, X):
        mean, logstd = self.encoder(X)
        x = self.z(mean, logstd)
        x = self.decoder(x)
        
        return x, mean, logstd
    
vae = VAE()
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

vae.apply(init_weights)


optim_VAE = torch.optim.Adam(vae.parameters(), lr=0.0002)
criterion = nn.BCELoss()


def loss_function(X_pred, X, mean, logstd):
    MSE = nn.functional.mse_loss(X_pred, X, reduction='sum')
    KLD = 0.5 * torch.sum(mean.pow(2) + torch.exp(logstd)-1-logstd)
    return KLD + MSE


## Train VAE
epochs = 50
imgs_VAE = []
fixed_noises = torch.randn(5,100)
for epoch in range(epochs):
    av_loss = 0
    for i, (X,_) in enumerate(train_loader):
        optim_VAE.zero_grad()
        y_hat, mean, logstd = vae(X.view(-1,784))
        with torch.enable_grad():
            loss = loss_function(y_hat, X.view(-1,784), mean, logstd)
        loss.backward()
        optim_VAE.step()
        av_loss += loss.data
    imgs_VAE.append(vae.decoder(fixed_noises))
    print("epoch : {}, loss : {}".format(epoch+1, av_loss/i))

for i in imgs_VAEs:
    plt.imshow(i[0].detach().numpy().reshape(28,28),cmap="gray")
    plt.show()
