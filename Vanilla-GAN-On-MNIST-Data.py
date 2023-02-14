# specifing batch size for the data
batch_size = 64
# for transforming data into tensors
transform = transforms.ToTensor()
# loading the mnist data
data = torchvision.datasets.MNIST('./data/', download=True, transform=transform, train=True)
# specifying data loader for data
dataLoader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
# iterable for dataloader
dataIter = iter(dataLoader)
# iterator for dataIter to call next item in list
imgs, labels = dataIter.next()
# input dimensions
in_dim = 100
# hidden dimensions
hid_dim = 128
# output dimensions
out_dim = 784


def img(imgs):
    # take imgs and turn it into grid of images
    img = torchvision.utils.make_grid(imgs)
    # converting imgs to numpy array
    npimgs = img.numpy()
    # ploting the img
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(npimgs, (1,2,0)), cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # layer
            nn.Linear(in_dim, hid_dim),
            # activation
            nn.ReLU(),
            # layer
            nn.Linear(hid_dim, out_dim),
            # activation
            nn.Sigmoid()
        )

    def forward(self, input):
        # forward pass
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # layer
            nn.Linear(out_dim, hid_dim),
            # activation
            nn.ReLU(),
            # layer
            nn.Linear(hid_dim, 1),
            # activation
            nn.Sigmoid()
        )

    def forward(self, input):
        # forward pass
        return self.model(input)

def loss(gen, dis):
    lr = 1e-3 # 0.001
    # loading optimizer for generative
    g_opt = opt.Adam(gen.parameters(), lr=lr)
    # loading optimizer for discrimnator
    d_opt = opt.Adam(dis.parameters(), lr=lr)

    return g_opt, d_opt

def training(gen, dis, g_opt, d_opt):
    for epoch in range(500):
      G_loss_run = 0.0
      D_loss_run = 0.0
      for i, data in enumerate(dataLoader):
        # get data
        X, _ = data
        # reshaping data
        X = X.view(X.size(0), -1).to(device)
        batch_size = X.size(0)

        # labels of 1
        one_labels = torch.ones(batch_size, 1).to(device)
        # labels of 0
        zero_labels = torch.zeros(batch_size, 1).to(device)
        # generating fake img
        z = torch.randn(batch_size, in_dim).to(device)
        # passing real img into discriminator model
        D_real = dis(X)
        # passing fake img into generator then discrimnator model
        D_fake = dis(gen(z))
        # calculating loss for discrimnator
        D_real_loss = F.binary_cross_entropy(D_real, one_labels) # 1 label for real imgae
        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels) # 0 label for fake image
        D_loss = D_real_loss + D_fake_loss
        # optimizing discrimnator based on loss
        d_opt.zero_grad()
        D_loss.backward()
        d_opt.step()
        # generating fake img
        z = torch.randn(batch_size, in_dim).to(device)
        # passing fake img into generator then discrimnator model
        D_fake = dis(gen(z))
        # calculating loss for generator
        G_loss = F.binary_cross_entropy(D_fake, one_labels) # 1 label cause have to prove real img and not fake
        # optimizing generator based on loss
        g_opt.zero_grad()
        G_loss.backward()
        g_opt.step()
        # calculating loss for discriminator and generator in each training epoch
        G_loss_run += G_loss.item()
        D_loss_run += D_loss.item()

      print('\nEpoch:{},   G_loss:{},    D_loss:{}'.format(epoch, G_loss_run/(i+1), D_loss_run/(i+1)))

    # saving generator model
    torch.save(gen.state_dict(), "gan_model.pth")

def main():
    gen = Generator().to(device)
    dis = Discriminator().to(device)
    g_opt, d_opt = loss(gen, dis)
    training(gen, dis, g_opt, d_opt)
    print("\nThe Generated Images: \n")

if __name__ == '__main__':
        main()
