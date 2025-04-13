

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


transform = transforms.ToTensor()

mnist_trainset = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_validate = dsets.MNIST(root='./data', train=False, download=True, transform=transform)

training_set   = DataLoader(mnist_trainset, batch_size=100)
validation_set = DataLoader(mnist_validate, batch_size=100)


model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)


optimizer = optim.SGD(model.parameters(), lr=0.01)
losses = []

for epoch in range(10):
    for x, y in training_set:
        x = x.view(-1, 784)
        y_pred = model(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


plt.plot(losses)
plt.show()


