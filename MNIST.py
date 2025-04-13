
import time
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
).to(device)


optimizer     = optim.Adam(model.parameters(), lr=0.01)
losses = []

start_time = time.time()

for epoch in range(10):
    for x, y in training_set:
        x, y  = x.view(-1, 784).to(device), y.to(device)
        y_pred = model(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

end_time = time.time()

plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

correct = 0
total = 0

with torch.no_grad():
    for x, y in validation_set:
        x, y = x.view(-1, 784).to(device), y.to(device)
        y_pred = model(x)
        predicted = torch.argmax(y_pred, dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

accuracy = correct / total * 100
print(f'Trained model in: {end_time - start_time} seconds\nWith an Validation Accuracy of: {accuracy:.2f}%')


