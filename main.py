import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

#Defining transforms and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.247, 0.243, 0.261])
])

#Loading Dataset CIFAR10
trainset = CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

#Defining Dataloaders
num_workers = os.cpu_count()
trainloader = DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=num_workers)
testloader = DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=num_workers)

#CIFAR10 Classes
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

#Assignation of device (cpu/gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device in use:")
print('CPU as device' if device.type == 'cpu' else 'GPU as device with CUDA')

##Definition of the model
class Convolutional_CIFAR10(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [3,32,32] -> [32,32,32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # [32,32,32] -> [32,16,16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [32,16,16] -> [64,16,16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # [64,16,16] -> [64,8,8]
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [64,8,8] -> [128,8,8]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # [128,8,8] -> [128,4,4]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 1024), #Based on last features output [128,4,4]
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#Calling the model, loss function and optimizer
model = Convolutional_CIFAR10(num_classes=10) #Classification of all the classes
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0005)
epochs = 30

#Optimization loop
loss_tr = []
loss_te = []
acc_tr = []
acc_te = []

print(f"\nStarting training with {epochs} epochs\n")

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct_train = 0
    total_train = 0

    #Training
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        _, predicted = torch.max(y_pred, 1)
        correct_train += (predicted == y).sum().item()
        total_train += y.size(0)

    avg_train_loss = running_loss / len(trainloader.dataset)
    train_accuracy = correct_train / total_train
    loss_tr.append(avg_train_loss)
    acc_tr.append(train_accuracy)

    #Evaluation
    model.eval()
    running_loss_test = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            running_loss_test += loss.item() * x.size(0)

            _, predicted = torch.max(y_pred, 1)
            correct_test += (predicted == y).sum().item()
            total_test += y.size(0)

    avg_test_loss = running_loss_test / len(testloader.dataset)
    test_accuracy = correct_test / total_test
    loss_te.append(avg_test_loss)
    acc_te.append(test_accuracy)
    #Printing results in each epoch
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}",
          f"| Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
    
print("\nTraning Finished")
input("Press a key to continue")