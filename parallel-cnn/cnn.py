
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*2*2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 128*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_train_objs():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root='/fs03/vf38/msyukron/data', train=True, download=True, transform=transform)
    model = SimpleCNN()  
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def train(model, train_loader, optimizer, epochs, save_every, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        print("Epoch {}, Loss: {:.4f}".format(epoch+1, running_loss / len(train_loader)))
        
        # Save the model every 'save_every' epochs
        if (epoch + 1) % save_every == 0:
            PATH = '/fs03/vf38/msyukron/cnn.pth'
            torch.save(model.state_dict(), PATH)
            print("Model saved at {}".format(PATH))

    print("Finished Training")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CNN on CIFAR10')
    parser.add_argument('epochs', type=int, help='Number of epochs to train')
    parser.add_argument('save_every', type=int, help='Save the model every X epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_set, model, optimizer = load_train_objs()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    train(model, train_loader, optimizer, args.epochs, args.save_every, device)
