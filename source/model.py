import torch
import torch.nn as nn
import torch.nn.functional as F

# Note the model and functions here defined do not have any FL-specific components.


class Net(nn.Module):
    """A simple CNN suitable for simple vision tasks."""
    classes_channels = 0
    classes_class = 0
    def __init__(self, num_classes: int, classes_channels, classes_class) -> None:
        super(Net, self).__init__()
        self.classes_channels = classes_channels
        self.classes_class = classes_class
        if classes_channels == 1 and classes_class ==10:
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
        elif classes_channels == 3 and classes_class == 10:
            self.conv1 = nn.Conv2d(3, 16, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 5)
            self.fc1 = nn.Linear(32 * 5 * 5, 120)  # Adjust the input size for CIFAR-10
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
        elif classes_channels == 3 and classes_class == 100:
            self.conv1 = nn.Conv2d(3, 16, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 5)
            self.fc1 = nn.Linear(32 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.classes_channels == 1 and self.classes_class == 10:
            x = x.view(-1, 16 * 4 * 4)  # For MNIST
        elif self.classes_channels == 3 and self.classes_class == 10:
            x = x.view(-1, 32 * 5 * 5)  # For CIFAR-10
        elif self.classes_channels == 3 and self.classes_class == 100:
            x = x.view(-1, 32 * 5 * 5) # For CIFAR-100
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy