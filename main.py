# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

# IMPORT FROM YOUR UTILS
from utils import get_sparsity_loss, evaluate_model

# 1. Keep your class definitions here
class PrunableLinear(nn.Module):
    # ... (Keep your implementation here) ...
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNet(nn.Module):
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = PrunableLinear(3072, 512)
        self.layer2 = PrunableLinear(512, 256)
        self.layer3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 2. Put ALL execution logic inside main()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelfPruningNet().to(device)
    print(f"Model initialized on {device}")

    # Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # Hyperparameters
    lam = 0.1
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    print("Starting Training...")
    for epoch in range(5):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Use the imported utility function!
            s_loss = get_sparsity_loss(model, PrunableLinear) 
            total_loss = criterion(outputs, labels) + (lam * s_loss)
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            if i % 200 == 199:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    # Evaluation - Use the imported utility function!
    acc, sparsity = evaluate_model(model, testloader, device, PrunableLinear, threshold=0.1)
    print(f"\nResults for Lambda {lam}:")
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Sparsity Level: {sparsity:.2f}%")

if __name__ == "__main__":
    main()