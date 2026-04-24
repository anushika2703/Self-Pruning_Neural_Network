# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Use 'Agg' backend so it saves the plot without needing a pop-up window
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# IMPORT FROM YOUR UTILS
from utils import get_sparsity_loss, evaluate_model

# 1. Custom Prunable Layer
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable Gate Scores
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, -2.0)
        # Starting at 0 means Sigmoid(0) = 0.5. It's easier to push to 0 from here.
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

# 2. Model Architecture
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

# 3. Plotting Helper
def save_gate_plot(model, lam_value):
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Final Gate Distribution (Lambda={lam_value})")
    plt.xlabel("Gate Value (Sigmoid Output)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    
    filename = f"sparsity_plot_lam_{lam_value}.png"
    plt.savefig(filename)
    plt.close()
    print(f"--- Graph saved as {filename} ---")

# 4. Training Engine
def train_and_evaluate(device, trainloader, testloader, lam_value):
    print(f"\n>>> Training with Lambda (lam) = {lam_value}")
    model = SelfPruningNet().to(device)
    # Using a slightly higher LR for gates helps them prune faster
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            # Sparsity Tax
            s_loss = get_sparsity_loss(model, PrunableLinear) 
            total_loss = criterion(outputs, labels) + (lam_value * s_loss)
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            if i % 400 == 399:
                print(f' Epoch {epoch + 1} | Batch {i + 1} | Loss: {running_loss / 400:.3f}')
                running_loss = 0.0

    # Save plot for the current lambda
    save_gate_plot(model, lam_value)
    
    # Evaluate with standard 0.01 threshold
    acc, sparsity = evaluate_model(model, testloader, device, PrunableLinear, threshold=0.1)
    return acc, sparsity

# 5. Main Loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # Comparing 3 Lambda values
    # Using 0.5 as the high value ensures we see pruning!
    lambda_list = [0.01, 0.1, 1.0]
    results_table = []

    for l_val in lambda_list:
        accuracy, sparsity = train_and_evaluate(device, trainloader, testloader, l_val)
        results_table.append({
            "lambda": l_val,
            "accuracy": accuracy,
            "sparsity": sparsity
        })

    # Output Table
    print("\n" + "="*50)
    print(f"{'Lambda':<10} | {'Test Accuracy':<15} | {'Sparsity Level':<15}")
    print("-" * 50)
    for row in results_table:
        print(f"{row['lambda']:<10} | {row['accuracy']:<14.2f}% | {row['sparsity']:<14.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()