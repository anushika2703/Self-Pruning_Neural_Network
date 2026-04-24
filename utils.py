# utils.py
import torch
from torchvision import datasets, transforms

def get_sparsity_loss(model, PrunableLinear_Class):
    total_l1 = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear_Class):
            total_l1 += torch.sigmoid(module.gate_scores).sum()
    return total_l1

def evaluate_model(model, testloader, device, PrunableLinear_Class, threshold=1e-2):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    
    total_weights = 0
    pruned_weights = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear_Class):
            gates = torch.sigmoid(module.gate_scores)
            total_weights += gates.numel()
            pruned_weights += (gates < threshold).sum().item()
    
    sparsity_pct = 100 * pruned_weights / total_weights
    return accuracy, sparsity_pct