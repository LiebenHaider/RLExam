import torch.nn as nn

def train_step(model, optimizer, batch, device):
    # Unpack batch
    data, labels = data
    data, labels = data.to(device), labels.to(device)
    
    # Predict
    pred = model(data)
    
    # Compute loss
    loss = nn.functional.cross_entropy(pred, labels)
    
    # Backprop
    loss.backward()
    
    # Optimize
    optimizer.step()
    
    return loss.item()