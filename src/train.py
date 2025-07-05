import torch
import torch.nn as nn
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
from augment import apply_augmentations, apply_random_augmentations

def train_step(model, optimizer, data, labels, device):
    model.train()
    data, labels = data.to(device), labels.to(device)
    
    optimizer.zero_grad()
    pred = model(data)
    loss = nn.functional.cross_entropy(pred, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train_loop(models, 
    dataloader_train, 
    dataloader_val, 
    agent, 
    epochs=10, 
    device='cuda', 
    early_stopping=5):
    """
    models: dict with keys 'rl', 'random', 'none'
    agent: RL agent
    """
    optimizers = {k: torch.optim.Adam(v.parameters(), lr=1e-3, weight_decay=1e-3) for k, v in models.items()}
    histories = {k: defaultdict(list) for k in models.keys()}
    best_scores = {k: 0.0 for k in models.keys()}
    patience = {k: 0 for k in models.keys()}

    for epoch in range(epochs):
        for batch in dataloader_train:
            data, labels = batch

            for name, model in models.items():
                # Choose augmentation based on method
                if name == 'rl':
                    aug_policy = agent.select_action(...)  # TODO: Define state & interface later
                    augmented_data = apply_augmentations(data, aug_policy)
                elif name == 'random':
                    augmented_data = apply_random_augmentations(data)
                else:  # 'none'
                    augmented_data = data

                loss = train_step(model, optimizers[name], augmented_data, labels, device)
                histories[name]['loss'].append(loss)

        # Validation
        for name, model in models.items():
            acc = evaluate(model, dataloader_val, device)
            histories[name]['val_acc'].append(acc)

            # Early stopping
            if acc > best_scores[name]:
                best_scores[name] = acc
                patience[name] = 0
            else:
                patience[name] += 1

        # Optionally: RL agent update every few epochs
        if agent and epoch % 5 == 0:
            agent.update(...)  # stored rewards/trajectories

        # Early stopping check
        if all(p >= early_stopping for p in patience.values()):
            print(f"Early stopping at epoch {epoch}")
            break

    return histories, best_scores

def final_test(models, testloader, device='cuda'):
    best_final_scores = {}

    for name, model in models.items():
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for data, labels in testloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        performance_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        best_final_scores[name] = performance_dict

    return best_final_scores
