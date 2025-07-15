import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
from augment import apply_auto_augmentations, apply_random_augmentations
from agent import AutoAugmentPolicy, collect_state_information, advantage_computation
from augment import AugmentationSpace

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

def train_loop(
    models, 
    dataloader_train, 
    dataloader_val, 
    agent, 
    epochs=10, 
    device='cuda', 
    early_stopping=5,
    lr=1e-3
    ):
    """
    models: dict with keys 'rl', 'random', 'none'
    agent: RL agent
    """
    optimizers = {k: torch.optim.Adam(v.parameters(), lr=lr, weight_decay=1e-3) for k, v in models.items()}
    histories = {k: defaultdict(list) for k in models.keys()}
    best_scores = {k: 0.0 for k in models.keys()}
    patience = {k: 0 for k in models.keys()}
    
    policy = AutoAugmentPolicy() # Policy decoder
    aug_space = AugmentationSpace() # Augmentation space
    
    # State info initialized /w random guesses
    val_acc = 0.1
    train_loss = 2
    val_loss = 2
    recent_train_loss = []
    recent_val_accs = []
    lr = lr
    
    # Agent update paramers
    state_trajectory = []           # State when policy was sampled
    actions_trajectory = []          # Actions chosen by policy  
    old_log_probs = []    # Log probability of chosen actions
    rewards = []          # Reward received after applying policy
    values_list = []
    
    # Augmentation policy
    aug_policy = None
    
    # Precompute policy on initial guesses
    state = collect_state_information(
        epoch=0,
        lr=lr,
        total_epochs=epochs,
        val_acc=val_acc,
        train_loss=train_loss,
        val_loss=val_loss,
        recent_train_loss=recent_train_loss,
        recent_val_accs=recent_val_accs,
        device=device
    )
    state_trajectory.append(state)
    action, log_prob, value = agent.actor_critic.get_action(state) # Get actions given current state
    actions_trajectory.append(action)
    old_log_probs.append(log_prob.detach())
    values_list.append(value)
    aug_policy = policy.decode_actions(action) # Convert raw action to applicable policy
    
    # Trajectory of policy
    policy_trajectory = []

    for epoch in range(epochs):
        
        for batch in dataloader_train:
            data, labels = batch

            for name, model in models.items():
                # Choose augmentation based on method
                if name == 'rl':
                    augmented_data = apply_auto_augmentations(data, aug_policy, aug_space)
                elif name == 'random':
                    augmented_data = apply_random_augmentations(data, aug_space)
                else:  # 'none'
                    augmented_data = data

                loss = train_step(model, optimizers[name], augmented_data, labels, device)
                if name == 'rl':
                    recent_train_loss.append(loss)
                histories[name]['loss'].append(loss)
            # break # for debugging

        # Validation
        for name, model in models.items():
            acc = evaluate(model, dataloader_val, device)
            if name == "rl": # Collect only rl agent's accuracy
                recent_val_accs.append(acc)
                if (epoch + 1) % 3 == 0: # Get FINAL reward
                    rewards.append(acc)
            histories[name]['val_acc'].append(acc)

            # Early stopping
            if acc > best_scores[name]:
                best_scores[name] = acc
                patience[name] = 0
            else:
                patience[name] += 1

        # Train agent every soa dn so epochs
        if agent and (epoch + 1) % 3 == 0:
            # Train for entire epoch with the same policy
            state = collect_state_information(
                epoch=epoch,
                lr=lr,
                total_epochs=epochs,
                val_acc=val_acc,
                train_loss=train_loss,
                val_loss=val_loss,
                recent_train_loss=recent_train_loss,
                recent_val_accs=recent_val_accs,
                device=device
            )
            state_trajectory.append(state)
            action, log_prob, value = agent.actor_critic.get_action(state) # Get actions given current state
            actions_trajectory.append(action)
            old_log_probs.append(log_prob.detach()) # Remove from CG or else silent bug
            values_list.append(value)
            aug_policy = policy.decode_actions(action) # Convert raw action to applicable policy
            advantages, returns, rewards_tensor = advantage_computation(rewards=rewards, values=values_list, device=device)
            
            states_tensor = torch.stack(state_trajectory)
            actions_tensor = torch.stack(actions_trajectory)
            old_log_probs_tensor = torch.stack(old_log_probs)
            
            agent.update(
                states=states_tensor,
                actions=actions_tensor,
                old_log_probs=old_log_probs_tensor,
                rewards=rewards_tensor,
                advantages=advantages
            )  # Buffered items
            
            # Clear buffer
            state_trajectory.clear()
            actions_trajectory.clear()
            old_log_probs.clear()
            rewards.clear()
            values_list.clear()
            
            # For inspection
            policy_trajectory.append(aug_policy)
            
        # Early stopping check
        if all(p >= early_stopping for p in patience.values()):
            print(f"Early stopping at epoch {epoch}")
            break

    return histories, best_scores, policy_trajectory

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
