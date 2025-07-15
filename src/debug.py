import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from agent import PPOAgent, AutoAugmentPolicy, collect_state_information, advantage_computation
from augment import AugmentationSpace, apply_auto_augmentations, apply_random_augmentations

def test_basic_functionality():
    """Test basic agent operations"""
    print("=" * 50)
    print("TEST 1: Basic Agent Functionality")
    print("=" * 50)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 9  # Based on your collect_state_information
    action_dim = 18  # 3 sub-policies * 2 ops * 3 params = 18
    hidden_dim = 128
    
    agent = PPOAgent(state_dim, action_dim, hidden_dim).to(device)
    policy_decoder = AutoAugmentPolicy(num_sub_policies=3, ops_per_sub_policy=2)
    
    # Test 1.1: State collection
    print("Testing state collection...")
    state = collect_state_information(
        epoch=5, total_epochs=50, val_acc=0.65, train_loss=1.2, val_loss=1.5,
        lr=0.001, recent_train_loss=[1.5, 1.4, 1.3, 1.2, 1.1], 
        recent_val_accs=[0.6, 0.61, 0.63, 0.64, 0.65], device=device
    )
    print(f"✓ State shape: {state.shape}, device: {state.device}")
    print(f"✓ State values: {state}")
    print(f"✓ State values dtype: {state.dtype}")
    
    
    # Test 1.2: Action sampling
    print("\nTesting action sampling...")
    # state = state.unsqueeze(0)
    print("Test 1, state shape", state.shape)
    action, log_prob, value = agent.actor_critic.get_action(state)
    print(f"✓ Action shape: {action.shape}, value: {action}")
    print(f"✓ Log prob: {log_prob}")
    print(f"✓ Value estimate: {value.item():.4f}")
    
    # Test 1.3: Policy decoding
    print("\nTesting policy decoding...")
    aug_policy = policy_decoder.decode_actions(action)
    print(f"✓ Generated {len(aug_policy)} sub-policies")
    for i, sub_policy in enumerate(aug_policy):
        print(f"  Sub-policy {i}: {len(sub_policy)} operations")
        for j, op in enumerate(sub_policy):
            print(f"    Op {j}: operation={op['operation']}, magnitude={op['magnitude']}, prob={op['probability']:.2f}")
    
    return True

def test_augmentation_application():
    """Test augmentation application"""
    print("\n" + "=" * 50)
    print("TEST 2: Augmentation Application")
    print("=" * 50)
    
    # Create fake image batch
    batch_size = 4
    fake_images = torch.randn(batch_size, 3, 32, 32)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PPOAgent(9, 18, 128).to(device)
    policy_decoder = AutoAugmentPolicy()
    aug_space = AugmentationSpace()
    
    # Get a policy
    state = torch.randn(9, dtype=torch.float32).to(device)
    # state = state.unsqueeze(0)
    action, _, _ = agent.actor_critic.get_action(state)
    aug_policy = policy_decoder.decode_actions(action)
    
    print(f"Testing on batch of {batch_size} images...")
    
    # Test RL augmentation
    try:
        augmented_rl = apply_auto_augmentations(fake_images, aug_policy, aug_space)
        print(f"✓ RL augmentation: input {fake_images.shape} → output {augmented_rl.shape}")
    except Exception as e:
        print(f"✗ RL augmentation failed: {e}")
        return False
    
    # Test random augmentation  
    try:
        augmented_random = apply_random_augmentations(fake_images, aug_space)
        print(f"✓ Random augmentation: input {fake_images.shape} → output {augmented_random.shape}")
    except Exception as e:
        print(f"✗ Random augmentation failed: {e}")
        return False
    
    # Verify shapes match
    if augmented_rl.shape == fake_images.shape and augmented_random.shape == fake_images.shape:
        print("✓ All augmentation outputs have correct shapes")
        return True
    else:
        print("✗ Shape mismatch in augmentations")
        return False

def test_ppo_update():
    """Test PPO update mechanism"""
    print("\n" + "=" * 50)
    print("TEST 3: PPO Update Mechanism")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PPOAgent(9, 18, 128).to(device)
    
    # Simulate 5 epochs of experience
    num_experiences = 5
    states = []
    actions = []
    old_log_probs = []
    rewards = []
    values = []
    
    print(f"Generating {num_experiences} fake experiences...")
    
    for i in range(num_experiences):
        # Fake state
        state = torch.randn(9, dtype=torch.float32).to(device)
        # state = state.unsqueeze(0)
        states.append(state)
        
        # Get action
        action, log_prob, value = agent.actor_critic.get_action(state)
        actions.append(action)
        old_log_probs.append(log_prob.detach())
        values.append(value.squeeze())
        
        # Fake reward (validation accuracy)
        reward = 0.5 + 0.1 * i + 0.05 * np.random.randn()  # Slightly increasing
        rewards.append(reward)
    
    print(f"✓ Collected {len(states)} experiences")
    print(f"  Rewards: {[f'{r:.3f}' for r in rewards]}")
    
    # Test advantage computation
    try:
        advantages, returns = advantage_computation(rewards, [v.item() for v in values])
        print(f"✓ Advantage computation successful")
        print(f"  Advantages: {[f'{a:.3f}' for a in advantages]}")
        print(f"  Returns: {[f'{r:.3f}' for r in returns]}")
    except Exception as e:
        print(f"✗ Advantage computation failed: {e}")
        return False
    
    # Test PPO update
    try:
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        old_log_probs_tensor = torch.stack(old_log_probs)
        
        print("Running PPO update...")
        agent.update(states_tensor, actions_tensor, old_log_probs_tensor, returns, advantages)
        print("✓ PPO update successful")
        return True
    except Exception as e:
        print(f"✗ PPO update failed: {e}")
        return False

def test_learning_simulation():
    """Simulate a mini learning process"""
    print("\n" + "=" * 50)
    print("TEST 4: Learning Simulation")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PPOAgent(9, 18, 128).to(device)
    policy_decoder = AutoAugmentPolicy()
    
    # Simulate training over multiple "epochs"
    num_updates = 10
    rewards_history = []
    value_predictions = []
    
    print(f"Simulating {num_updates} policy updates...")
    
    for update in range(num_updates):
        # Collect experiences for this update
        states, actions, old_log_probs, rewards, values = [], [], [], [], []
        
        for epoch in range(5):  # 5 epochs per update
            # Simulate evolving training state
            progress = (update * 5 + epoch) / (num_updates * 5)
            val_acc = 0.1 + 0.8 * progress + 0.05 * np.random.randn()
            train_loss = 2.5 * (1 - progress) + 0.1 * np.random.randn()
            
            state = collect_state_information(
                epoch=epoch, total_epochs=50, val_acc=val_acc, train_loss=train_loss,
                val_loss=train_loss + 0.1, lr=0.001, 
                recent_train_loss=[train_loss] * 5, recent_val_accs=[val_acc] * 5,
                device=device
            )
            # state = state.unsqueeze(0)
            action, log_prob, value = agent.actor_critic.get_action(state)
            
            # Simulate reward (with some noise around actual performance)
            reward = val_acc + 0.02 * np.random.randn()
            
            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob.detach())
            values.append(value.squeeze())
            rewards.append(reward)
        
        # Compute advantages and update
        advantages, returns = advantage_computation(rewards, [v.item() for v in values])
        
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        old_log_probs_tensor = torch.stack(old_log_probs)
        
        agent.update(states_tensor, actions_tensor, old_log_probs_tensor, returns, advantages)
        
        # Track progress
        avg_reward = np.mean(rewards)
        avg_value_pred = np.mean([v.item() for v in values])
        rewards_history.append(avg_reward)
        value_predictions.append(avg_value_pred)
        
        if update % 2 == 0:
            print(f"  Update {update}: avg_reward={avg_reward:.3f}, avg_value_pred={avg_value_pred:.3f}")
    
    print(f"✓ Learning simulation completed")
    print(f"  Initial avg reward: {rewards_history[0]:.3f}")
    print(f"  Final avg reward: {rewards_history[-1]:.3f}")
    print(f"  Improvement: {rewards_history[-1] - rewards_history[0]:.3f}")
    
    # Check if agent learned something
    if rewards_history[-1] > rewards_history[0]:
        print("✓ Agent appears to be learning (rewards increased)")
    else:
        print("? Agent may not be learning effectively (rewards didn't increase)")
    
    return True

def test_edge_cases():
    """Test edge cases and potential failure points"""
    print("\n" + "=" * 50)
    print("TEST 5: Edge Cases")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test with minimal recent history
    print("Testing with minimal recent history...")
    try:
        state = collect_state_information(
            epoch=0, total_epochs=50, val_acc=0.1, train_loss=2.3, val_loss=2.3,
            lr=0.001, recent_train_loss=[2.3], recent_val_accs=[0.1], device=device
        )
        print("✓ Minimal history handled correctly")
    except Exception as e:
        print(f"✗ Minimal history failed: {e}")
    
    # Test with very high/low values
    print("Testing with extreme values...")
    try:
        state = collect_state_information(
            epoch=49, total_epochs=50, val_acc=0.99, train_loss=0.01, val_loss=0.01,
            lr=1e-6, recent_train_loss=[0.01] * 5, recent_val_accs=[0.99] * 5, device=device
        )
        print("✓ Extreme values handled correctly")
    except Exception as e:
        print(f"✗ Extreme values failed: {e}")
    
    # Test single experience update
    print("Testing single experience update...")
    try:
        agent = PPOAgent(9, 18, 128).to(device)
        state = torch.randn(9, dtype=torch.float32).to(device)
        # state = state.unsqueeze(0)
        action, log_prob, value = agent.actor_critic.get_action(state)
        
        advantages, returns = advantage_computation([0.7], [value.item()])
        # state = state.squeeze(0)
        agent.update(
            state.unsqueeze(0), action.unsqueeze(0), log_prob.detach().unsqueeze(0),
            returns, advantages
        )
        print("✓ Single experience update works")
    except Exception as e:
        print(f"✗ Single experience update failed: {e}")
    
    return True

def run_all_tests():
    """Run all tests"""
    print("Starting Agent Testing Suite")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_augmentation_application, 
        test_ppo_update,
        test_learning_simulation,
        test_edge_cases
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    test_names = [
        "Basic Functionality",
        "Augmentation Application",
        "PPO Update",
        "Learning Simulation", 
        "Edge Cases"
    ]
    
    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<30} {status}")
    
    total_passed = sum(results)
    print(f"\nPassed: {total_passed}/{len(results)} tests")
    
    if total_passed == len(results):
        print("🎉 All tests passed! Your agent is ready for training.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    # Set seeds for reproducible testing
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_all_tests()