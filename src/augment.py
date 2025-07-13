import torch
import torchvision.transforms.v2 as transforms
import numpy as np

class AugmentationSpace:
    """Simplified AutoAugment search space"""
    def __init__(self):
        # Reduced set of 6 standard augm strategies from torchvision
        self.operations = [
            'rotate', 'flip_h', 'flip_v',
            'brightness', 'contrast', 'cutout'
        ]
        self.num_ops = len(self.operations)
        self.num_magnitudes = 5  # discretized levels
        
    def apply_operation(self, image, op_idx, magnitude_idx):
        """Apply augmentation operation to image"""
        op_name = self.operations[op_idx]
        magnitude = (magnitude_idx + 1) / self.num_magnitudes  # scale 0-1
        
        # Simplified implementation - you'd expand this
        transform_map = {
            'rotate': transforms.RandomRotation(degrees=magnitude * 30),  # 6-30 degrees
            'flip_h': transforms.RandomHorizontalFlip(p=magnitude),       # 0.2-1.0 probability  
            'flip_v': transforms.RandomVerticalFlip(p=magnitude),         # 0.2-1.0 probability
            'brightness': transforms.ColorJitter(brightness=magnitude * 0.5),  # 0.1-0.5 
            'contrast': transforms.ColorJitter(contrast=magnitude * 0.5),      # 0.1-0.5
            'cutout': transforms.RandomErasing(p=1.0, scale=(0.02, magnitude * 0.4))  # area 0.02-0.4
        }
        
        return transform_map[op_name](image)

def apply_auto_augmentations(data, aug_policy, space: AugmentationSpace):
    augm_imgs = []
    for img in range(data.shape[0]):
        selected_sub_policy = np.random.choice(len(aug_policy), 2)
        selected_aug = aug_policy[selected_sub_policy]
        
        for op in selected_aug:
            random_prob = np.random.uniform(0.0, 1.0)
            
            if np.random.random() < random_prob:
                aug_img = space.apply_operation(data[img], op[0], op[1])
            augm_imgs.append(aug_img.unsqueeze(0))
            
    return torch.vstack(augm_imgs)

def apply_random_augmentations(data, space: AugmentationSpace, num_ops_per_policy=2):
    """Random baseline that mimics AutoAugment policy structure"""
    augm_imgs = []
    
    for i in range(data.shape[0]):
        img = data[i]
        augmented_img = img
        
        # Apply num_ops_per_policy random operations (same as policy)
        for _ in range(num_ops_per_policy):
            selected_op = np.random.choice(space.num_ops)
            selected_magn = np.random.choice(space.num_magnitudes)
            random_prob = np.random.uniform(0.0, 1.0)
            
            if np.random.random() < random_prob:
                augmented_img = space.apply_operation(
                    augmented_img, selected_op, selected_magn
                )
        
        augm_imgs.append(augmented_img.unsqueeze(0))
    
    return torch.vstack(augm_imgs)