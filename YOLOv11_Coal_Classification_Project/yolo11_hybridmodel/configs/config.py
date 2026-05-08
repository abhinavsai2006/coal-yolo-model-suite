"""
Configuration for YOLO11 Hybrid Model
"""

# Model configuration
MODEL_CONFIG = {
    'num_classes': 6,
    'channels_list': [64, 128, 256, 512],
    'input_size': 224,
    'reduction_ratio': 16,
    'mci_scales': [1, 3, 5],
}

# Training configuration
TRAIN_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'label_smoothing': 0.1,
    'optimizer': 'adamw',  # 'adam', 'adamw', 'sgd'
    'scheduler': 'cosine',  # 'cosine', 'step', 'none'
}

# Data configuration
DATA_CONFIG = {
    'train_dir': '../data/train',
    'val_dir': '../data/val',
    'test_dir': '../data/test',
    'input_size': 224,
    'augment': True,
    'num_workers': 4,
}

# Class names
CLASS_NAMES = [
    'destructive_coal',
    'fully_pulverized_coal',
    'non_destructive_coal',
    'not_coal',
    'pulverized_coal',
    'strongly_destructive_coal'
]

# Output directories
OUTPUT_CONFIG = {
    'runs_dir': './runs',
    'eval_dir': './evaluation',
    'weights_dir': './weights',
}
