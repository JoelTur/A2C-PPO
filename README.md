# A2C-PPO

A PyTorch implementation of Proximal Policy Optimization (PPO) for Atari games. This implementation follows the paper [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf).

## Features

- PPO implementation with clipping and KL divergence early stopping
- Support for Atari games through Gymnasium
- Configurable hyperparameters through YAML
- Optional Weights & Biases integration for experiment tracking
- Support for pretrained model loading and fine-tuning
- Configurable rendering modes (human/rgb_array)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/A2C-PPO.git
cd A2C-PPO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses a YAML configuration file (`config.yaml`) for all settings. Key configuration sections:

### Environment Settings
```yaml
environment:
  name: "BreakoutDeterministic-v4"  # Gym environment name
  input_size: [84, 84]              # Size of processed input frames
  frame_crop:                       # Frame cropping parameters
    top: 35
    bottom: 210
    left: 0
    right: 160
  render_mode: "human"              # Rendering mode: "human" or "rgb_array"
```

### Training Parameters
```yaml
training:
  learning_rate: 0.0001
  gamma: 0.99                       # Discount factor
  episodes: 5000
  batch_size: 5
  epsilon: 0.2                      # PPO clipping parameter
  alpha: 0.01                       # Entropy coefficient
  beta: 0.5                         # Value loss coefficient
  max_iterations: 4                 # Maximum PPO iterations
  max_updates: 15000               # Maximum training updates
  kl_limit: 0.015                   # KL divergence limit
  batch_steps: 2048                 # Steps per batch update
  save_interval: 500                # Save model every N updates
  gradient_clip: 0.5                # Gradient clipping value
```

### Pretrained Model Settings
```yaml
pretrained:
  use_pretrained: false            # Whether to use a pretrained model
  model_path: null                 # Path to pretrained model (if null, will use default path)
  freeze_layers: []                # List of layer names to freeze during training
  learning_rate_multiplier: 1.0    # Learning rate multiplier for fine-tuning
```

### Logging Settings
```yaml
logging:
  use_wandb: false                  # Whether to use Weights & Biases for logging
  wandb:
    project: "PPO_PONG_BreakoutDeterministic-v4"
    entity: "neuroori"
```

## Usage

### Training

To train a new model:
```bash
python PPO_TRAIN.py
```

The training script will:
1. Load configuration from `config.yaml`
2. Initialize the environment and model
3. Train the model using PPO
4. Save checkpoints periodically
5. Log metrics to Weights & Biases if enabled

### Testing

To test a trained model:
```bash
python PPO_TEST.py
```

The testing script will:
1. Load the trained model
2. Run episodes with the trained policy
3. Display the game window
4. Log metrics to Weights & Biases if enabled

## Model Architecture

The model uses a CNN architecture:
- 3 convolutional layers
- 2 fully connected layers (policy and value heads)
- ReLU activation
- LogSoftmax for policy output

## Results

| Game          | Score         |        
| ------------- | ------------- |
| BREAKOUT      | 295           |
|               |               |
|               |               |
|               |               |

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
