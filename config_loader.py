import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EnvironmentConfig:
    name: str
    input_size: tuple
    frame_crop: Dict[str, int]
    render_mode: str  # "human" or "rgb_array"

@dataclass
class TrainingConfig:
    learning_rate: float
    gamma: float
    episodes: int
    batch_size: int
    epsilon: float
    alpha: float
    beta: float
    max_iterations: int
    max_updates: int
    kl_limit: float
    batch_steps: int
    save_interval: int
    gradient_clip: float

@dataclass
class ModelConfig:
    conv_layers: list
    fc_layers: list

@dataclass
class PretrainedConfig:
    use_pretrained: bool
    model_path: Optional[str]
    freeze_layers: List[str]
    learning_rate_multiplier: float

@dataclass
class LoggingConfig:
    use_wandb: bool
    wandb: Dict[str, str]
    metrics: list

@dataclass
class PathsConfig:
    model_save: str

@dataclass
class Config:
    environment: EnvironmentConfig
    training: TrainingConfig
    model: ModelConfig
    pretrained: PretrainedConfig
    logging: LoggingConfig
    paths: PathsConfig

def load_config(config_path: str = "config.yaml") -> Config:
    """Load and parse the configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config object containing all configuration parameters
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert input_size list to tuple
    config_dict['environment']['input_size'] = tuple(config_dict['environment']['input_size'])
    
    # Create Config object with nested dataclasses
    return Config(
        environment=EnvironmentConfig(**config_dict['environment']),
        training=TrainingConfig(**config_dict['training']),
        model=ModelConfig(**config_dict['model']),
        pretrained=PretrainedConfig(**config_dict['pretrained']),
        logging=LoggingConfig(**config_dict['logging']),
        paths=PathsConfig(**config_dict['paths'])
    )

def get_config() -> Config:
    """Get the configuration, creating default if it doesn't exist.
    
    Returns:
        Config object containing all configuration parameters
    """
    config_path = Path("config.yaml")
    if not config_path.exists():
        # Create default config if it doesn't exist
        with open(config_path, 'w') as f:
            yaml.dump({
                'environment': {
                    'name': 'BreakoutDeterministic-v4',
                    'input_size': [84, 84],
                    'frame_crop': {
                        'top': 35,
                        'bottom': 210,
                        'left': 0,
                        'right': 160
                    },
                    'render_mode': 'human'
                },
                'training': {
                    'learning_rate': 0.0001,
                    'gamma': 0.99,
                    'episodes': 5000,
                    'batch_size': 5,
                    'epsilon': 0.2,
                    'alpha': 0.01,
                    'beta': 0.5,
                    'max_iterations': 4,
                    'max_updates': 15000,
                    'kl_limit': 0.015,
                    'batch_steps': 2048,
                    'save_interval': 500,
                    'gradient_clip': 0.5
                },
                'model': {
                    'conv_layers': [
                        {'in_channels': 4, 'out_channels': 32, 'kernel_size': 8, 'stride': 4},
                        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 4, 'stride': 2},
                        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1}
                    ],
                    'fc_layers': [
                        {'in_features': 3136, 'out_features': 512},
                        {'policy_out': None},
                        {'value_out': 1}
                    ]
                },
                'pretrained': {
                    'use_pretrained': False,
                    'model_path': '{game}_PPO_WEIGHTS.pth',
                    'freeze_layers': [],
                    'learning_rate_multiplier': 1.0
                },
                'logging': {
                    'use_wandb': False,
                    'wandb': {
                        'project': 'PPO_PONG_BreakoutDeterministic-v4',
                        'entity': 'neuroori'
                    },
                    'metrics': [
                        'RUNNING REWARD',
                        'TOTAL LOSS',
                        'ENTROPY LOSS',
                        'VALUES LOSS',
                        'POLICY LOSS',
                        'KL_APPROX MEAN'
                    ]
                },
                'paths': {
                    'model_save': '{game}_PPO_WEIGHTS.pth'
                }
            }, f)
    
    return load_config(str(config_path)) 