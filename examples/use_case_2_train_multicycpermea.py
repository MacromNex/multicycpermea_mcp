#!/usr/bin/env python3
"""
Use Case 2: Train MultiCycPermea Model for Permeability Prediction

This script trains the MultiCycPermea model using both 1D SMILES sequences and 2D molecular
images to predict membrane permeability of cyclic peptides. It combines text and image features
using different fusion strategies.

Usage:
    python examples/use_case_2_train_multicycpermea.py --use_text_info True --use_image_info True --feature_cmb_type concate

Environment: Use ./env_py37 (Python 3.7 environment)

Requirements:
    - PyTorch with CUDA support
    - Transformers library
    - RDKit for molecular handling
    - All dependencies from environment.yml
"""

import argparse
import sys
import os
from pathlib import Path

def setup_environment():
    """Setup the Python path for importing MultiCycPermea modules."""
    # Add the repo path to Python path
    repo_path = Path(__file__).parent.parent / "repo" / "MultiCycPermea"
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))
        sys.path.insert(0, str(repo_path / "DL"))
    else:
        print(f"Error: Repository path not found: {repo_path}")
        print("Make sure you're running this from the MCP root directory")
        sys.exit(1)

def check_environment():
    """Check if we're in the correct environment."""
    try:
        import torch
        import pandas as pd
        from transformers import AutoTokenizer
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"✗ Missing required packages: {e}")
        print("Please activate the Python 3.7 environment:")
        print("  mamba activate ./env_py37")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train MultiCycPermea for cyclic peptide permeability prediction')

    # Model configuration
    parser.add_argument('--all_config', type=str,
                       default='examples/config/model.yaml',
                       help='Path to the main config file')

    # Data configuration overrides
    parser.add_argument("--text_data_yaml", type=str, default=None,
                       help="Override text data config file")
    parser.add_argument("--image_data_yaml", type=str, default=None,
                       help="Override image data config file")

    # Model configuration overrides
    parser.add_argument("--text_model_yaml", type=str, default=None,
                       help="Override text model config file")
    parser.add_argument("--image_model_yaml", type=str, default=None,
                       help="Override image model config file")

    # Feature combination settings
    parser.add_argument("--use_text_info", type=bool, default=True,
                       help="Use 1D SMILES sequence features")
    parser.add_argument("--use_image_info", type=bool, default=True,
                       help="Use 2D molecular image features")
    parser.add_argument("--feature_cmb_type", type=str, default='concate',
                       choices=['concate', 'cross_attention', 'attention'],
                       help="Feature combination method")

    # Training settings
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device number")

    args = parser.parse_args()

    print("=" * 60)
    print("MultiCycPermea Training - Cyclic Peptide Permeability Prediction")
    print("=" * 60)

    # Setup environment and paths
    setup_environment()

    if not check_environment():
        return

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    try:
        # Import MultiCycPermea modules
        from transformers import AutoTokenizer
        from dataset import SMILESDataset, collate_wrapper
        from torch.utils.data import DataLoader
        import pandas as pd
        from model import Peptide_Regression
        import torch
        import torch.nn as nn
        from utils import get_vocabulary, str2bool, define_optimizer
        from train import train
        import datetime
        from torch.utils.tensorboard import SummaryWriter
        import yaml

        print("✓ Successfully imported MultiCycPermea modules")

    except ImportError as e:
        print(f"✗ Error importing MultiCycPermea modules: {e}")
        print("Make sure you're using the Python 3.7 environment with all dependencies installed")
        return

    # Load configuration
    config_path = Path(args.all_config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Creating default config file...")

        # Create examples/config directory and default config
        config_dir = Path("examples/config")
        config_dir.mkdir(exist_ok=True)

        default_config = {
            'data_yaml': {
                'text_data_yaml': 'examples/config/smi_dataset.yaml',
                'image_data_yaml': 'examples/config/img_dataset.yaml'
            },
            'model_yaml': {
                'text_model_yaml': 'examples/config/Transformer.yaml',
                'image_model_yaml': 'examples/config/TIMM.yaml',
                'use_text_info': True,
                'use_image_info': True,
                'feature_cmb_type': 'concate'
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        print(f"Created default config at: {config_path}")
        print("You may need to create additional config files. Check the original repo/MultiCycPermea/DL/config/ for examples.")
        return

    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        all_config = yaml.safe_load(f)

    # Apply command line overrides
    if args.text_data_yaml:
        all_config['data_yaml']['text_data_yaml'] = args.text_data_yaml
    if args.image_data_yaml:
        all_config['data_yaml']['image_data_yaml'] = args.image_data_yaml
    if args.text_model_yaml:
        all_config['model_yaml']['text_model_yaml'] = args.text_model_yaml
    if args.image_model_yaml:
        all_config['model_yaml']['image_model_yaml'] = args.image_model_yaml

    all_config['model_yaml']['use_text_info'] = args.use_text_info
    all_config['model_yaml']['use_image_info'] = args.use_image_info
    all_config['model_yaml']['feature_cmb_type'] = args.feature_cmb_type

    print(f"Configuration:")
    print(f"  - Use 1D SMILES features: {args.use_text_info}")
    print(f"  - Use 2D image features: {args.use_image_info}")
    print(f"  - Feature combination: {args.feature_cmb_type}")

    # Setup logging
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path("examples/logs")
    log_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(f'examples/logs/runs/{current_time}')
    log_file = f"examples/logs/runs/{current_time}.txt"

    with open(log_file, "w") as f:
        f.write("MultiCycPermea Training Configuration:\n")
        f.write(yaml.dump(all_config))

    print(f"Logging to: {log_file}")

    # TODO: Continue with actual training implementation
    # This would require the full config files and data setup
    print("\nNote: This example shows the training setup structure.")
    print("To run full training, you need:")
    print("1. Complete config files in examples/config/")
    print("2. Training data with both SMILES and image paths")
    print("3. Pre-trained model weights (if using transfer learning)")
    print("4. Sufficient GPU memory for training")

    print(f"\nExample training command:")
    print(f"CUDA_VISIBLE_DEVICES=0 python examples/use_case_2_train_multicycpermea.py \\")
    print(f"  --use_text_info True \\")
    print(f"  --use_image_info True \\")
    print(f"  --feature_cmb_type concate")

if __name__ == '__main__':
    main()