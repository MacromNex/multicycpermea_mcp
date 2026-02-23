#!/usr/bin/env python3
"""
Use Case 3: Predict Membrane Permeability of Cyclic Peptides

This script uses a trained MultiCycPermea model to predict the membrane permeability of
cyclic peptides from their SMILES sequences and/or molecular images. It supports both
single peptide prediction and batch prediction.

Usage:
    python examples/use_case_3_predict_permeability.py --model_path models/best_model.pt --input examples/data/sequences/test.csv

Environment: Use ./env_py37 (Python 3.7 environment)

Requirements:
    - Trained MultiCycPermea model
    - PyTorch with CUDA support
    - RDKit for molecular handling
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

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

def predict_single_peptide(smiles, model, vocab, image_path=None):
    """
    Predict permeability for a single cyclic peptide.

    Args:
        smiles (str): SMILES string of the cyclic peptide
        model: Trained MultiCycPermea model
        vocab: Tokenizer/vocabulary
        image_path (str, optional): Path to 2D molecular image

    Returns:
        float: Predicted permeability value
    """
    try:
        # Import required modules
        from dataset import SMILESDataset, collate_wrapper
        from torch.utils.data import DataLoader
        import torch

        # Create single-item dataset
        dataset = SMILESDataset(
            [smiles], [0],  # dummy target
            vocab, ['dummy_id'],
            image_folder=os.path.dirname(image_path) if image_path else None,
            text_data_config={'data_type': 'SMILES', 'max_len': 250},
            image_size=224,
            image_augment=False,
            smiles_augment=False
        )

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wrapper(vocab))

        # Predict
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            for batch_idx, (data, target, length, image, fg, weight, peptide_id) in enumerate(dataloader):
                data = data.to(device)
                image = image.to(device) if image[0] is not None else image
                fg = fg.to(device)
                weight = weight.to(device)

                output, _, _, _ = model(data, image, length, fg, weight=None, return_feature=True)
                prediction = output.cpu().item()

                return prediction

    except Exception as e:
        print(f"Error predicting for SMILES {smiles}: {e}")
        return None

def predict_batch(input_file, model_path, config_path, output_file=None):
    """
    Predict permeability for a batch of cyclic peptides from CSV file.

    Args:
        input_file (str): Path to CSV file with SMILES and optionally image paths
        model_path (str): Path to trained model
        config_path (str): Path to model configuration
        output_file (str, optional): Path to save predictions

    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    try:
        # Import required modules
        import torch
        import yaml
        from transformers import AutoTokenizer
        from model import Peptide_Regression
        from utils import get_vocabulary

        print(f"Loading model from: {model_path}")

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load data
        print(f"Loading data from: {input_file}")
        df = pd.read_csv(input_file)

        if 'SMILES' not in df.columns:
            raise ValueError("Input CSV must contain 'SMILES' column")

        # Setup model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize vocabulary/tokenizer
        text_model_type = config.get('model_yaml', {}).get('text_model_type', 'custom')

        if text_model_type == 'PubChemLM':
            vocab = AutoTokenizer.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_150k")
        elif text_model_type == 'ChemBERTa':
            vocab = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        else:
            # Use custom vocabulary
            vocab = get_vocabulary(df['SMILES'].tolist(), {'data_type': 'SMILES', 'max_len': 250})

        # Initialize model
        model = Peptide_Regression(
            all_model_config=config['model_yaml'],
            text_vocab=vocab,
            text_model_config=config['text_model_config'],
            text_data_config=config['text_data_config'],
            image_model_config=config['image_model_config'],
            image_data_config=config['image_data_config']
        )

        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        print("Model loaded successfully")

        # Make predictions
        predictions = []
        print(f"Predicting permeability for {len(df)} peptides...")

        for idx, row in df.iterrows():
            smiles = row['SMILES']
            image_path = row.get('image_path', None)

            pred = predict_single_peptide(smiles, model, vocab, image_path)
            predictions.append(pred)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} peptides")

        # Add predictions to dataframe
        df['predicted_permeability'] = predictions

        # Calculate metrics if true values available
        if 'Permeability' in df.columns:
            from utils import r2_score, mse, mae, rmse, pearson_correlation_coefficient

            true_values = df['Permeability'].values
            pred_values = np.array([p for p in predictions if p is not None])

            if len(pred_values) > 0:
                r2 = r2_score(true_values[:len(pred_values)], pred_values)
                mse_val = mse(true_values[:len(pred_values)], pred_values)
                mae_val = mae(true_values[:len(pred_values)], pred_values)
                rmse_val = rmse(true_values[:len(pred_values)], pred_values)
                pcc = pearson_correlation_coefficient(true_values[:len(pred_values)], pred_values)

                print(f"\nPrediction Metrics:")
                print(f"  R² Score: {r2:.4f}")
                print(f"  MSE: {mse_val:.4f}")
                print(f"  MAE: {mae_val:.4f}")
                print(f"  RMSE: {rmse_val:.4f}")
                print(f"  Pearson Correlation: {pcc:.4f}")

        # Save results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Predictions saved to: {output_file}")

        return df

    except Exception as e:
        print(f"Error in batch prediction: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Predict cyclic peptide membrane permeability')

    parser.add_argument('--model_path', '-m', required=True,
                       help='Path to trained MultiCycPermea model (.pt file)')
    parser.add_argument('--config', '-c',
                       default='examples/config/model.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--input', '-i',
                       help='Input CSV file with SMILES column')
    parser.add_argument('--smiles', '-s',
                       help='Single SMILES string to predict')
    parser.add_argument('--output', '-o',
                       help='Output CSV file for batch predictions')
    parser.add_argument('--image_folder',
                       help='Folder containing molecular images (for 2D features)')

    args = parser.parse_args()

    print("=" * 60)
    print("MultiCycPermea Prediction - Cyclic Peptide Permeability")
    print("=" * 60)

    # Setup environment
    setup_environment()

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("\nTo train a model, use:")
        print("  python examples/use_case_2_train_multicycpermea.py")
        print("\nOr download a pre-trained model from the original repository.")
        return

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print("Create a config file or use the training script to generate one.")
        return

    try:
        # Single SMILES prediction
        if args.smiles:
            print(f"Predicting permeability for SMILES: {args.smiles}")
            # TODO: Implement single prediction
            print("Note: Single SMILES prediction requires complete model setup.")
            print("Use batch prediction with a CSV file for now.")

        # Batch prediction
        elif args.input:
            if not os.path.exists(args.input):
                print(f"Error: Input file not found: {args.input}")
                print("Available data files:")
                data_dir = Path("examples/data/sequences/")
                if data_dir.exists():
                    for csv_file in data_dir.glob("*.csv"):
                        print(f"  {csv_file}")
                return

            output_file = args.output or args.input.replace('.csv', '_predictions.csv')
            df = predict_batch(args.input, args.model_path, args.config, output_file)

            if df is not None:
                print(f"\nPrediction complete!")
                successful_predictions = df['predicted_permeability'].notna().sum()
                print(f"Successfully predicted: {successful_predictions}/{len(df)} peptides")

        else:
            print("Error: Provide either --smiles for single prediction or --input for batch prediction")
            parser.print_help()

    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Make sure you're using the Python 3.7 environment with all dependencies installed:")
        print("  mamba activate ./env_py37")

if __name__ == '__main__':
    main()