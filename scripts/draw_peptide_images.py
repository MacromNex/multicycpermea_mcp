#!/usr/bin/env python3
"""
Script: draw_peptide_images.py
Description: Generate 2D molecular structure images for cyclic peptides from SMILES

Original Use Case: examples/use_case_1_draw_peptide_images.py
Dependencies Removed: None (all essential)

Usage:
    python scripts/draw_peptide_images.py --input <input_file> --output <output_dir>

Example:
    python scripts/draw_peptide_images.py --input examples/data/sequences/test_small.csv --output results/images
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple

# Essential scientific packages
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen, rdDepictor
import cairosvg

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "image_size": [600, 600],
    "image_format": "png",
    "use_coord_gen": True,
    "batch_size": 100,
    "temp_dir": "./temp",
    "cleanup_temp_files": True
}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def validate_input_file(file_path: Path) -> bool:
    """Validate that input file exists and has required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        required_columns = ['SMILES', 'CycPeptMPDB_ID']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return True
    except Exception as e:
        raise ValueError(f"Error reading input file: {e}")

def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES string to RDKit molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol

def generate_2d_coords(mol: Chem.Mol, use_coord_gen: bool = True) -> Chem.Mol:
    """Generate 2D coordinates for molecule."""
    if use_coord_gen:
        rdCoordGen.AddCoords(mol)
    else:
        rdDepictor.Compute2DCoords(mol)
    return mol

def create_molecule_image(mol: Chem.Mol, image_size: Tuple[int, int]) -> str:
    """Create SVG image of molecule."""
    view = Draw.rdMolDraw2D.MolDraw2DSVG(image_size[0], image_size[1])
    view.DrawMolecule(Draw.rdMolDraw2D.PrepareMolForDrawing(mol))
    view.FinishDrawing()
    return view.GetDrawingText()

def save_image(svg_content: str, output_path: Path, temp_dir: Path) -> None:
    """Save SVG content as PNG image."""
    # Create temp SVG file
    temp_svg = temp_dir / f"temp_{output_path.stem}.svg"
    with open(temp_svg, "w") as f:
        f.write(svg_content)

    # Convert to PNG
    cairosvg.svg2png(url=str(temp_svg), write_to=str(output_path))

    # Cleanup temp file
    temp_svg.unlink()

def update_csv_with_image_paths(input_file: Path, output_dir: Path) -> Path:
    """Update input CSV with image path column."""
    df = pd.read_csv(input_file)
    df['image_path'] = df['CycPeptMPDB_ID'].apply(
        lambda x: str(output_dir / f"{x}.png")
    )

    output_csv = input_file.parent / f"{input_file.stem}_with_images.csv"
    df.to_csv(output_csv, index=False)
    return output_csv

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_draw_peptide_images(
    input_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate 2D molecular structure images for cyclic peptides from SMILES.

    Args:
        input_file: Path to CSV file with SMILES and CycPeptMPDB_ID columns
        output_dir: Directory to save generated images (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - generated_count: Number of successfully generated images
            - failed_count: Number of failed image generations
            - output_dir: Path to output directory
            - updated_csv: Path to updated CSV with image paths
            - metadata: Execution metadata

    Example:
        >>> result = run_draw_peptide_images("input.csv", "output_images/")
        >>> print(f"Generated {result['generated_count']} images")
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate input
    validate_input_file(input_file)

    # Set output directory
    if output_dir is None:
        output_dir = input_file.parent / "images"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup temp directory
    temp_dir = Path(config["temp_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Saving images to: {output_dir}")

    # Configure RDKit
    if config["use_coord_gen"]:
        rdDepictor.SetPreferCoordGen(True)

    generated_count = 0
    failed_count = 0
    image_size = tuple(config["image_size"])

    for index, row in df.iterrows():
        try:
            smiles = row['SMILES']
            peptide_id = row['CycPeptMPDB_ID']
            image_path = output_dir / f"{peptide_id}.{config['image_format']}"

            # Parse SMILES
            mol = parse_smiles(smiles)
            if mol is None:
                print(f"Warning: Could not parse SMILES for {peptide_id}: {smiles}")
                failed_count += 1
                continue

            # Generate 2D coordinates
            mol = generate_2d_coords(mol, config["use_coord_gen"])

            # Create image
            svg_content = create_molecule_image(mol, image_size)

            # Save image
            save_image(svg_content, image_path, temp_dir)

            generated_count += 1
            if generated_count % config["batch_size"] == 0:
                print(f"Generated {generated_count} images...")

        except Exception as e:
            print(f"Error processing {peptide_id}: {str(e)}")
            failed_count += 1
            continue

    print(f"\nImage generation complete!")
    print(f"Successfully generated: {generated_count} images")
    print(f"Failed to generate: {failed_count} images")

    # Update CSV with image paths
    updated_csv = update_csv_with_image_paths(input_file, output_dir)
    print(f"Updated CSV saved to: {updated_csv}")

    # Cleanup temp directory if requested
    if config["cleanup_temp_files"] and temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass  # Directory not empty, leave it

    return {
        "generated_count": generated_count,
        "failed_count": failed_count,
        "output_dir": str(output_dir),
        "updated_csv": str(updated_csv),
        "metadata": {
            "input_file": str(input_file),
            "config": config,
            "total_peptides": len(df)
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file with SMILES and CycPeptMPDB_ID columns')
    parser.add_argument('--output', '-o',
                       help='Output directory for generated images')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')
    parser.add_argument('--size', '-s', default='600,600',
                       help='Image size as width,height (default: 600,600)')
    parser.add_argument('--format', '-f', default='png',
                       choices=['png', 'svg'],
                       help='Output image format (default: png)')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Parse image size
    try:
        width, height = map(int, args.size.split(','))
        size_override = {"image_size": [width, height]}
    except:
        print("Error: Invalid size format. Use width,height (e.g., 600,600)")
        return

    # Run
    result = run_draw_peptide_images(
        input_file=args.input,
        output_dir=args.output,
        config=config,
        image_format=args.format,
        **size_override
    )

    print(f"Success: Generated {result['generated_count']} images in {result['output_dir']}")
    return result

if __name__ == '__main__':
    main()