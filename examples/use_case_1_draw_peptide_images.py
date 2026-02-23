#!/usr/bin/env python3
"""
Use Case 1: Draw Cyclic Peptide 2D Structure Images from SMILES

This script generates 2D molecular structure images from SMILES strings of cyclic peptides.
This is essential for creating the image features used in MultiCycPermea for membrane
permeability prediction.

Usage:
    python examples/use_case_1_draw_peptide_images.py --input examples/data/sequences/test.csv --output examples/data/images/

Requirements:
    - RDKit for molecular visualization
    - cairosvg for SVG to PNG conversion
    - pandas for data handling
"""

import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen, rdDepictor
import cairosvg
import os
from pathlib import Path

def draw_peptide_images(csv_file, output_folder, image_size=(600, 600)):
    """
    Generate 2D structure images for cyclic peptides from SMILES.

    Args:
        csv_file (str): Path to CSV file with SMILES and CycPeptMPDB_ID columns
        output_folder (str): Directory to save generated images
        image_size (tuple): Image dimensions (width, height)
    """
    print(f"Reading data from: {csv_file}")
    df = pd.read_csv(csv_file)

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving images to: {output_folder}")

    # Configure RDKit for better 2D coordinates
    rdDepictor.SetPreferCoordGen(True)

    generated_count = 0
    failed_count = 0

    for index, row in df.iterrows():
        try:
            smiles = row['SMILES']
            peptide_id = row['CycPeptMPDB_ID']
            image_path = os.path.join(output_folder, f"{peptide_id}.png")

            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Could not parse SMILES for {peptide_id}: {smiles}")
                failed_count += 1
                continue

            # Generate 2D coordinates
            rdCoordGen.AddCoords(mol)

            # Create SVG drawing
            view = Draw.rdMolDraw2D.MolDraw2DSVG(image_size[0], image_size[1])
            view.DrawMolecule(Draw.rdMolDraw2D.PrepareMolForDrawing(mol))
            view.FinishDrawing()
            svg = view.GetDrawingText()

            # Save as temporary SVG file
            svg_file_path = f"./temp_molecule_{index}.svg"
            with open(svg_file_path, "w") as f:
                f.write(svg)

            # Convert SVG to PNG
            cairosvg.svg2png(url=svg_file_path, write_to=image_path)

            # Clean up temporary SVG
            os.remove(svg_file_path)

            generated_count += 1
            if generated_count % 100 == 0:
                print(f"Generated {generated_count} images...")

        except Exception as e:
            print(f"Error processing {peptide_id}: {str(e)}")
            failed_count += 1
            continue

    print(f"\nImage generation complete!")
    print(f"Successfully generated: {generated_count} images")
    print(f"Failed to generate: {failed_count} images")

    # Update the dataframe with image paths
    df['image_path'] = df['CycPeptMPDB_ID'].apply(
        lambda x: os.path.join(output_folder, f"{x}.png")
    )

    # Save updated CSV
    output_csv = csv_file.replace('.csv', '_with_images.csv')
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to: {output_csv}")

    return df

def main():
    parser = argparse.ArgumentParser(description='Generate 2D structure images for cyclic peptides')
    parser.add_argument('--input', '-i',
                       default='examples/data/sequences/test.csv',
                       help='Input CSV file with SMILES and CycPeptMPDB_ID columns')
    parser.add_argument('--output', '-o',
                       default='examples/data/images/',
                       help='Output directory for generated images')
    parser.add_argument('--size', '-s',
                       default='600,600',
                       help='Image size as width,height (default: 600,600)')

    args = parser.parse_args()

    # Parse image size
    try:
        width, height = map(int, args.size.split(','))
        image_size = (width, height)
    except:
        print("Error: Invalid size format. Use width,height (e.g., 600,600)")
        return

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("Available data files:")
        data_dir = Path("examples/data/sequences/")
        if data_dir.exists():
            for csv_file in data_dir.glob("*.csv"):
                print(f"  {csv_file}")
        return

    # Generate images
    draw_peptide_images(args.input, args.output, image_size)

if __name__ == '__main__':
    main()