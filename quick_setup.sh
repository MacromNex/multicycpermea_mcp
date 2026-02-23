#!/bin/bash
# Quick Setup Script for MultiCycPermea MCP
# MultiCycPermea: Membrane permeability prediction using cyclic peptide images and SMILES
# Multimodal model combining image and text features
# Source: https://github.com/viko-3/MultiCycPermea

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up MultiCycPermea MCP ==="

# Step 1: Create Python environment
echo "[1/5] Creating Python 3.10 environment..."
(command -v mamba >/dev/null 2>&1 && mamba create -p ./env python=3.10 pip -y) || \
(command -v conda >/dev/null 2>&1 && conda create -p ./env python=3.10 pip -y) || \
(echo "Warning: Neither mamba nor conda found, creating venv instead" && python3 -m venv ./env)

# Step 2: Install core dependencies
echo "[2/5] Installing core dependencies..."
./env/bin/pip install loguru click pandas numpy tqdm

# Step 3: Install fastmcp
echo "[3/5] Installing fastmcp..."
./env/bin/pip install --force-reinstall --no-cache-dir fastmcp

# Step 4: Install RDKit
echo "[4/5] Installing RDKit..."
(command -v mamba >/dev/null 2>&1 && mamba install -p ./env -c conda-forge rdkit -y) || \
(command -v conda >/dev/null 2>&1 && conda install -p ./env -c conda-forge rdkit -y) || \
./env/bin/pip install rdkit

# Step 5: Install image processing packages
echo "[5/5] Installing image processing packages..."
./env/bin/pip install cairosvg matplotlib seaborn

echo ""
echo "=== MultiCycPermea MCP Setup Complete ==="
echo "Note: For pretrained models, download from:"
echo "https://www.dropbox.com/scl/fo/bhl86a9cjjl6enweowola/APV6blUE2Q41_08M0l4fIJA"
echo "To run the MCP server: ./env/bin/python src/server.py"
