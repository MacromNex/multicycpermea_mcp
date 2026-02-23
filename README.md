# MultiCycPermea MCP

> MCP tools for cyclic peptide computational analysis - Generate molecular images, predict membrane permeability, and analyze feature fusion methods using MultiCycPermea

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

MultiCycPermea MCP provides computational tools for cyclic peptide membrane permeability prediction through a standardized Model Context Protocol (MCP) interface. This implementation combines 1D sequence features (SMILES) and 2D molecular image features for accurate permeability prediction and comprehensive molecular analysis.

### Features
- **2D Molecular Structure Generation** for cyclic peptides with high-quality PNG/SVG output
- **Feature Fusion Analysis** comparing concatenation, cross-attention, and attention-based methods
- **Membrane Permeability Prediction** using deep learning models trained on CycPeptMPDB
- **Batch Processing** for virtual screening of large cyclic peptide libraries
- **Dual API Design** with synchronous (fast) and asynchronous (background) processing

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment (Python 3.10)
├── src/
│   └── server.py           # MCP server (14 tools)
├── scripts/
│   ├── draw_peptide_images.py      # Generate 2D molecular images
│   ├── feature_analysis.py         # Analyze feature fusion methods
│   └── lib/                # Shared utilities (24 functions)
├── examples/
│   └── data/               # Demo data
│       ├── sequences/      # Sample cyclic peptide datasets
│       │   ├── test_small.csv      # Small test dataset (5 peptides)
│       │   ├── test.csv            # Test dataset (~1000 peptides)
│       │   ├── train.csv           # Training dataset (5559+ peptides)
│       │   └── val.csv             # Validation dataset
│       └── images/         # Generated molecular images
├── configs/                # Configuration files
│   ├── draw_peptide_images_config.json
│   ├── feature_analysis_config.json
│   └── default_config.json
└── repo/                   # Original repository
```

---

## Installation

### Quick Setup

Run the automated setup script:

```bash
./quick_setup.sh
```

This will create the environment and install all dependencies automatically.

### Manual Setup (Advanced)

For manual installation or customization, follow these steps.

#### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- RDKit (installed automatically)
- Linux/macOS (tested on Linux)
- ~2GB RAM for typical operations

#### Create Environment
Please strictly follow the information in `reports/step3_environment.md` to obtain the procedure to setup the environment. The complete workflow is shown below.

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/multicycpermea_mcp

# Determine package manager (prefer mamba over conda)
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
else
    PKG_MGR="conda"
fi
echo "Using package manager: $PKG_MGR"

# Create conda environment (use mamba if available)
$PKG_MGR create -p ./env python=3.10 pip -y

# Activate environment
$PKG_MGR activate ./env

# Install core dependencies
$PKG_MGR run -p ./env pip install loguru click pandas numpy tqdm

# Install FastMCP (force reinstall for clean installation)
$PKG_MGR run -p ./env pip install --force-reinstall --no-cache-dir fastmcp

# Install RDKit via conda-forge (recommended method)
$PKG_MGR install -p ./env -c conda-forge rdkit -y

# Install additional dependencies for image generation
$PKG_MGR run -p ./env pip install cairosvg matplotlib seaborn
```

### Legacy Environment (Optional)
For running original MultiCycPermea training code:

```bash
# Create legacy environment from original environment.yml
$PKG_MGR env create -p ./env_py37 -f repo/MultiCycPermea/environment.yml
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/draw_peptide_images.py` | Generate 2D molecular structure images from SMILES | See below |
| `scripts/feature_analysis.py` | Analyze and compare feature fusion methods | See below |

### Script Examples

#### Generate 2D Molecular Images

```bash
# Activate environment
$PKG_MGR activate ./env

# Run script
python scripts/draw_peptide_images.py \
  --input examples/data/sequences/test_small.csv \
  --output results/images \
  --size 600,600 \
  --format png
```

**Parameters:**
- `--input, -i`: CSV file with SMILES and CycPeptMPDB_ID columns (required)
- `--output, -o`: Output directory path (default: results/images/)
- `--size, -s`: Image dimensions as "width,height" (default: 600,600)
- `--format, -f`: Output format (png, svg) (default: png)
- `--config`: JSON configuration file (optional)

**Expected Output:**
- High-quality PNG images (20-30KB each)
- Updated CSV with image_path column
- ~1.3 seconds for 5 peptides

#### Analyze Feature Fusion Methods

```bash
python scripts/feature_analysis.py \
  --input examples/data/sequences/test_small.csv \
  --output results/analysis \
  --methods concate,cross_attention,attention
```

**Parameters:**
- `--input, -i`: CSV file with SMILES column (required)
- `--output, -o`: Output directory for analysis (default: results/analysis/)
- `--methods, -m`: Comma-separated fusion methods (default: concate,cross_attention)
- `--seed`: Random seed for reproducibility (default: 42)
- `--config`: JSON configuration file (optional)

**Expected Output:**
- Performance comparison plots (PNG, 150-250KB each)
- Feature importance heatmaps
- Computational cost analysis
- Comprehensive markdown report
- ~3.4 seconds for 5 peptides

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name cycpep-tools
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "cycpep-tools": {
      "command": "/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/multicycpermea_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/multicycpermea_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from cycpep-tools?
```

#### Property Calculation (Fast)
```
Generate molecular images for @examples/data/sequences/test_small.csv
Save the results to @results/my_images/
```

#### Feature Analysis (Fast)
```
Analyze feature fusion methods for the peptides in @examples/data/sequences/test_small.csv
Compare concatenation and cross-attention methods
```

#### Data Validation
```
Validate the CSV file at @examples/data/sequences/test.csv
Check for required columns and SMILES validity
```

#### Batch Processing (Submit API)
```
Submit a batch image generation job for @examples/data/sequences/test.csv
Use high resolution 800x800 images
Name the job "test_dataset_imaging"
```

#### Check Job Status
```
Check the status of job abc12345 and show me the log output
```

#### End-to-End Workflow
```
I want to analyze the peptides in @examples/data/sequences/test_small.csv:
1. First validate the data
2. Generate molecular images
3. Analyze feature fusion methods
4. Show me the best performing method
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/sequences/test_small.csv` | Reference small test dataset |
| `@examples/data/sequences/test.csv` | Reference full test dataset |
| `@configs/draw_peptide_images_config.json` | Reference image config |
| `@configs/feature_analysis_config.json` | Reference analysis config |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "cycpep-tools": {
      "command": "/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/multicycpermea_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/multicycpermea_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What cyclic peptide analysis tools are available?
> Generate images for the peptides in test_small.csv
> Compare feature fusion methods for my dataset
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 seconds):

| Tool | Description | Parameters |
|------|-------------|------------|
| `generate_peptide_images` | Generate 2D molecular structure images | `input_file`, `output_dir`, `image_size`, `image_format`, `use_coord_gen` |
| `analyze_peptide_features` | Compare feature fusion methods | `input_file`, `output_dir`, `methods`, `random_seed`, `plot_dpi` |
| `validate_peptide_csv` | Validate CSV format and SMILES | `input_file`, `required_columns` |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes or large datasets):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_batch_image_generation` | Batch molecular image generation | `input_file`, `output_dir`, `image_size`, `image_format`, `job_name` |
| `submit_batch_feature_analysis` | Batch feature analysis | `input_file`, `output_dir`, `methods`, `random_seed`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress and status |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs with filters |
| `cleanup_old_jobs` | Remove old completed jobs |

### Utility Tools

| Tool | Description |
|------|-------------|
| `get_server_info` | Get server status and tool information |
| `get_example_data_info` | List available datasets with statistics |
| `load_config_template` | Load configuration templates |

---

## Examples

### Example 1: Quick Molecular Visualization

**Goal:** Generate 2D molecular images for a small dataset

**Using Script:**
```bash
$PKG_MGR activate ./env
python scripts/draw_peptide_images.py \
  --input examples/data/sequences/test_small.csv \
  --output results/quick_images \
  --size 800,800
```

**Using MCP (in Claude Code):**
```
Generate molecular images for @examples/data/sequences/test_small.csv
Use high resolution 800x800 images and save to @results/quick_images/
```

**Expected Output:**
- 5 high-quality PNG images (800x800 pixels)
- Updated CSV with image_path column
- Processing time: ~1.3 seconds

### Example 2: Feature Fusion Analysis

**Goal:** Compare different fusion methods for combining molecular features

**Using Script:**
```bash
$PKG_MGR activate ./env
python scripts/feature_analysis.py \
  --input examples/data/sequences/test_small.csv \
  --output results/fusion_analysis \
  --methods concate,cross_attention,attention
```

**Using MCP (in Claude Code):**
```
Analyze feature fusion methods for @examples/data/sequences/test_small.csv
Compare concatenation, cross-attention, and attention methods
Save results to @results/fusion_analysis/
```

**Expected Output:**
- Performance comparison plots
- Feature importance heatmaps
- Computational cost analysis
- Best method recommendation: cross_attention (R² = 0.796)
- Processing time: ~3.4 seconds

### Example 3: Large Dataset Processing Pipeline

**Goal:** Process a large cyclic peptide library with comprehensive analysis

**Using MCP (in Claude Code):**
```
I need to process @examples/data/sequences/test.csv for virtual screening:

1. First validate the dataset structure and SMILES
2. Submit batch image generation (600x600 PNG format)
3. Submit batch feature analysis for all three methods
4. Monitor both jobs and show me the results when complete
5. Generate a summary comparing the computational methods
```

**Expected Workflow:**
- Data validation: Immediate results
- Batch processing: Background jobs with job IDs
- Progress monitoring: Real-time status updates
- Results: Comprehensive analysis of ~1000 peptides

### Example 4: Configuration-Based Processing

**Goal:** Use custom configurations for specialized analysis

**Using MCP (in Claude Code):**
```
Load the configuration from @configs/feature_analysis_config.json
Analyze peptides in @examples/data/sequences/test_small.csv using those settings
Show me the performance metrics and best fusion method
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With | Size |
|------|-------------|----------|------|
| `sequences/test_small.csv` | Small dataset (5 peptides) | All tools | 13KB |
| `sequences/test.csv` | Test dataset (~1000 peptides) | Batch processing | 1.6MB |
| `sequences/train.csv` | Training dataset (5559+ peptides) | Large scale analysis | 12MB |
| `sequences/val.csv` | Validation dataset | Model evaluation | 1.6MB |

### Sample SMILES
```
CC(C)C[C@@H]1NC(=O)[C@H](C)N(C)C(=O)[C@@H]2CCCN2C(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H]2CCCN2C(=O)[C@@H](CC(C)C)N(C)C1=O
```
This represents a 7-residue cyclic peptide with stereochemistry and N-methylation patterns.

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Key Parameters |
|--------|-------------|----------------|
| `draw_peptide_images_config.json` | Image generation settings | image_size: [600,600], image_format: "png" |
| `feature_analysis_config.json` | Feature analysis settings | methods: ["concate", "cross_attention", "attention"] |
| `default_config.json` | Global defaults | common settings for all scripts |

### Example Configuration Usage

```bash
# Use custom configuration
python scripts/draw_peptide_images.py \
  --input examples/data/sequences/test_small.csv \
  --config configs/draw_peptide_images_config.json
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment using package manager detection
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
else
    PKG_MGR="conda"
fi

$PKG_MGR create -p ./env python=3.10 -y
$PKG_MGR activate ./env
pip install -r requirements.txt
$PKG_MGR install -c conda-forge rdkit -y
```

**Problem:** RDKit import errors
```bash
# Install RDKit from conda-forge (preferred method)
$PKG_MGR install -p ./env -c conda-forge rdkit -y

# Verify installation
python -c "from rdkit import Chem; print('RDKit working!')"
```

**Problem:** Import errors
```bash
# Verify all dependencies
python -c "
import pandas, numpy, matplotlib, seaborn, rdkit, fastmcp, loguru
print('All dependencies available')
"
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove cycpep-tools
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify connection
claude mcp list | grep cycpep-tools
```

**Problem:** Invalid SMILES error
```
Ensure your SMILES string represents a cyclic peptide. Use the cyclo() notation
or ensure ring closure numbers are properly specified.

Valid example:
CC(C)C[C@@H]1NC(=O)[C@H](C)N(C)C(=O)[C@@H]2CCCN2C(=O)[C@H](Cc2ccccc2)NC1=O
```

**Problem:** Tools not working
```bash
# Test server directly
python -c "
import sys
sys.path.append('src')
from server import mcp
print('Available tools:', len(list(mcp.list_tools())))
"
```

### Cyclic Peptide Specific Issues

**Problem:** Peptide structure not recognized
- Ensure SMILES contains proper cyclization
- Check for valid amino acid connectivity
- Verify stereochemistry annotations (@@ or @)

**Problem:** Image generation fails
```bash
# Install system dependencies for SVG conversion
sudo apt-get install libcairo2-dev  # Ubuntu/Debian
brew install cairo  # macOS

# Reinstall cairosvg
pip install --force-reinstall cairosvg
```

**Problem:** Feature analysis produces poor results
- Verify peptide dataset has proper molecular descriptors
- Check for missing or invalid SMILES strings
- Ensure adequate dataset size (>10 peptides recommended)

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory and permissions
ls -la jobs/
df -h .  # Check disk space
```

**Problem:** Job failed
```
Use get_job_log with job_id "your_job_id" and tail 100 to see detailed error logs.
Common causes: insufficient memory, corrupted input files, missing dependencies.
```

### Performance Optimization

**Problem:** Slow processing
- Use mamba instead of conda for faster package operations
- For large datasets (>100 peptides), use submit_batch_* tools
- Monitor memory usage with `htop` or similar tools
- Use SSD storage for better I/O performance

---

## Development

### Running Tests

```bash
# Activate environment
$PKG_MGR activate ./env

# Run integration tests
python tests/run_integration_tests.py

# Test individual tools
python test_direct_functions.py
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
fastmcp dev src/server.py

# Or run directly
python src/server.py
```

### Adding New Tools

Follow the patterns in `src/server.py`:
1. Add tool function with `@mcp.tool()` decorator
2. Include comprehensive docstring
3. Add error handling and validation
4. Update `get_server_info()` tool listing

---

## Performance Metrics

| Operation | Small (5 peptides) | Medium (100 peptides) | Large (1000+ peptides) |
|-----------|-------------------|---------------------|----------------------|
| Image Generation | 1.3s | 30s | Use batch API |
| Feature Analysis | 3.4s | 60s | Use batch API |
| Data Validation | <1s | 5s | 30s |

### Resource Requirements

- **Memory**: 1-2GB for typical operations, 4GB+ for large datasets
- **CPU**: Multi-core utilization for batch processing
- **Storage**: ~10GB for job directory (configurable)
- **Network**: Not required (all processing local)

---

## License

This project is based on the MultiCycPermea repository. Please see the original repository for license information.

## Credits

Based on [MultiCycPermea](https://github.com/ACE-KAIST/MultiCycPermea) for cyclic peptide membrane permeability prediction.

## Citations

When using this tool, please cite the original MultiCycPermea work:

[![DOI](https://zenodo.org/badge/911614028.svg)](https://doi.org/10.5281/zenodo.14795687)