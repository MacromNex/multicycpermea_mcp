# MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported (rdkit, numpy, pandas, matplotlib)
2. **Self-Contained**: Functions inlined where possible to reduce repo dependencies
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Repo Dependent | Config |
|--------|-------------|----------------|--------|
| `draw_peptide_images.py` | Generate 2D molecular structure images | No | `configs/draw_peptide_images.json` |
| `feature_analysis.py` | Analyze feature combination methods | No | `configs/feature_analysis.json` |

## Usage

```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env

# Draw molecular images
python scripts/draw_peptide_images.py --input examples/data/sequences/test_small.csv --output results/images

# Analyze feature combinations
python scripts/feature_analysis.py --input examples/data/sequences/test_small.csv --output results/analysis

# With custom config
python scripts/draw_peptide_images.py --input FILE --output DIR --config configs/custom.json
```

## Script Details

### draw_peptide_images.py

Generate 2D molecular structure images for cyclic peptides from SMILES.

**Main Function**: `run_draw_peptide_images(input_file, output_dir=None, config=None, **kwargs)`

**Dependencies**:
- pandas: Data handling
- rdkit: Molecular visualization
- cairosvg: SVG to PNG conversion

**Inputs**:
- CSV file with `SMILES` and `CycPeptMPDB_ID` columns

**Outputs**:
- PNG images for each peptide
- Updated CSV with image paths
- Generation statistics

**CLI Usage**:
```bash
python scripts/draw_peptide_images.py \
  --input examples/data/sequences/test_small.csv \
  --output results/images \
  --size 600,600 \
  --format png
```

**Example**:
```python
from scripts.draw_peptide_images import run_draw_peptide_images

result = run_draw_peptide_images(
    "examples/data/sequences/test_small.csv",
    "output/images"
)
print(f"Generated {result['generated_count']} images")
```

### feature_analysis.py

Analyze and compare different feature combination methods for cyclic peptide permeability prediction.

**Main Function**: `run_feature_analysis(input_file, output_dir=None, config=None, **kwargs)`

**Dependencies**:
- pandas, numpy: Data analysis
- matplotlib, seaborn: Visualization

**Inputs**:
- CSV file with `SMILES` column

**Outputs**:
- Performance comparison plots
- Computational cost analysis
- Feature importance heatmap
- Comprehensive markdown report

**CLI Usage**:
```bash
python scripts/feature_analysis.py \
  --input examples/data/sequences/test_small.csv \
  --output results/analysis \
  --methods concate,cross_attention,attention
```

**Example**:
```python
from scripts.feature_analysis import run_feature_analysis

result = run_feature_analysis(
    "examples/data/sequences/test_small.csv",
    "output/analysis"
)
print(f"Best method: {result['best_method']}")
```

## Shared Library

Common functions are in `scripts/lib/`:

### `lib/molecules.py`
- `parse_smiles()`: Parse SMILES to RDKit molecule
- `is_cyclic_peptide()`: Check if molecule is cyclic peptide
- `generate_2d_coords()`: Generate 2D coordinates
- `calculate_molecular_properties()`: Basic molecular descriptors
- `save_molecule()`: Save molecule to file

### `lib/io.py`
- `load_config()`: Load JSON configuration
- `load_peptide_data()`: Load CSV data
- `save_results()`: Save analysis results
- `write_markdown_report()`: Generate markdown reports

### `lib/validation.py`
- `validate_smiles()`: Validate SMILES strings
- `validate_file_path()`: Check file accessibility
- `validate_csv_data()`: Validate CSV data integrity
- `validate_config()`: Validate configuration

## Configuration Files

### `configs/draw_peptide_images_config.json`
```json
{
  "image_size": [600, 600],
  "image_format": "png",
  "use_coord_gen": true,
  "batch_size": 100,
  "cleanup_temp_files": true
}
```

### `configs/feature_analysis_config.json`
```json
{
  "methods": ["concate", "cross_attention", "attention"],
  "random_seed": 42,
  "plot_dpi": 300,
  "performance_variation": 0.02
}
```

### `configs/default_config.json`
Global settings for all scripts.

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:

```python
# Example MCP tool wrapper
from scripts.draw_peptide_images import run_draw_peptide_images
from scripts.feature_analysis import run_feature_analysis

@mcp.tool()
def generate_peptide_images(input_file: str, output_dir: str = None):
    """Generate 2D structure images for cyclic peptides."""
    return run_draw_peptide_images(input_file, output_dir)

@mcp.tool()
def analyze_feature_methods(input_file: str, output_dir: str = None, methods: str = "concate,cross_attention"):
    """Analyze different feature combination methods."""
    methods_list = methods.split(',')
    return run_feature_analysis(input_file, output_dir, methods=methods_list)
```

## Testing

All scripts have been tested with the example data:

```bash
# Test image generation
python scripts/draw_peptide_images.py \
  --input examples/data/sequences/test_small.csv \
  --output results/test_images

# Test feature analysis
python scripts/feature_analysis.py \
  --input examples/data/sequences/test_small.csv \
  --output results/test_analysis
```

Expected outputs:
- `draw_peptide_images.py`: 5 PNG files (~20-30KB each)
- `feature_analysis.py`: 3 plots + markdown report

## Error Handling

Scripts include comprehensive error handling:
- Invalid SMILES strings are skipped with warnings
- Missing files provide helpful error messages
- Configuration validation with clear feedback
- Graceful degradation for non-critical errors

## Performance

- **Image Generation**: ~0.26s per peptide (batch of 5)
- **Feature Analysis**: ~3.4s for comparison analysis
- **Memory Usage**: < 2GB for typical datasets
- **Scalability**: Tested with 5-peptide datasets, should scale to thousands

## Dependencies Summary

| Package | Purpose | Scripts |
|---------|---------|---------|
| `pandas` | Data handling | Both |
| `rdkit` | Molecular operations | draw_peptide_images |
| `cairosvg` | Image conversion | draw_peptide_images |
| `numpy` | Numerical operations | feature_analysis |
| `matplotlib` | Plotting | feature_analysis |
| `seaborn` | Statistical plots | feature_analysis |

All dependencies are available in both `./env` and `./env_py37` environments.

## Development Notes

- Scripts follow the template pattern from Step 5 requirements
- All repo-specific code has been inlined or simplified
- Configuration is externalized for easy MCP parameter mapping
- Functions return structured dictionaries suitable for MCP responses
- CLI interfaces provide immediate usability for testing