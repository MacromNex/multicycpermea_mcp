# Step 3: Use Cases Report

## Scan Information
- **Scan Date**: 2026-01-01
- **Filter Applied**: cyclic peptide membrane permeability prediction using MultiCycPermea, 2D image and 1D sequence features
- **Python Version Strategy**: Dual (3.10 main + 3.7 legacy)
- **Repository**: MultiCycPermea for cyclic peptide permeability prediction

## Use Cases

### UC-001: Draw Peptide 2D Structure Images
- **Description**: Generate 2D molecular structure images from SMILES strings of cyclic peptides for visual analysis and image feature extraction
- **Script Path**: `examples/use_case_1_draw_peptide_images.py`
- **Complexity**: Simple
- **Priority**: High
- **Environment**: `./env` (Python 3.10 with RDKit 2025.09.4)
- **Source**: `repo/MultiCycPermea/data/draw_peptide_images.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| csv_file | file | CSV with SMILES and CycPeptMPDB_ID columns | --input, -i |
| output_folder | path | Directory to save generated PNG images | --output, -o |
| image_size | string | Image dimensions as "width,height" | --size, -s |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| png_images | files | 2D molecular structure images (600x600 PNG) |
| updated_csv | file | Original CSV with added image_path column |

**Example Usage:**
```bash
python examples/use_case_1_draw_peptide_images.py \
  --input examples/data/sequences/test.csv \
  --output examples/data/images/ \
  --size 600,600
```

**Example Data**: `examples/data/sequences/test.csv` (sample cyclic peptide SMILES)

---

### UC-002: Train MultiCycPermea Model
- **Description**: Train deep learning model combining 1D SMILES sequences and 2D molecular images for membrane permeability prediction of cyclic peptides
- **Script Path**: `examples/use_case_2_train_multicycpermea.py`
- **Complexity**: Complex
- **Priority**: High
- **Environment**: `./env_py37` (Python 3.7 with PyTorch + CUDA)
- **Source**: `repo/MultiCycPermea/DL/main.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| config_file | file | Main YAML configuration file | --all_config |
| use_text_info | bool | Enable 1D SMILES features | --use_text_info |
| use_image_info | bool | Enable 2D image features | --use_image_info |
| feature_cmb_type | string | Feature fusion method | --feature_cmb_type |
| gpu_device | int | CUDA GPU device number | --gpu |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| trained_model | file | PyTorch model weights (.pt file) |
| training_logs | file | TensorBoard logs and metrics |
| config_logs | file | Training configuration record |

**Example Usage:**
```bash
mamba activate ./env_py37
CUDA_VISIBLE_DEVICES=0 python examples/use_case_2_train_multicycpermea.py \
  --use_text_info True \
  --use_image_info True \
  --feature_cmb_type concate
```

**Example Data**: Training requires `examples/data/sequences/train.csv` and corresponding molecular images

---

### UC-003: Predict Membrane Permeability
- **Description**: Use trained MultiCycPermea model to predict membrane permeability of new cyclic peptides from SMILES and/or images
- **Script Path**: `examples/use_case_3_predict_permeability.py`
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env_py37` (Python 3.7 with trained model)
- **Source**: `repo/MultiCycPermea/DL/test.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| model_path | file | Trained PyTorch model (.pt file) | --model_path, -m |
| config_file | file | Model configuration YAML | --config, -c |
| input_data | file | CSV with SMILES to predict | --input, -i |
| single_smiles | string | Single SMILES string for prediction | --smiles, -s |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| predictions | file | CSV with predicted permeability values |
| metrics | text | Performance metrics (if true values available) |

**Example Usage:**
```bash
mamba activate ./env_py37
python examples/use_case_3_predict_permeability.py \
  --model_path models/best_model.pt \
  --input examples/data/sequences/test.csv \
  --config examples/config/model.yaml
```

**Example Data**: Requires trained model and test peptides in `examples/data/sequences/test.csv`

---

### UC-004: Analyze Feature Combination Methods
- **Description**: Compare different fusion strategies for combining 1D SMILES and 2D image features in MultiCycPermea
- **Script Path**: `examples/use_case_4_feature_analysis.py`
- **Complexity**: Medium
- **Priority**: Medium
- **Environment**: `./env_py37` (Python 3.7 for analysis)
- **Source**: Analysis of `repo/MultiCycPermea/DL/model.py` feature fusion methods

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| data_file | file | CSV with peptide data for analysis | --data, -d |
| fusion_methods | string | Comma-separated fusion methods | --methods, -m |
| output_dir | path | Directory for plots and reports | --output_dir, -o |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| performance_plots | files | Comparison plots for different methods |
| feature_importance | file | Heatmap of feature importance |
| analysis_report | file | Markdown report with recommendations |

**Example Usage:**
```bash
mamba activate ./env_py37
python examples/use_case_4_feature_analysis.py \
  --data examples/data/sequences/test.csv \
  --methods concate,cross_attention,attention \
  --output_dir examples/analysis_results
```

**Example Data**: Uses `examples/data/sequences/test.csv` for analysis

---

## Summary

| Metric | Count |
|--------|-------|
| Total Found | 4 |
| Scripts Created | 4 |
| High Priority | 3 |
| Medium Priority | 1 |
| Low Priority | 0 |
| Demo Data Copied | Yes |
| Environments Required | 2 (main + legacy) |

## Demo Data Index

| Source | Destination | Description |
|--------|-------------|-------------|
| `repo/MultiCycPermea/data/remove_strange_values/train.csv` | `examples/data/sequences/train.csv` | Training dataset (5559+ cyclic peptides with permeability) |
| `repo/MultiCycPermea/data/remove_strange_values/test.csv` | `examples/data/sequences/test.csv` | Test dataset for evaluation |
| `repo/MultiCycPermea/data/remove_strange_values/val.csv` | `examples/data/sequences/val.csv` | Validation dataset |

## Feature Compatibility Matrix

| Use Case | 1D SMILES | 2D Images | Environment | CUDA Required |
|----------|-----------|-----------|-------------|---------------|
| UC-001: Draw Images | ✓ | ✓ (generates) | ./env | No |
| UC-002: Training | ✓ | ✓ | ./env_py37 | Yes (recommended) |
| UC-003: Prediction | ✓ | ✓ | ./env_py37 | No (CPU ok) |
| UC-004: Analysis | ✓ | ✓ | ./env_py37 | No |

## Data Requirements

### SMILES Format
- **Input**: Canonical SMILES strings representing cyclic peptides
- **Example**: `CC(C)C[C@@H]1NC(=O)[C@H](C)N(C)C(=O)[C@@H]2CCCN2C(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H]2CCCN2C(=O)[C@@H](CC(C)C)N(C)C1=O`
- **Features**: Stereochemistry, cyclization, N-methylation patterns

### Image Requirements
- **Format**: PNG images (600x600 pixels recommended)
- **Content**: 2D molecular structure diagrams
- **Naming**: Matching CycPeptMPDB_ID from CSV data
- **Generation**: Automated via RDKit (Use Case 1)

### Training Data Structure
- **Columns**: CycPeptMPDB_ID, SMILES, Permeability (target), 100+ molecular descriptors
- **Size**: 5559+ training examples, ~1000 test examples
- **Target**: Log permeability values (continuous regression)
- **Features**: Both sequence-based and image-based features

## Performance Expectations

### Use Case 1: Image Generation
- **Speed**: ~1-2 seconds per peptide image
- **Memory**: Low (< 1GB)
- **Quality**: High-resolution 2D molecular diagrams

### Use Case 2: Model Training
- **Time**: 2-4 hours with GPU (depends on data size)
- **Memory**: 4-8GB GPU memory required
- **Performance**: R² > 0.75 on test set (literature benchmark)

### Use Case 3: Prediction
- **Speed**: ~0.05-0.1 seconds per peptide
- **Batch Size**: 100-1000 peptides efficiently
- **Accuracy**: Consistent with training performance

### Use Case 4: Feature Analysis
- **Time**: 10-30 minutes for comprehensive analysis
- **Output**: Detailed comparison of fusion methods
- **Insights**: Feature importance and computational trade-offs

## Dependencies and Requirements

### System Requirements
- **OS**: Linux (tested), macOS/Windows (should work)
- **Memory**: 8GB+ RAM recommended
- **GPU**: CUDA-compatible for training (optional for prediction)
- **Storage**: 2-5GB for environments and data

### Environment Dependencies
- **Main Environment**: RDKit 2025.09.4, FastMCP 2.14.2
- **Legacy Environment**: PyTorch 1.11.0, RDKit 2020.9.1.0, Transformers
- **CUDA**: Version 11.3 toolkit in legacy environment

## Limitations and Notes

1. **Model Training**: Requires substantial computational resources and pre-trained molecular transformers
2. **Data Dependency**: Performance depends on quality and size of training dataset
3. **Environment Complexity**: Dual environment setup required for compatibility
4. **CUDA Version**: Legacy environment locked to CUDA 11.3 for PyTorch compatibility

## Future Enhancements

1. **MCP Server Implementation**: Convert use cases to MCP server endpoints
2. **Model Optimization**: Implement model compression and quantization
3. **Batch Processing**: Enhanced batch prediction capabilities
4. **Web Interface**: User-friendly interface for non-technical users
5. **API Integration**: REST API for programmatic access