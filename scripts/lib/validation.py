"""
Input validation utilities for cyclic peptide MCP scripts.

Extracted and simplified from repo code to minimize dependencies.
"""
import re
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import pandas as pd
from rdkit import Chem

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string."""
    if not smiles or not isinstance(smiles, str):
        return False

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False

def validate_file_path(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate file path and return validation results."""
    path = Path(file_path)

    result = {
        'valid': False,
        'exists': path.exists(),
        'is_file': False,
        'readable': False,
        'size_mb': 0,
        'error_message': None
    }

    try:
        if not path.exists():
            result['error_message'] = f"File does not exist: {path}"
            return result

        if not path.is_file():
            result['error_message'] = f"Path is not a file: {path}"
            return result

        result['is_file'] = True

        # Check if readable
        try:
            with open(path, 'r') as f:
                f.read(1)  # Try to read first character
            result['readable'] = True
        except Exception:
            result['error_message'] = f"File is not readable: {path}"
            return result

        # Get file size
        stat = path.stat()
        result['size_mb'] = round(stat.st_size / (1024 * 1024), 2)

        result['valid'] = True

    except Exception as e:
        result['error_message'] = f"Error validating file: {str(e)}"

    return result

def validate_csv_data(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """Validate CSV data for required columns and basic integrity."""
    result = {
        'valid': False,
        'row_count': len(df) if df is not None else 0,
        'column_count': len(df.columns) if df is not None else 0,
        'missing_columns': [],
        'empty_columns': [],
        'invalid_smiles': [],
        'error_message': None
    }

    try:
        if df is None or df.empty:
            result['error_message'] = "DataFrame is None or empty"
            return result

        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        result['missing_columns'] = missing_cols

        if missing_cols:
            result['error_message'] = f"Missing required columns: {missing_cols}"
            return result

        # Check for empty columns
        for col in required_columns:
            if df[col].isna().all():
                result['empty_columns'].append(col)

        # Validate SMILES if present
        if 'SMILES' in df.columns:
            for idx, smiles in df['SMILES'].items():
                if pd.isna(smiles) or not validate_smiles(str(smiles)):
                    result['invalid_smiles'].append({'row': idx, 'smiles': str(smiles)})

        result['valid'] = len(result['missing_columns']) == 0 and len(result['empty_columns']) == 0

        if result['invalid_smiles']:
            result['error_message'] = f"Found {len(result['invalid_smiles'])} invalid SMILES"

    except Exception as e:
        result['error_message'] = f"Error validating CSV data: {str(e)}"

    return result

def validate_config(config: Dict[str, Any], required_keys: List[str] = None) -> Dict[str, Any]:
    """Validate configuration dictionary."""
    result = {
        'valid': True,
        'missing_keys': [],
        'invalid_types': [],
        'error_message': None
    }

    try:
        if not isinstance(config, dict):
            result['valid'] = False
            result['error_message'] = "Config must be a dictionary"
            return result

        if required_keys:
            missing = [key for key in required_keys if key not in config]
            result['missing_keys'] = missing

            if missing:
                result['valid'] = False
                result['error_message'] = f"Missing required config keys: {missing}"

    except Exception as e:
        result['valid'] = False
        result['error_message'] = f"Error validating config: {str(e)}"

    return result

def validate_image_size(size_str: str) -> Optional[tuple]:
    """Validate and parse image size string (e.g., '600,600')."""
    try:
        if not size_str:
            return None

        parts = size_str.strip().split(',')
        if len(parts) != 2:
            return None

        width, height = map(int, parts)

        # Basic sanity checks
        if width <= 0 or height <= 0:
            return None

        if width > 5000 or height > 5000:  # Reasonable upper limit
            return None

        return (width, height)

    except Exception:
        return None

def validate_output_directory(path: Union[str, Path], create_if_missing: bool = True) -> Dict[str, Any]:
    """Validate output directory and optionally create it."""
    path = Path(path)

    result = {
        'valid': False,
        'exists': path.exists(),
        'is_directory': False,
        'writable': False,
        'created': False,
        'error_message': None
    }

    try:
        if path.exists():
            if not path.is_dir():
                result['error_message'] = f"Path exists but is not a directory: {path}"
                return result
            result['is_directory'] = True

        elif create_if_missing:
            path.mkdir(parents=True, exist_ok=True)
            result['created'] = True
            result['exists'] = True
            result['is_directory'] = True

        # Test if writable
        if result['exists'] and result['is_directory']:
            test_file = path / '.write_test'
            try:
                test_file.write_text('test')
                test_file.unlink()
                result['writable'] = True
                result['valid'] = True
            except Exception:
                result['error_message'] = f"Directory is not writable: {path}"

    except Exception as e:
        result['error_message'] = f"Error validating output directory: {str(e)}"

    return result