"""
Shared I/O utilities for cyclic peptide MCP scripts.

Extracted and simplified from repo code to minimize dependencies.
"""
import json
import csv
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
import pandas as pd

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
    """Save configuration to JSON file."""
    try:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False

def load_peptide_data(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load peptide data from CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception:
        return None

def save_peptide_data(df: pd.DataFrame, file_path: Union[str, Path]) -> bool:
    """Save peptide data to CSV file."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        return True
    except Exception:
        return False

def validate_csv_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    """Validate that DataFrame contains required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns

def create_output_directory(path: Union[str, Path]) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_results(results: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """Save analysis results to JSON file."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        serializable_results = convert_numpy(results)

        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        return True
    except Exception:
        return False

def load_results(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load analysis results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def write_markdown_report(content: List[str], file_path: Union[str, Path]) -> bool:
    """Write markdown report to file."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write('\n'.join(content))
        return True
    except Exception:
        return False

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get file information (size, modification time, etc.)."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {}

        stat = path.stat()
        return {
            'size_bytes': stat.st_size,
            'size_kb': round(stat.st_size / 1024, 2),
            'modified_time': stat.st_mtime,
            'exists': True,
            'is_file': path.is_file(),
            'suffix': path.suffix
        }
    except Exception:
        return {'exists': False}