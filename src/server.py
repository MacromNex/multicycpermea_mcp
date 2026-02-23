"""MCP Server for Cyclic Peptide Tools

Provides both synchronous and asynchronous (submit) APIs for cyclic peptide computational tools.
Supports image generation, feature analysis, and batch processing.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import sys
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"

# Add paths to Python path
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

# Import job manager
from jobs.manager import job_manager

# Import script functions
try:
    from draw_peptide_images import run_draw_peptide_images
    from feature_analysis import run_feature_analysis
except ImportError as e:
    print(f"Warning: Could not import script functions: {e}")
    run_draw_peptide_images = None
    run_feature_analysis = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

# Create MCP server
mcp = FastMCP("cycpep-tools")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted cyclic peptide computation job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, progress, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)


@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed cyclic peptide computation job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed.
        Includes output_directory and list of generated files.
    """
    return job_manager.get_job_result(job_id)


@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running cyclic peptide computation job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)


@mcp.tool()
def list_jobs(status: Optional[str] = None, limit: int = 50) -> dict:
    """
    List all submitted cyclic peptide computation jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)
        limit: Maximum number of jobs to return (default: 50)

    Returns:
        List of jobs with their status, sorted by submission time (newest first)
    """
    return job_manager.list_jobs(status, limit)


@mcp.tool()
def cleanup_old_jobs(older_than_days: int = 7) -> dict:
    """
    Clean up old completed/failed jobs to free disk space.

    Args:
        older_than_days: Remove jobs completed more than this many days ago

    Returns:
        Count of cleaned jobs and any errors
    """
    return job_manager.cleanup_old_jobs(older_than_days)


# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def generate_peptide_images(
    input_file: str,
    output_dir: Optional[str] = None,
    image_size: str = "600,600",
    image_format: str = "png",
    use_coord_gen: bool = True
) -> dict:
    """
    Generate 2D molecular structure images for cyclic peptides from SMILES.

    Fast operation - returns results immediately (typically 1-3 seconds).

    Args:
        input_file: CSV file with SMILES and CycPeptMPDB_ID columns
        output_dir: Output directory for images (optional)
        image_size: Image dimensions as "width,height" (default: "600,600")
        image_format: Output format (png, svg) (default: "png")
        use_coord_gen: Use RDKit coordinate generation (default: True)

    Returns:
        Dictionary with generation results and output paths
    """
    if run_draw_peptide_images is None:
        return {"status": "error", "error": "draw_peptide_images script not available"}

    try:
        # Parse image size
        width, height = map(int, image_size.split(','))
        size_tuple = (width, height)

        # Run the function
        result = run_draw_peptide_images(
            input_file=input_file,
            output_dir=output_dir,
            image_size=size_tuple,
            image_format=image_format,
            use_coord_gen=use_coord_gen
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def analyze_peptide_features(
    input_file: str,
    output_dir: Optional[str] = None,
    methods: str = "concate,cross_attention",
    random_seed: int = 42,
    plot_dpi: int = 300
) -> dict:
    """
    Analyze and compare feature combination methods for cyclic peptides.

    Fast operation - returns results immediately (typically 2-5 seconds).

    Args:
        input_file: CSV file with SMILES column
        output_dir: Output directory for analysis (optional)
        methods: Comma-separated list of methods (concate,cross_attention,attention)
        random_seed: Random seed for reproducibility (default: 42)
        plot_dpi: Plot resolution (default: 300)

    Returns:
        Dictionary with analysis results, best method, and generated files
    """
    if run_feature_analysis is None:
        return {"status": "error", "error": "feature_analysis script not available"}

    try:
        # Parse methods list
        methods_list = [method.strip() for method in methods.split(',')]

        # Run the function
        result = run_feature_analysis(
            input_file=input_file,
            output_dir=output_dir,
            methods=methods_list,
            random_seed=random_seed,
            plot_dpi=plot_dpi
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Feature analysis failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def validate_peptide_csv(
    input_file: str,
    required_columns: str = "SMILES,CycPeptMPDB_ID"
) -> dict:
    """
    Validate a CSV file for cyclic peptide analysis.

    Checks file format, required columns, and SMILES validity.

    Args:
        input_file: CSV file to validate
        required_columns: Comma-separated list of required columns

    Returns:
        Dictionary with validation results and file statistics
    """
    try:
        from pathlib import Path
        import pandas as pd

        file_path = Path(input_file)
        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {input_file}"}

        # Load CSV
        df = pd.read_csv(file_path)

        # Check required columns
        required_cols = [col.strip() for col in required_columns.split(',')]
        missing_cols = [col for col in required_cols if col not in df.columns]

        result = {
            "status": "success",
            "file_path": str(file_path),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "required_columns": required_cols,
            "missing_columns": missing_cols,
            "file_size_bytes": file_path.stat().st_size
        }

        # Validate SMILES if present
        if "SMILES" in df.columns:
            try:
                from rdkit import Chem
                valid_smiles = 0
                invalid_smiles = []

                for idx, smiles in enumerate(df["SMILES"]):
                    if pd.notna(smiles):
                        mol = Chem.MolFromSmiles(str(smiles))
                        if mol is not None:
                            valid_smiles += 1
                        else:
                            invalid_smiles.append({"row": idx, "smiles": str(smiles)})

                result["smiles_validation"] = {
                    "valid_count": valid_smiles,
                    "invalid_count": len(invalid_smiles),
                    "invalid_examples": invalid_smiles[:5]  # First 5 invalid
                }
            except ImportError:
                result["smiles_validation"] = {"error": "RDKit not available for SMILES validation"}

        if missing_cols:
            result["status"] = "warning"
            result["warning"] = f"Missing required columns: {missing_cols}"

        return result

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==============================================================================
# Submit Tools (for long-running operations > 10 min or batch processing)
# ==============================================================================

@mcp.tool()
def submit_batch_image_generation(
    input_file: str,
    output_dir: Optional[str] = None,
    image_size: str = "600,600",
    image_format: str = "png",
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch image generation for multiple cyclic peptides.

    Use this for large datasets (>100 peptides) or when you want to run
    image generation in the background.

    Args:
        input_file: CSV file with SMILES and CycPeptMPDB_ID columns
        output_dir: Output directory for images
        image_size: Image dimensions as "width,height" (default: "600,600")
        image_format: Output format (png, svg) (default: "png")
        job_name: Optional name for tracking

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "draw_peptide_images.py")

    # Parse image size
    try:
        width, height = map(int, image_size.split(','))
        size_str = f"{width},{height}"
    except ValueError:
        return {"status": "error", "error": f"Invalid image size format: {image_size}"}

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "size": size_str,
            "format": image_format
        },
        job_name=job_name or f"batch_images_{Path(input_file).stem}"
    )


@mcp.tool()
def submit_batch_feature_analysis(
    input_file: str,
    output_dir: Optional[str] = None,
    methods: str = "concate,cross_attention,attention",
    random_seed: int = 42,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch feature analysis for multiple cyclic peptides.

    Use this for comprehensive analysis of large datasets or when
    you want to run analysis in the background.

    Args:
        input_file: CSV file with SMILES column
        output_dir: Output directory for analysis
        methods: Comma-separated list of methods to analyze
        random_seed: Random seed for reproducibility
        job_name: Optional name for tracking

    Returns:
        Dictionary with job_id for tracking
    """
    script_path = str(SCRIPTS_DIR / "feature_analysis.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "methods": methods.split(','),  # Pass as list
            "seed": random_seed
        },
        job_name=job_name or f"batch_analysis_{Path(input_file).stem}"
    )


# ==============================================================================
# Configuration and Utility Tools
# ==============================================================================

@mcp.tool()
def get_server_info() -> dict:
    """
    Get information about the MCP server and available tools.

    Returns:
        Server version, available tools, and system information
    """
    tools_info = {
        "job_management": [
            "get_job_status",
            "get_job_result",
            "get_job_log",
            "cancel_job",
            "list_jobs",
            "cleanup_old_jobs"
        ],
        "synchronous": [
            "generate_peptide_images",
            "analyze_peptide_features",
            "validate_peptide_csv"
        ],
        "asynchronous": [
            "submit_batch_image_generation",
            "submit_batch_feature_analysis"
        ]
    }

    return {
        "status": "success",
        "server_name": "cycpep-tools",
        "version": "1.0.0",
        "tools": tools_info,
        "total_tools": sum(len(tools) for tools in tools_info.values()),
        "scripts_directory": str(SCRIPTS_DIR),
        "jobs_directory": str(job_manager.jobs_dir),
        "script_availability": {
            "draw_peptide_images": run_draw_peptide_images is not None,
            "feature_analysis": run_feature_analysis is not None
        }
    }


@mcp.tool()
def load_config_template(tool_name: str) -> dict:
    """
    Load configuration template for a specific tool.

    Args:
        tool_name: Name of tool (draw_peptide_images, feature_analysis)

    Returns:
        Dictionary with default configuration parameters
    """
    config_file = MCP_ROOT / "configs" / f"{tool_name}_config.json"

    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            return {
                "status": "success",
                "tool_name": tool_name,
                "config": config,
                "config_file": str(config_file)
            }
        except Exception as e:
            return {"status": "error", "error": f"Failed to load config: {e}"}
    else:
        return {"status": "error", "error": f"Config file not found: {config_file}"}


@mcp.tool()
def get_example_data_info() -> dict:
    """
    Get information about available example datasets.

    Returns:
        List of example CSV files with their statistics
    """
    try:
        examples_dir = MCP_ROOT / "examples" / "data"
        result = {
            "status": "success",
            "examples_directory": str(examples_dir),
            "datasets": []
        }

        if examples_dir.exists():
            import pandas as pd

            for csv_file in examples_dir.rglob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    dataset_info = {
                        "name": csv_file.name,
                        "path": str(csv_file),
                        "relative_path": str(csv_file.relative_to(MCP_ROOT)),
                        "rows": len(df),
                        "columns": list(df.columns),
                        "size_bytes": csv_file.stat().st_size
                    }
                    result["datasets"].append(dataset_info)
                except Exception as e:
                    result["datasets"].append({
                        "name": csv_file.name,
                        "path": str(csv_file),
                        "error": f"Could not read: {e}"
                    })

        result["total_datasets"] = len(result["datasets"])
        return result

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    """Run the MCP server."""
    logger.info("Starting CycPep MCP Server...")
    logger.info(f"Scripts directory: {SCRIPTS_DIR}")
    logger.info(f"Jobs directory: {job_manager.jobs_dir}")

    # Check script availability
    if run_draw_peptide_images is None:
        logger.warning("draw_peptide_images.py not available")
    if run_feature_analysis is None:
        logger.warning("feature_analysis.py not available")

    mcp.run()


if __name__ == "__main__":
    main()