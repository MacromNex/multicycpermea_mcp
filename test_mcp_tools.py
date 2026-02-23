#!/usr/bin/env python3
"""
Test script for MCP tools functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from server import (
    generate_peptide_images,
    analyze_peptide_features,
    validate_peptide_csv,
    submit_batch_image_generation,
    submit_batch_feature_analysis,
    get_job_status,
    get_job_result,
    list_jobs,
    get_server_info,
    get_example_data_info
)

def test_server_info():
    """Test server information."""
    print("=== Testing Server Info ===")
    try:
        # Call the tool function directly
        result = get_server_info()
        print(f"Status: {result['status']}")
        print(f"Server: {result['server_name']}")
        print(f"Version: {result['version']}")
        print(f"Total tools: {result['total_tools']}")

        for category, tools in result['tools'].items():
            print(f"  {category.title()}: {len(tools)} tools")

        print(f"Script availability:")
        for script, available in result['script_availability'].items():
            print(f"  {script}: {'✅' if available else '❌'}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_example_data():
    """Test example data info."""
    print("\n=== Testing Example Data Info ===")
    try:
        result = get_example_data_info()
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Found {result['total_datasets']} datasets")
            for dataset in result['datasets']:
                if 'error' not in dataset:
                    print(f"  📄 {dataset['name']}: {dataset['rows']} rows, {len(dataset['columns'])} cols")
                    print(f"     Columns: {', '.join(dataset['columns'])}")
                else:
                    print(f"  ❌ {dataset['name']}: {dataset['error']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_validate_csv():
    """Test CSV validation."""
    print("\n=== Testing CSV Validation ===")

    # Find a test CSV file
    test_files = [
        "examples/data/sequences/test_small.csv",
        "examples/data/test_small.csv"
    ]

    test_file = None
    for file_path in test_files:
        if Path(file_path).exists():
            test_file = file_path
            break

    if not test_file:
        print("❌ No test CSV file found")
        return False

    try:
        result = validate_peptide_csv(test_file)
        print(f"Status: {result['status']}")
        print(f"File: {result['file_path']}")
        print(f"Rows: {result['total_rows']}")
        print(f"Columns: {result['columns']}")

        if 'smiles_validation' in result:
            smiles_val = result['smiles_validation']
            if 'error' not in smiles_val:
                print(f"SMILES validation: {smiles_val['valid_count']} valid, {smiles_val['invalid_count']} invalid")
            else:
                print(f"SMILES validation: {smiles_val['error']}")

        return result['status'] in ['success', 'warning']
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_sync_image_generation():
    """Test synchronous image generation."""
    print("\n=== Testing Sync Image Generation ===")

    test_files = [
        "examples/data/sequences/test_small.csv",
        "examples/data/test_small.csv"
    ]

    test_file = None
    for file_path in test_files:
        if Path(file_path).exists():
            test_file = file_path
            break

    if not test_file:
        print("❌ No test CSV file found")
        return False

    try:
        output_dir = "results/test_mcp_images"
        result = generate_peptide_images(
            input_file=test_file,
            output_dir=output_dir,
            image_size="400,400"
        )

        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Generated: {result.get('generated_count', 0)} images")
            print(f"Failed: {result.get('failed_count', 0)} images")
            print(f"Output: {result.get('output_dir', 'N/A')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

        return result['status'] == 'success'
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_sync_feature_analysis():
    """Test synchronous feature analysis."""
    print("\n=== Testing Sync Feature Analysis ===")

    test_files = [
        "examples/data/sequences/test_small.csv",
        "examples/data/test_small.csv"
    ]

    test_file = None
    for file_path in test_files:
        if Path(file_path).exists():
            test_file = file_path
            break

    if not test_file:
        print("❌ No test CSV file found")
        return False

    try:
        output_dir = "results/test_mcp_analysis"
        result = analyze_peptide_features(
            input_file=test_file,
            output_dir=output_dir,
            methods="concate,cross_attention"
        )

        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Best method: {result.get('best_method', 'N/A')}")
            print(f"Output: {result.get('output_dir', 'N/A')}")
            if 'plots_created' in result:
                print(f"Plots: {len(result['plots_created'])} created")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

        return result['status'] == 'success'
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_async_submission():
    """Test async job submission."""
    print("\n=== Testing Async Job Submission ===")

    test_files = [
        "examples/data/sequences/test_small.csv",
        "examples/data/test_small.csv"
    ]

    test_file = None
    for file_path in test_files:
        if Path(file_path).exists():
            test_file = file_path
            break

    if not test_file:
        print("❌ No test CSV file found")
        return False

    try:
        # Submit image generation job
        result = submit_batch_image_generation(
            input_file=test_file,
            output_dir="results/test_mcp_async_images",
            job_name="test_image_job"
        )

        print(f"Image job status: {result['status']}")
        if result['status'] == 'submitted':
            job_id = result['job_id']
            print(f"Job ID: {job_id}")

            # Check status
            import time
            time.sleep(1)  # Give it a moment
            status = get_job_status(job_id)
            print(f"Job status: {status['status']}")

            return True
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_job_listing():
    """Test job listing functionality."""
    print("\n=== Testing Job Listing ===")

    try:
        result = list_jobs()
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Total jobs: {result['total']}")
            for job in result['jobs'][:3]:  # Show first 3 jobs
                print(f"  📋 {job['job_id']}: {job['status']} - {job['job_name']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing MCP Tools Functionality\n")

    tests = [
        ("Server Info", test_server_info),
        ("Example Data", test_example_data),
        ("CSV Validation", test_validate_csv),
        ("Sync Image Generation", test_sync_image_generation),
        ("Sync Feature Analysis", test_sync_feature_analysis),
        ("Async Job Submission", test_async_submission),
        ("Job Listing", test_job_listing)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        success = test_func()
        results.append((test_name, success))

    print(f"\n{'='*60}")
    print("📊 Test Results Summary:")
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! MCP server is ready for use.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()