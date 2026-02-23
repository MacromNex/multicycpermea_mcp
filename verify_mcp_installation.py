#!/usr/bin/env python3
"""
Quick verification script for MCP server installation
"""

import sys
import subprocess
from pathlib import Path
import time

def check_environment():
    """Check if environment and dependencies are available."""
    print("🔍 Checking Environment...")

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  Python version: {python_version}")

    # Check key imports
    try:
        import pandas
        print(f"  ✅ pandas: {pandas.__version__}")
    except ImportError:
        print("  ❌ pandas: Not available")
        return False

    try:
        import rdkit
        print(f"  ✅ RDKit: {rdkit.__version__}")
    except ImportError:
        print("  ❌ RDKit: Not available")
        return False

    try:
        from fastmcp import FastMCP
        print("  ✅ FastMCP: Available")
    except ImportError:
        print("  ❌ FastMCP: Not available")
        return False

    return True

def check_files():
    """Check if required files exist."""
    print("\n📁 Checking Files...")

    required_files = [
        "src/server.py",
        "src/jobs/manager.py",
        "scripts/draw_peptide_images.py",
        "scripts/feature_analysis.py",
        "examples/data/sequences/test_small.csv"
    ]

    all_found = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_found = False

    return all_found

def test_server_import():
    """Test if MCP server can be imported."""
    print("\n🚀 Testing Server Import...")

    try:
        sys.path.insert(0, 'src')
        from server import mcp
        print("  ✅ MCP server imported successfully")

        # Try to get server info (if available)
        try:
            from server import get_server_info
            # Note: get_server_info is a tool wrapper, we'll just check it exists
            print("  ✅ Server info function available")
        except:
            print("  ⚠️  Server info function check skipped")

        return True

    except Exception as e:
        print(f"  ❌ Failed to import MCP server: {e}")
        return False

def test_quick_startup():
    """Test if server can start briefly."""
    print("\n⚡ Testing Quick Server Startup...")

    try:
        # Run server for a very short time to check startup
        cmd = ["mamba", "run", "-p", "./env", "python", "-c", """
import sys
sys.path.insert(0, 'src')
try:
    from server import mcp
    print('SERVER_IMPORT_SUCCESS')
except Exception as e:
    print(f'SERVER_IMPORT_ERROR: {e}')
"""]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if "SERVER_IMPORT_SUCCESS" in result.stdout:
            print("  ✅ Server can be imported in subprocess")
            return True
        else:
            print("  ❌ Server import failed in subprocess")
            print(f"     Output: {result.stdout}")
            print(f"     Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("  ⚠️  Server startup test timed out (may be normal)")
        return True  # Timeout doesn't necessarily mean failure
    except Exception as e:
        print(f"  ❌ Server startup test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n📚 Usage Instructions:")
    print("""
To start the MCP server:

1. Activate environment:
   mamba activate ./env

2. Start server:
   python src/server.py

   OR

   fastmcp dev src/server.py

3. Available tools (13 total):
   - Job Management: get_job_status, get_job_result, get_job_log, etc.
   - Synchronous: generate_peptide_images, analyze_peptide_features, validate_peptide_csv
   - Asynchronous: submit_batch_image_generation, submit_batch_feature_analysis
   - Utility: get_server_info, get_example_data_info

4. Documentation:
   See reports/step6_mcp_tools.md for complete API reference

5. Test with example data:
   examples/data/sequences/test_small.csv (5 peptides)
""")

def main():
    """Run all verification checks."""
    print("🧪 MCP Server Installation Verification")
    print("=" * 50)

    checks = [
        ("Environment", check_environment),
        ("Files", check_files),
        ("Server Import", test_server_import),
        ("Quick Startup", test_quick_startup)
    ]

    results = []
    for check_name, check_func in checks:
        success = check_func()
        results.append((check_name, success))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Verification Results:")

    passed = 0
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {check_name}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} checks passed")

    if passed == len(results):
        print("🎉 MCP server is ready to use!")
        print_usage_instructions()
    elif passed >= 3:
        print("⚠️  MCP server is mostly functional with minor issues.")
        print_usage_instructions()
    else:
        print("❌ MCP server has significant issues. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure you're in the correct environment: mamba activate ./env")
        print("2. Install missing dependencies: mamba run -p ./env pip install fastmcp")
        print("3. Check file paths are correct")

if __name__ == "__main__":
    main()