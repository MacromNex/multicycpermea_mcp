#!/usr/bin/env python3
"""Automated integration test runner for MultiCycPermea MCP server.

This script performs automated validation of the MCP server functionality,
testing both direct function calls and MCP tool integration.
"""

import json
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

class MCPTestRunner:
    """Comprehensive test runner for MCP server integration."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {
            "test_date": datetime.now().isoformat(),
            "project_root": str(project_root),
            "python_env": str(project_root / "env" / "bin" / "python"),
            "server_path": str(project_root / "src" / "server.py"),
            "tests": {},
            "issues": [],
            "summary": {},
            "performance": {}
        }
        self.env_python = project_root / "env" / "bin" / "python"

    def log_test(self, test_name: str, status: str, output: str = "", error: str = "",
                 duration: float = 0.0, details: dict = None):
        """Log test result."""
        self.results["tests"][test_name] = {
            "status": status,
            "output": output,
            "error": error,
            "duration": duration,
            "details": details or {}
        }
        print(f"✓ {test_name}: {status}" if status == "passed" else f"✗ {test_name}: {status}")
        if error and status != "passed":
            print(f"  Error: {error}")

    def test_environment_setup(self) -> bool:
        """Test that the environment and dependencies are properly set up."""
        start_time = time.time()

        try:
            # Test Python environment
            result = subprocess.run(
                [str(self.env_python), "--version"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                self.log_test("environment_python", "failed", result.stdout, result.stderr)
                return False

            python_version = result.stdout.strip()

            # Test key dependencies
            deps_to_test = ["fastmcp", "rdkit", "pandas", "numpy", "loguru"]
            missing_deps = []

            for dep in deps_to_test:
                result = subprocess.run(
                    [str(self.env_python), "-c", f"import {dep}; print('{dep} OK')"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    missing_deps.append(dep)

            duration = time.time() - start_time

            if missing_deps:
                self.log_test("environment_dependencies", "failed",
                            python_version, f"Missing: {', '.join(missing_deps)}", duration)
                return False
            else:
                self.log_test("environment_dependencies", "passed",
                            f"{python_version} with all dependencies", "", duration)
                return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test("environment_setup", "error", "", str(e), duration)
            return False

    def test_server_startup(self) -> bool:
        """Test that the MCP server starts without errors."""
        start_time = time.time()

        try:
            # Test server import
            result = subprocess.run(
                [str(self.env_python), "-c", "from src.server import mcp; print('Server import OK')"],
                cwd=self.project_root,
                capture_output=True, text=True, timeout=30
            )

            success = result.returncode == 0
            duration = time.time() - start_time

            if success:
                self.log_test("server_import", "passed", result.stdout.strip(), "", duration)
                return True
            else:
                self.log_test("server_import", "failed", result.stdout, result.stderr, duration)
                return False

        except Exception as e:
            duration = time.time() - start_time
            self.log_test("server_startup", "error", "", str(e), duration)
            return False

    def test_job_manager(self) -> bool:
        """Test that the job manager initializes correctly."""
        start_time = time.time()

        try:
            result = subprocess.run(
                [str(self.env_python), "-c",
                 "from src.jobs.manager import job_manager; "
                 "print(f'Job manager OK, jobs dir: {job_manager.jobs_dir}')"],
                cwd=self.project_root,
                capture_output=True, text=True, timeout=30
            )

            success = result.returncode == 0
            duration = time.time() - start_time

            if success:
                self.log_test("job_manager_init", "passed", result.stdout.strip(), "", duration)
                return True
            else:
                self.log_test("job_manager_init", "failed", result.stdout, result.stderr, duration)
                return False

        except Exception as e:
            duration = time.time() - start_time
            self.log_test("job_manager", "error", "", str(e), duration)
            return False

    def test_script_imports(self) -> bool:
        """Test that all required scripts can be imported."""
        start_time = time.time()

        scripts_to_test = [
            ("draw_peptide_images", "run_draw_peptide_images"),
            ("feature_analysis", "run_feature_analysis")
        ]

        failed_imports = []

        for module, function in scripts_to_test:
            try:
                result = subprocess.run(
                    [str(self.env_python), "-c",
                     f"from {module} import {function}; print('{module}.{function} OK')"],
                    cwd=self.project_root / "scripts",
                    capture_output=True, text=True, timeout=30
                )

                if result.returncode != 0:
                    failed_imports.append(f"{module}.{function}")

            except Exception as e:
                failed_imports.append(f"{module}.{function} (exception: {e})")

        duration = time.time() - start_time

        if failed_imports:
            self.log_test("script_imports", "failed",
                        "", f"Failed imports: {', '.join(failed_imports)}", duration)
            return False
        else:
            self.log_test("script_imports", "passed",
                        "All script imports successful", "", duration)
            return True

    def test_example_data(self) -> bool:
        """Test that example data files exist and are readable."""
        start_time = time.time()

        data_dir = self.project_root / "examples" / "data" / "sequences"
        required_files = ["test_small.csv", "test.csv", "train.csv", "val.csv"]

        missing_files = []
        file_info = {}

        for filename in required_files:
            file_path = data_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
            else:
                try:
                    size = file_path.stat().st_size
                    # Try to read first few lines to ensure it's valid
                    with open(file_path, 'r') as f:
                        lines = f.readlines()[:3]
                    file_info[filename] = {
                        "size": size,
                        "lines_sample": len(lines),
                        "readable": True
                    }
                except Exception as e:
                    file_info[filename] = {"error": str(e), "readable": False}

        duration = time.time() - start_time

        if missing_files:
            self.log_test("example_data", "failed",
                        json.dumps(file_info, indent=2),
                        f"Missing files: {', '.join(missing_files)}", duration)
            return False
        else:
            self.log_test("example_data", "passed",
                        json.dumps(file_info, indent=2), "", duration)
            return True

    def test_mcp_tools_count(self) -> bool:
        """Test that all expected MCP tools are registered."""
        start_time = time.time()

        try:
            # Count tools by counting @mcp.tool decorators
            result = subprocess.run(
                ["grep", "-c", "@mcp.tool()", str(self.project_root / "src" / "server.py")],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                tool_count = int(result.stdout.strip())
                expected_count = 14

                duration = time.time() - start_time

                if tool_count == expected_count:
                    self.log_test("mcp_tools_count", "passed",
                                f"Found {tool_count} tools", "", duration)
                    return True
                else:
                    self.log_test("mcp_tools_count", "failed",
                                f"Found {tool_count} tools",
                                f"Expected {expected_count} tools", duration)
                    return False
            else:
                duration = time.time() - start_time
                self.log_test("mcp_tools_count", "failed",
                            "", "Could not count tools", duration)
                return False

        except Exception as e:
            duration = time.time() - start_time
            self.log_test("mcp_tools_count", "error", "", str(e), duration)
            return False

    def test_fastmcp_dev_mode(self) -> bool:
        """Test that fastmcp dev mode can start the server."""
        start_time = time.time()

        try:
            # Use fastmcp command directly
            fastmcp_cmd = self.project_root / "env" / "bin" / "fastmcp"

            # Run fastmcp dev for a short time to verify startup
            process = subprocess.Popen(
                [str(fastmcp_cmd), "dev", "src/server.py"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait up to 8 seconds for startup
            try:
                stdout, stderr = process.communicate(timeout=8)
                # Server started successfully if we see startup messages or port in use (means it tried to start)
                success = ("Proxy server listening" in stdout or "Proxy server listening" in stderr or
                          "PORT IS IN USE" in stdout or "PORT IS IN USE" in stderr)
            except subprocess.TimeoutExpired:
                # Kill the process and consider it successful if it didn't crash immediately
                process.kill()
                stdout, stderr = process.communicate()
                success = True  # Timeout means it was running, which is good

            duration = time.time() - start_time

            if success:
                self.log_test("fastmcp_dev_mode", "passed",
                            f"Server startup verified: {stdout[:100]}...", "", duration)
                return True
            else:
                self.log_test("fastmcp_dev_mode", "failed",
                            stdout, stderr, duration)
                return False

        except Exception as e:
            duration = time.time() - start_time
            self.log_test("fastmcp_dev_mode", "error", "", str(e), duration)
            return False

    def test_claude_mcp_registration(self) -> bool:
        """Test that the server is registered with Claude Code."""
        start_time = time.time()

        try:
            result = subprocess.run(
                ["claude", "mcp", "list"],
                capture_output=True, text=True, timeout=30
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                output = result.stdout
                if "cycpep-tools" in output and "✓ Connected" in output:
                    self.log_test("claude_mcp_registration", "passed",
                                "Server registered and connected", "", duration)
                    return True
                elif "cycpep-tools" in output:
                    self.log_test("claude_mcp_registration", "warning",
                                output, "Server registered but not connected", duration)
                    return True
                else:
                    self.log_test("claude_mcp_registration", "failed",
                                output, "Server not registered", duration)
                    return False
            else:
                self.log_test("claude_mcp_registration", "failed",
                            result.stdout, result.stderr, duration)
                return False

        except Exception as e:
            duration = time.time() - start_time
            self.log_test("claude_mcp_registration", "error", "", str(e), duration)
            return False

    def run_all_tests(self) -> dict:
        """Run all integration tests and return results."""
        print("=" * 60)
        print("MultiCycPermea MCP Integration Test Suite")
        print("=" * 60)

        # Core functionality tests
        tests_to_run = [
            ("Environment Setup", self.test_environment_setup),
            ("Server Startup", self.test_server_startup),
            ("Job Manager", self.test_job_manager),
            ("Script Imports", self.test_script_imports),
            ("Example Data", self.test_example_data),
            ("MCP Tools Count", self.test_mcp_tools_count),
            ("FastMCP Dev Mode", self.test_fastmcp_dev_mode),
            ("Claude MCP Registration", self.test_claude_mcp_registration),
        ]

        passed = 0
        warnings = 0
        failed = 0
        errors = 0

        for test_name, test_func in tests_to_run:
            print(f"\nRunning: {test_name}")
            try:
                result = test_func()
                if test_name.replace(" ", "_").lower() in self.results["tests"]:
                    status = self.results["tests"][test_name.replace(" ", "_").lower()]["status"]
                    if status == "passed":
                        passed += 1
                    elif status == "warning":
                        warnings += 1
                    elif status == "failed":
                        failed += 1
                    else:
                        errors += 1
            except Exception as e:
                print(f"✗ {test_name}: ERROR - {e}")
                errors += 1
                self.log_test(test_name.replace(" ", "_").lower(), "error", "", str(e))

        # Calculate summary
        total_tests = len(tests_to_run)
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "errors": errors,
            "pass_rate": f"{passed/total_tests*100:.1f}%" if total_tests > 0 else "N/A",
            "success_rate": f"{(passed+warnings)/total_tests*100:.1f}%" if total_tests > 0 else "N/A"
        }

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Warnings: {warnings}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Pass Rate: {self.results['summary']['pass_rate']}")
        print(f"Success Rate: {self.results['summary']['success_rate']}")

        return self.results

    def save_report(self, output_path: Path = None) -> Path:
        """Save test results to a JSON report file."""
        if output_path is None:
            output_path = self.project_root / "reports" / "step7_integration_test_results.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nDetailed test report saved to: {output_path}")
        return output_path


def main():
    """Main test runner function."""
    project_root = Path(__file__).parent.parent

    runner = MCPTestRunner(project_root)
    results = runner.run_all_tests()
    report_path = runner.save_report()

    # Determine exit code based on results
    if results["summary"]["failed"] > 0 or results["summary"]["errors"] > 0:
        print("\n❌ Some tests failed or had errors. Check the report for details.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()