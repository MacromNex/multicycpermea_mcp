# Comprehensive Test Prompts for MultiCycPermea MCP Server

This file contains a systematic set of test prompts to validate all functionality of the cyclic peptide MCP server. These prompts should be run in Claude Code to verify the integration works correctly.

## Tool Discovery Tests

### Prompt 1: List All Tools
**Test Type**: Discovery
**Expected Result**: Should list all 14 available MCP tools
**Prompt**:
```
What MCP tools are available for cyclic peptides? Give me a brief description of each tool and organize them by category.
```

### Prompt 2: Tool Details
**Test Type**: Tool Inspection
**Expected Result**: Detailed parameter information
**Prompt**:
```
Explain how to use the generate_peptide_images tool, including all required and optional parameters.
```

### Prompt 3: Server Information
**Test Type**: Server Status
**Expected Result**: Server metadata and status
**Prompt**:
```
Get information about the cycpep MCP server - what scripts are available and what's the server status?
```

## Synchronous Tool Tests (Fast Operations)

### Prompt 4: Generate Peptide Images (Small Dataset)
**Test Type**: Image Generation
**Expected Result**: Generated images with metadata
**Prompt**:
```
Generate molecular structure images for the small test dataset (test_small.csv). Use the default configuration and save to results/test_images/.
```

### Prompt 5: Feature Analysis
**Test Type**: Computational Analysis
**Expected Result**: Comparison of feature fusion methods
**Prompt**:
```
Run feature analysis on the small test dataset comparing concatenated, averaged, and max-pooled feature fusion methods. Save results to results/test_analysis/.
```

### Prompt 6: CSV Validation
**Test Type**: Data Validation
**Expected Result**: Validation report with any issues
**Prompt**:
```
Validate the structure and content of the test_small.csv file. Check for missing values, invalid SMILES, and data consistency.
```

### Prompt 7: Example Data Information
**Test Type**: Data Discovery
**Expected Result**: Information about available datasets
**Prompt**:
```
What example datasets are available for testing? Show me the structure and size of each dataset.
```

### Prompt 8: Error Handling Test
**Test Type**: Error Handling
**Expected Result**: Structured error response
**Prompt**:
```
Try to generate images for a non-existent CSV file called "nonexistent_file.csv".
```

## Asynchronous Tool Tests (Long-Running Operations)

### Prompt 9: Submit Batch Image Generation
**Test Type**: Async Job Submission
**Expected Result**: Job ID and status
**Prompt**:
```
Submit a batch job to generate images for the full test.csv dataset. Use high-quality settings and save to results/batch_images/.
```

### Prompt 10: Check Job Status
**Test Type**: Job Monitoring
**Expected Result**: Current job status and progress
**Prompt**:
```
Check the status of job [use actual job_id from previous prompt]. Show me the current progress and any available logs.
```

### Prompt 11: Submit Batch Feature Analysis
**Test Type**: Batch Processing
**Expected Result**: Job submission confirmation
**Prompt**:
```
Submit a batch feature analysis job for the validation dataset (val.csv). Compare all three fusion methods and save results to results/batch_analysis/.
```

### Prompt 12: List All Jobs
**Test Type**: Job Management
**Expected Result**: List of all submitted jobs
**Prompt**:
```
Show me all submitted jobs, their current status, and when they were submitted.
```

### Prompt 13: Get Job Results
**Test Type**: Result Retrieval
**Expected Result**: Complete job results (when job is finished)
**Prompt**:
```
Get the complete results for job [use completed job_id]. Include any output files and performance metrics.
```

### Prompt 14: View Job Logs
**Test Type**: Debugging
**Expected Result**: Recent log entries
**Prompt**:
```
Show me the last 30 lines of logs for job [use job_id]. Include any error messages or warnings.
```

### Prompt 15: Cancel Running Job
**Test Type**: Job Control
**Expected Result**: Job cancellation confirmation
**Prompt**:
```
Cancel the running job [use active job_id] and confirm it was properly terminated.
```

## Configuration and Utility Tests

### Prompt 16: Load Configuration Template
**Test Type**: Configuration Management
**Expected Result**: Configuration template with parameters
**Prompt**:
```
Load the configuration template for image generation. Show me all available parameters and their default values.
```

### Prompt 17: Custom Configuration
**Test Type**: Custom Settings
**Expected Result**: Successful execution with custom settings
**Prompt**:
```
Generate images for 5 peptides using custom settings: image_size=800x800, molecule_size=0.8, and save_format=PNG.
```

## End-to-End Workflow Tests

### Prompt 18: Complete Analysis Pipeline
**Test Type**: Full Workflow
**Expected Result**: End-to-end processing results
**Prompt**:
```
Run a complete analysis pipeline:
1. Validate the test_small.csv dataset
2. Generate molecular images for all peptides
3. Run feature analysis comparing all fusion methods
4. Summarize the results with performance metrics

Use the test_small.csv dataset and save all outputs to results/pipeline_test/.
```

### Prompt 19: Permeability Prediction Workflow
**Test Type**: Research Workflow
**Expected Result**: Research-ready analysis
**Prompt**:
```
I want to analyze membrane permeability patterns:
1. Load the training dataset (train.csv)
2. Generate images for the first 50 peptides
3. Analyze features using averaged fusion method
4. Create a summary of permeability values and molecular descriptors

This is for a research publication on cyclic peptide permeability.
```

### Prompt 20: Batch Processing Workflow
**Test Type**: Large-Scale Processing
**Expected Result**: Efficient batch processing
**Prompt**:
```
Set up batch processing for the complete dataset:
1. Submit batch image generation for the full test.csv dataset
2. While that's running, submit batch feature analysis for val.csv
3. Monitor progress of both jobs
4. When complete, summarize the total processing time and results

Use default configurations for both jobs.
```

## Error Recovery and Edge Cases

### Prompt 21: Invalid Dataset Test
**Test Type**: Error Handling
**Expected Result**: Graceful error handling
**Prompt**:
```
Try to process a CSV file with invalid SMILES strings and missing data. Test the robustness of the validation and processing tools.
```

### Prompt 22: Large Dataset Warning
**Test Type**: Performance Testing
**Expected Result**: Appropriate sync/async selection
**Prompt**:
```
What happens if I try to use the synchronous image generation tool on the full training dataset (5000+ peptides)? Should it switch to async mode?
```

### Prompt 23: Job Recovery Test
**Test Type**: System Recovery
**Expected Result**: Job state persistence
**Prompt**:
```
If the server restarts, can it recover the status of previously submitted jobs? Check job persistence.
```

### Prompt 24: Cleanup Test
**Test Type**: Maintenance
**Expected Result**: Successful cleanup
**Prompt**:
```
Clean up old completed jobs that are more than 1 hour old. Show me how much storage space was freed.
```

## Performance and Resource Tests

### Prompt 25: Performance Benchmark
**Test Type**: Performance Measurement
**Expected Result**: Timing and resource usage
**Prompt**:
```
Benchmark the performance of image generation for 10, 50, and 100 peptides. Report processing time, memory usage, and throughput (peptides/second).
```

### Prompt 26: Resource Monitoring
**Test Type**: Resource Usage
**Expected Result**: Resource consumption data
**Prompt**:
```
Monitor resource usage during a batch processing job. Report CPU, memory, and disk usage throughout the job execution.
```

## Integration Verification Tests

### Prompt 27: Cross-Tool Integration
**Test Type**: Tool Integration
**Expected Result**: Seamless tool interaction
**Prompt**:
```
Use multiple tools in sequence:
1. Get example data information
2. Validate a dataset
3. Generate images for valid peptides
4. Analyze features from the generated images
5. Submit any large batches as async jobs

Demonstrate that tools work together seamlessly.
```

### Prompt 28: File Path Resolution
**Test Type**: Path Handling
**Expected Result**: Correct file handling
**Prompt**:
```
Test file path resolution by:
1. Using relative paths (./examples/data/test_small.csv)
2. Using absolute paths
3. Using paths with spaces
4. Verify all tools handle paths correctly
```

---

## Expected Success Criteria

For the MCP integration to be considered successful, all of the following should be true:

### Tool Discovery (Prompts 1-3)
- [ ] All 14 tools are discoverable and listed correctly
- [ ] Tool descriptions are clear and accurate
- [ ] Server information is accessible

### Synchronous Operations (Prompts 4-8)
- [ ] All sync tools complete within 5 minutes for small datasets
- [ ] Results are returned in structured format
- [ ] Error handling provides clear, helpful messages
- [ ] File operations work with various path formats

### Asynchronous Operations (Prompts 9-15)
- [ ] Jobs submit successfully with unique IDs
- [ ] Job status updates correctly (submitted -> running -> completed/failed)
- [ ] Job logs are accessible and informative
- [ ] Job cancellation works properly
- [ ] Results are retrievable when jobs complete

### Configuration and Utilities (Prompts 16-17)
- [ ] Configuration templates load successfully
- [ ] Custom configurations are applied correctly
- [ ] All utility functions work as expected

### End-to-End Workflows (Prompts 18-20)
- [ ] Multi-step workflows execute completely
- [ ] Data flows correctly between steps
- [ ] Research workflows produce publication-ready results
- [ ] Batch processing handles large datasets efficiently

### Error Handling and Edge Cases (Prompts 21-24)
- [ ] Invalid inputs are handled gracefully
- [ ] System recovery works after interruptions
- [ ] Performance warnings are shown for large datasets
- [ ] Cleanup operations work correctly

### Performance (Prompts 25-26)
- [ ] Performance is within acceptable ranges
- [ ] Resource usage is reasonable
- [ ] Throughput meets expectations

### Integration (Prompts 27-28)
- [ ] Tools integrate seamlessly with each other
- [ ] File path resolution works correctly
- [ ] Cross-tool data flow is reliable

---

## Notes for Testers

1. **Run tests in order**: Some tests depend on jobs created by previous tests
2. **Replace placeholders**: Use actual job IDs where indicated with [job_id]
3. **Check file outputs**: Verify that files are created in the expected locations
4. **Monitor resource usage**: Watch for memory leaks or excessive resource consumption
5. **Document issues**: Note any errors or unexpected behavior for debugging
6. **Test timing**: Allow sufficient time for async jobs to complete before checking results

## Test Automation

This test suite can be partially automated using the test runner script `tests/run_integration_tests.py`. However, many tests require manual verification of outputs and Claude Code interaction.