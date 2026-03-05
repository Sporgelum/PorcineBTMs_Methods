# Analysis Logging and Reporting Guide

Your network analysis now includes comprehensive logging and timing tracking!

## 📊 What Gets Logged

### 1. **Console Output + Log File** (Dual Output)
Everything printed to console is simultaneously saved to:
```
network_analysis_YYYYMMDD_HHMMSS.log
```

### 2. **Analysis Report** (Summary Report)
Comprehensive summary saved to:
```
analysis_report_YYYYMMDD_HHMMSS.txt
```

## 📝 Log File Contents

The log file includes:
- ✅ Every step of the analysis with timestamps
- ✅ All print statements and progress messages
- ✅ Timing for each major step
- ✅ Error messages (if any)
- ✅ GPU information (if used)
- ✅ All configuration parameters

**Example log entry:**
```
[2026-03-04 11:30:45] Starting: MI Matrix Computation (CPU parallelized)
--------------------------------------------------------------------------------
[INFO] Computing MI for 32763 genes using 24 cores...
[INFO] Discretizing expression data into 5 bins...
Progress: [##########] 100%
--------------------------------------------------------------------------------
[2026-03-04 13:45:23] Completed: MI Matrix Computation (CPU parallelized)
[TIMING] Duration: 2.24 hours (134.6m)
```

## 📈 Report File Contents

The report includes:

### Configuration
```
Input Data: /path/to/logCPM_matrix_filtered_samples.csv
Number of Bins: 5
Discretization Strategy: quantile
Threshold Percentile: 95%
CPU Cores: All available
```

### Results Summary
```
Number of Genes: 32763
Number of Samples: 485
MI Matrix Size: 32763 × 32763
MI Range: 0.0000 to 1.2345
CLR Matrix Range: 0.0000 to 8.7654
CLR Number of Edges: 53678
CLR Number of Modules: 342
```

### Timing Breakdown (Sorted by Duration)
```
TIMING BREAKDOWN
--------------------------------------------------------------------------------
MI Matrix Computation (CPU parallelized)          :    2.24 hours (134.6m) ( 65.3%)
Louvain Clustering (MI+CLR)                       :   45.67 minutes (2740s) ( 22.4%)
Network Thresholding                              :   18.34 minutes (1100s) (  9.0%)
CLR Transformation (GPU accelerated)              :         3.21 seconds  (  0.2%)
Data Loading                                       :         2.45 seconds  (  0.1%)
Saving MI+CLR Results                             :         1.89 seconds  (  0.1%)
--------------------------------------------------------------------------------
TOTAL TIME                                        :    3.43 hours (205.8m)
```

### GPU Information (if used)
```
Using GPU: NVIDIA GeForce RTX 2080 Ti
GPU Memory: 11.0 GB
```

### Output Files
```
Log file: network_analysis_20260304_113045.log
Report file: analysis_report_20260304_113045.txt
  - MI+CLR results saved with prefix: mi_clr_python
  - GRNBoost2 results saved with prefix: grnboost2_python
```

## 🚀 How to Run with Logging

### Option 1: Interactive (for testing)
```bash
cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network
source /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/bin/activate

# Load PyTorch for GPU (if available)
module load PyTorch

# Run analysis
python generate_net_python.py
```

All output will be shown on screen AND saved to log file automatically.

### Option 2: Submit to SLURM (recommended for full analysis)
```bash
cd /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network

# Edit partition name if needed
nano submit_network_job.sh  # Change --partition=compute to your cluster's name

# Submit job
sbatch submit_network_job.sh

# Check status
squeue -u $USER

# Watch log file in real-time
tail -f network_analysis_*.log
```

SLURM will create additional files:
- `slurm_JOBID_network_analysis.out` - SLURM standard output
- `slurm_JOBID_network_analysis.err` - SLURM error output

## 📁 Output Files After Run

```bash
03_network/
├── network_analysis_20260304_113045.log         # Complete execution log
├── analysis_report_20260304_113045.txt          # Summary report
├── slurm_12345_network_analysis.out             # SLURM output (if using SLURM)
├── slurm_12345_network_analysis.err             # SLURM errors (if using SLURM)
├── CLR_adjacency_matrix_mi_clr_python.mtx       # MI+CLR adjacency matrix
├── CLR_network_edgelist_mi_clr_python.txt       # MI+CLR edge list
├── CLR_network_weighted_mi_clr_python.txt       # MI+CLR weighted edges
├── CLR_network_mi_clr_python.graphml            # MI+CLR graph (Cytoscape)
├── BTM_modules_mi_clr_python.tsv                # MI+CLR modules
└── ... (similar files for GRNBoost2 if available)
```

## 🔍 Monitoring Progress

### Real-time Log Monitoring
```bash
# Follow log file as it updates
tail -f network_analysis_*.log

# Show only timing information
grep "TIMING" network_analysis_*.log

# Show only warnings/errors
grep -E "WARNING|ERROR" network_analysis_*.log
```

### Check GPU Usage (if using GPU acceleration)
```bash
# On the compute node where job is running
watch -n 1 nvidia-smi
```

## 📊 Comparing Multiple Runs

Each run gets a unique timestamp, so you can compare:

```bash
# List all analysis reports
ls -lht analysis_report_*.txt

# Compare timing between runs
grep "TOTAL TIME" analysis_report_*.txt

# Compare results
grep "Number of Edges" analysis_report_*.txt
grep "Number of Modules" analysis_report_*.txt
```

## 🐛 Troubleshooting

### Log file not created?
Check permissions:
```bash
ls -ld /data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network
touch test.log && rm test.log  # Test write permissions
```

### Want to disable logging?
Comment out these lines in `generate_net_python.py`:
```python
# sys.stdout = TeeLogger(LOG_FILE)
```

### Log file too large?
Large log files are normal for big datasets. To reduce size:
- The log compresses well: `gzip network_analysis_*.log`
- Remove old logs: `rm network_analysis_2026*.log`

## 💡 Tips

1. **Keep your logs organized**: Create a logs subdirectory
   ```bash
   mkdir -p logs
   # Then move old logs: mv network_analysis_*.log logs/
   ```

2. **Quick timing check**: Look at report file instead of full log
   ```bash
   cat analysis_report_*.txt
   ```

3. **Archive completed runs**:
   ```bash
   tar -czf run_20260304.tar.gz network_analysis_20260304*.log analysis_report_20260304*.txt *.mtx *.graphml
   ```

4. **Email notification on completion** (SLURM):
   Add to submit script:
   ```bash
   #SBATCH --mail-type=END,FAIL
   #SBATCH --mail-user=your.email@example.com
   ```

## 📋 Example Report Output

```
================================================================================
GENE NETWORK ANALYSIS REPORT
================================================================================

Generated: 2026-03-04 15:30:45
Script: generate_net_python.py
Working Directory: /data/users/mbotos/Environments/.../03_network

CONFIGURATION
--------------------------------------------------------------------------------
Input Data: /data/users/mbotos/.../logCPM_matrix_filtered_samples.csv
Number of Bins: 5
Discretization Strategy: quantile
Threshold Percentile: 95%
CPU Cores: All available

RESULTS SUMMARY
--------------------------------------------------------------------------------
Number of Genes: 32763
Number of Samples: 485
MI Matrix Size: 32763 × 32763
MI Range: 0.0000 to 1.2345
CLR Matrix Range: 0.0000 to 8.7654
CLR Threshold Value: 4.5678
CLR Number of Edges: 53678
CLR Number of Modules: 342

TIMING BREAKDOWN
--------------------------------------------------------------------------------
MI Matrix Computation (CPU parallelized)          :    2.24 hours (134.6m) ( 65.3%)
Louvain Clustering (MI+CLR)                       :   45.67 minutes (2740s) ( 22.4%)
Network Thresholding                              :   18.34 minutes (1100s) (  9.0%)
CLR Transformation (GPU accelerated)              :         3.21 seconds  (  0.2%)
Data Loading                                       :         2.45 seconds  (  0.1%)
Saving MI+CLR Results                             :         1.89 seconds  (  0.1%)
--------------------------------------------------------------------------------
TOTAL TIME                                        :    3.43 hours (205.8m)

GPU INFORMATION
--------------------------------------------------------------------------------
Using GPU: NVIDIA GeForce RTX 2080 Ti
GPU Memory: 11.0 GB

OUTPUT FILES
--------------------------------------------------------------------------------
Log file: network_analysis_20260304_113045.log
Report file: analysis_report_20260304_113045.txt
  - MI+CLR results saved with prefix: mi_clr_python

================================================================================
```

---

**You're all set!** Your analysis will now automatically log everything with detailed timing information. 🎉
