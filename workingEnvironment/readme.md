For environment to work with R, in bnode... install :
install.packages(c("jsonlite", "rlang"), 
                 type = "source",
                 lib = "/storage/homefs/mb23h197/RLibs/R_4.4.2/")

                 It is a pain, to work in ubelix if always need to rebuild and reinstall packages as CPUs are different setups...

                 
# In a fresh R session
.libPaths("/storage/homefs/mb23h197/RLibs/R_4.4.2/")

# Remove and reinstall from source
remove.packages(c("dplyr", "limma"))

# Install dplyr from CRAN (compiles for YOUR CPU)
install.packages("dplyr", type = "source", 
                 lib = "/storage/homefs/mb23h197/RLibs/R_4.4.2/")

# Install limma from Bioconductor
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("limma", type = "source", 
                     lib = "/storage/homefs/mb23h197/RLibs/R_4.4.2/")





As of date: 25.02.2026 I copied all to IBU and working there with R, as in ubelix without the option to use cnode it is a pain to work...

PD: in ubelix to connect if the user ssh profile key is not provided may load the emari ("microsoft") profile and not connect keep in mind!

# 03 network.
TO run the network generation, there are two methods i am running now. 
    -1. MI with CLR using GPU
    -2. GRN where we use the rest of genes to predict each genes expression

for these: the main script is generate_net_python.py
inside we find the functions taht we need and it incorporates the clr_gpu.py function which runs CLR, a bottleneck CPU process that can be parallelized and/or gpu"ized" but it was not yet, so i did it.

for running the whole can be used:
    -1. run_network_analysis.sh, but does not keep the log...
    -2. run_network_gpu.sh, which only runs the same script... as the above.
    -3. submit_network_job.sh, which runs loggend generate_net_python.py and in a slurm job directly. preferentially, use 3 or 1.

        currently as of date: 04.03.2026, the combo: 
        
    `sbatch submit_network_job.sh`  
with resources in the sbatch: 
        `#!/bin/bash
        `#SBATCH --job-name=gene_network`
        `#SBATCH --time=48:00:00`
        `#SBATCH --cpus-per-task=64`
        `##SBATCH --mem=527G`
        `#SBATCH --mem-per-cpu=6G`
        `#SBATCH --output=slurm_%j_network_analysis.out`
        `#SBATCH --error=slurm_%j_network_analysis.err`
        `#SBATCH --partition=pgpu       `
        `#SBATCH --gres=gpu:1          # Request 1 GPU (optional, for GPU-accelerated CLR)`
works correctly.
