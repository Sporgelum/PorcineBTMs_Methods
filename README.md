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


### Improvements:  
##### Calculate MI (was the slowest step)
Replaced slow mutual_info_classif (sklearn) with fast histogram-based MI
Expected speedup: 50-100x faster 🚀
Old method: 7.86 hours → New method: ~5-10 minutes estimated

### Newer implementation after meeting 05.03.2026
In order to filter significant gene-pairs, withing dataset MI was calculated for gene pairs and 30.000 permutations were run to break every possible "random effect" and computed, selecting those that are present with a p-val < 0.001 and present in at least 3 studies. Master network is built using the results of the per-study nodes and edges. For module detection MCODE was applied, being the largest difference to other clustering methods that overlap is accepted.

To run this  updated version we are using `sbatch submit_pval_job.sh` --> including `generate_net_python_pval.py`


Now i updated the workflow that the gene pairs with p-val < 0.001 need to be in at least 30% studies, due to the previous run which was generating a huge number of edges and this probably affect the final module detection by MCODE.

TODO: Furthermore, we shall code the remove of ribosomal gene pairs as it will probably create another huge mess.


## Git update summary (MINE package pipeline)

The new implementation is now packaged in:

workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED

Key points of this implementation:

1. MI estimation moved to neural MINE (continuous data) instead of the older histogram/minet-style workflow.
2. The pipeline is a proper Python package (`mine_network`) with modular components:
    data loading, pre-screening, MINE estimation, permutation testing, master network, MCODE, and annotation.
3. CLI entry point is `run_pipeline.py` with configurable arguments for counts, metadata, device, permutations, pre-screening, and output.
4. The SLURM workflow `submit_network_job.sh` runs the packaged pipeline and writes outputs to the local `output/` directory.
5. Consensus network logic supports minimum study count and study-fraction thresholds.
6. Optional gene filtering is implemented (ribosomal/miRNA/custom exclusion) before pair selection.
7. Module annotation via GMT enrichment is integrated in the package.

Run on cluster:

sbatch workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED/submit_network_job.sh

Direct run:

python workingEnvironment/03_network/MINE_NETWORK_PERMUTATION_FILTER_MCODE_ANNOTATED/run_pipeline.py --output ./output


