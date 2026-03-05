# %%
.libPaths() # --> kind of already established in the R_Home
.libPaths("/storage/homefs/mb23h197/RLibs/R_4.4.2")
#install.packages("httpgd")
#install.packages("remotes")
#remotes::install_github("nx10/httpgd")
#BiocManager::install("ggplot2")

# %% test
print(mtcars)

# %%
library("edgeR")
library("dplyr")
library("ggplot2")

# Install uwot if needed: install.packages("uwot")
# if (!requireNamespace("uwot", quietly = TRUE)) {
#   cat("Installing uwot package for UMAP...\n")
#   install.packages("uwot")
# }
library("uwot")
library("pheatmap")


# %% read in the counts from each folder within each sample
# path to projects
projects_path <- "/storage/research/vetsuisse_ivi/Summerfield/Marius/Pig/mapped_bam"
# get the list of projects
projects <- list.dirs(projects_path, recursive = FALSE)

# initialize an empty list to store counts samples per project
counts_samples <- list()

# loop through each project and read in the counts
for (project in projects) {
  cat("\nProcessing project:", basename(project), "\n")
  
  # Find all ReadsPerGene.out.tab files in the project directory
  # Searches in subdirectories too (recursive = TRUE)
  samples <- list.files(project, recursive = TRUE, full.names = TRUE, 
                        pattern = "_ReadsPerGene\\.out\\.tab$", ignore.case = TRUE)  
  
  cat("  Found", length(samples), "sample files\n")
  
  # loop through each sample file and read in the counts
  for (sample_file in samples) {
    # Extract sample name from filename
    sample_name <- basename(sample_file)
    sample_name <- gsub("_ReadsPerGene\\.out\\.tab$", "", sample_name, ignore.case = TRUE)
    
    if (nchar(sample_name) > 0) {
      # read in the counts
      # STAR ReadsPerGene.out.tab format: gene_id, unstranded, stranded_forward, stranded_reverse
      # Column 1: gene ID
      # Column 2: unstranded counts
      # Column 3: strand-specific forward
      # Column 4: strand-specific reverse
      counts_data <- read.table(sample_file, header = FALSE, row.names = 1, 
                                 col.names = c("gene_id", "unstranded", "forward", "reverse"),
                                 stringsAsFactors = FALSE)
      
      # Skip the first 4 rows (STAR summary statistics: N_unmapped, N_multimapping, N_noFeature, N_ambiguous)
      counts_data <- counts_data[-(1:4), , drop = FALSE]
      
      # Store counts (using unstranded column 2, change if needed)
      counts_samples[[sample_name]] <- counts_data$unstranded
      # IMPORTANT: Assign gene IDs as names to the vector
      names(counts_samples[[sample_name]]) <- rownames(counts_data)
      
      cat("  Read counts for:", sample_name, "- Genes:", length(counts_samples[[sample_name]]), "\n")
    } else {
      warning(paste("Could not extract sample name from:", basename(sample_file)))
    }
  }
}

# %% check the counts_samples list
cat("\nTotal samples loaded:", length(counts_samples), "\n")
print(names(counts_samples))

# %% check the first few rows of the counts for the first sample
if (length(counts_samples) > 0) {
  first_sample <- names(counts_samples)[1]
  cat("\nFirst few genes from sample:", first_sample, "\n")
  print(head(counts_samples[[first_sample]]))
}

# %% Create count matrix
# Convert the list to a matrix
# First, get all unique gene IDs
all_genes <- unique(unlist(lapply(counts_samples, names)))
cat("\nTotal unique genes:", length(all_genes), "\n")

# Create an empty matrix
count_matrix <- matrix(0, nrow = length(all_genes), ncol = length(counts_samples))
rownames(count_matrix) <- all_genes
colnames(count_matrix) <- names(counts_samples)

# Fill in the counts
for (i in seq_along(counts_samples)) {
  sample_name <- names(counts_samples)[i]
  gene_names <- names(counts_samples[[i]])
  count_matrix[gene_names, sample_name] <- counts_samples[[i]]
}

# Convert to numeric matrix
count_matrix <- apply(count_matrix, 2, as.numeric)
rownames(count_matrix) <- all_genes

cat("Count matrix dimensions:", dim(count_matrix), "\n")
cat("Samples:", ncol(count_matrix), "\n")
cat("Genes:", nrow(count_matrix), "\n")

write.table(count_matrix, file = "/storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/workingEnvironment/02_counts/raw_count_matrix.csv", sep = "\t", quote = FALSE)
cat("Raw count matrix saved as raw_count_matrix.csv\n")
# %% Filter low-expressed genes (optional but recommended)
# Keep genes with at least 1 count in at least 10 samples
keep <- rowSums(count_matrix > 1) >= 10
count_matrix_filtered <- count_matrix[keep, ]
cat("\nFiltered count matrix dimensions:", dim(count_matrix_filtered), "\n")

# Additional quality check: remove genes with zero variance
gene_var <- apply(count_matrix_filtered, 1, var)
zero_var_genes <- sum(gene_var == 0)
if (zero_var_genes > 0) {
  cat("Removing", zero_var_genes, "genes with zero variance\n")
  count_matrix_filtered <- count_matrix_filtered[gene_var > 0, ]
}

# %% Create DGEList object for edgeR
count_matrix_filtered <- count_matrix_filtered[,colSums(count_matrix_filtered) > 1000000]
dim(count_matrix_filtered)
dge <- DGEList(counts = count_matrix_filtered)
cat("\nDGEList created with", nrow(dge), "genes and", ncol(dge), "samples\n")
length(rownames(dge$samples)) 

#dge <- dge[,dge$samples$lib.size >= 1000000] # Remove samples with library size < 1e6
length(rownames(dge$samples)) # Check how many samples remain
cat("Samples with library size < 1e6 removed\n")
cat("\nDGEList created with", nrow(dge), "genes and", ncol(dge), "samples\n")


#save
write.table(count_matrix_filtered, file = "/storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/workingEnvironment/02_counts/filtered_count_matrix.csv", sep = "\t", quote = FALSE)
cat("Filtered count matrix saved as filtered_count_matrix.csv\n")

# %% Normalization with edgeR (TMM normalization)
# TMM assumes most genes are not DE - can fail with very heterogeneous samples
# Alternative methods if TMM fails:
# - "RLE" (DESeq2 method, more robust to outliers)
# - "upperquartile" (uses 75th percentile instead of mean)
# - "none" (no normalization, just library size)

# Try TMM first, if it produces extreme factors, use RLE
dge <- calcNormFactors(dge, method = "TMM")
cat("TMM normalization factors calculated\n")

# Check if TMM produced reasonable factors
if (max(dge$samples$norm.factors) / min(dge$samples$norm.factors) > 5) {
  cat("TMM produced very heterogeneous factors, switching to RLE normalization...\n")
  dge <- calcNormFactors(dge, method = "RLE")
  cat("RLE normalization factors calculated\n")
}

print(head(dge$samples, 20))

# %% Check for extreme normalization factors (indicates problematic samples)
cat("\n=== CHECKING FOR PROBLEMATIC SAMPLES ===\n")
norm_factors <- dge$samples$norm.factors
lib_sizes <- dge$samples$lib.size

cat("Normalization factor range:", round(min(norm_factors), 3), "to", round(max(norm_factors), 3), "\n")
cat("Median norm factor:", round(median(norm_factors), 3), "\n")
cat("Library size range:", min(lib_sizes), "to", max(lib_sizes), "\n")
cat("Library size median:", median(lib_sizes), "\n")

# Filter based on BOTH norm factors AND library size outliers
# Identify outlier samples (norm factors < 0.6 or > 1.7, OR extreme library sizes)
lib_median <- median(lib_sizes)
lib_mad <- mad(lib_sizes)  # Median absolute deviation - robust to outliers

# Keep samples within 3 MAD of median library size
lib_low <- lib_median - 3 * lib_mad
lib_high <- lib_median + 3 * lib_mad

outlier_norm <- norm_factors < 0.6 | norm_factors > 1.7
outlier_lib <- lib_sizes < lib_low | lib_sizes > lib_high
outlier_samples <- outlier_norm | outlier_lib

if (sum(outlier_samples) > 0) {
  cat("\nWARNING:", sum(outlier_samples), "samples with extreme values:\n")
  cat("  Norm factor outliers:", sum(outlier_norm), "\n")
  cat("  Library size outliers:", sum(outlier_lib), "\n")
  
  cat("\nRemoving", sum(outlier_samples), "outlier samples to prevent numerical issues...\n")
  dge <- dge[, !outlier_samples]
  cat("Retained", ncol(dge), "samples after filtering\n")
} else {
  cat("All samples have acceptable values\n")
}

# %% Check for any remaining issues
cat("\n=== DATA QUALITY CHECKS ===\n")
cat("Final sample count:", ncol(dge), "\n")
cat("Final gene count:", nrow(dge), "\n")
cat("Library size range:", min(dge$samples$lib.size), "to", max(dge$samples$lib.size), "\n")
cat("Norm factor range:", round(min(dge$samples$norm.factors), 3), "to", 
    round(max(dge$samples$norm.factors), 3), "\n")

# Check for any zero counts or NA values
if (any(is.na(dge$counts))) {
  cat("WARNING: NA values detected in count matrix!\n")
}
if (any(is.infinite(dge$counts))) {
  cat("WARNING: Infinite values detected in count matrix!\n")
}

# %% Get normalized counts (log2-CPM) with manual calculation
cat("\n=== CALCULATING LOG-CPM (Manual Method) ===\n")

# Manual CPM calculation to avoid edgeR internal issues
# CPM = (counts / (lib.size * norm.factors)) * 1e6
# log2-CPM = log2((counts + prior) / ((lib.size * norm.factors) + 2*prior) * 1e6)

# Get effective library sizes (lib.size * norm.factors)
effective_lib_sizes <- dge$samples$lib.size * dge$samples$norm.factors
cat("Effective library size range:", round(min(effective_lib_sizes)), "to", round(max(effective_lib_sizes)), "\n")

# Manual calculation with prior count = 2 (more stable than 1)
prior_count <- 2
counts_plus_prior <- dge$counts + prior_count

# Calculate CPM manually
logCPM <- matrix(0, nrow = nrow(dge$counts), ncol = ncol(dge$counts))
rownames(logCPM) <- rownames(dge$counts)
colnames(logCPM) <- colnames(dge$counts)

for (i in 1:ncol(dge$counts)) {
  # Calculate for each sample
  lib_norm <- effective_lib_sizes[i] + 2 * prior_count
  logCPM[, i] <- log2((counts_plus_prior[, i] / lib_norm) * 1e6)
}

cat("Log2-CPM matrix created successfully:", dim(logCPM), "\n")

# Check for any problematic values
if (any(is.na(logCPM))) {
  cat("WARNING:", sum(is.na(logCPM)), "NA values in logCPM\n")
  # Replace NAs with minimum value
  logCPM[is.na(logCPM)] <- min(logCPM[!is.na(logCPM)])
}
if (any(is.infinite(logCPM))) {
  cat("WARNING:", sum(is.infinite(logCPM)), "Inf values in logCPM\n")
  # Replace Inf with maximum value
  logCPM[is.infinite(logCPM) & logCPM > 0] <- max(logCPM[!is.infinite(logCPM)])
  logCPM[is.infinite(logCPM) & logCPM < 0] <- min(logCPM[!is.infinite(logCPM)])
}

cat("logCPM range:", round(min(logCPM), 2), "to", round(max(logCPM), 2), "\n")

write.table(logCPM, file = "/storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/workingEnvironment/02_counts/logCPM_matrix_filtered_samples.csv", sep = "\t", quote = FALSE)
cat("Log-CPM matrix saved to logCPM_matrix_filtered_samples.csv\n")

# %% Principal Component Analysis (PCA)
message("PCA")
message("PCA")
message("PCA")
# Perform PCA on log-CPM values, transposing the matrix to have samples as rows and genes as columns
# Centering and scaling is important for PCA to ensure that all genes contribute equally to the analysis
matrix_pca <- logCPM
matrix_string <- "logCPM"

#matrix_pca <- count_matrix_filtered
#matrix_string <- "count_matrix_filtered"


pca_result <- prcomp(t(matrix_pca), center = TRUE, scale. = TRUE)

# Calculate variance explained
var_explained <- summary(pca_result)$importance[2, ] * 100

# Create PCA data frame
pca_data <- data.frame(
  Sample = colnames(matrix_pca),
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  PC3 = pca_result$x[, 3]
)
write.table(pca_data, file = paste0(output_dir, "pca_coordinates_", matrix_string, ".csv"), sep = "\t", quote = FALSE)
cat("\nPCA completed\n")
cat("Variance explained by PC1:", round(var_explained[1], 2), "%\n")
cat("Variance explained by PC2:", round(var_explained[2], 2), "%\n")


pca_data <- pca_data |> dplyr::left_join(md_df, by = c("Sample" = "Run")) # Join metadata to UMAP data for better interpretation


# %% Plot PCA
output_dir <- "/storage/homefs/mb23h197/Environments/2026_02_16_BTM_PIGS_+/workingEnvironment/02_counts/"
pca_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, label = Sample,color=SampleStyle,shape=SampleTissue)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_text(size = 2, hjust = 0, vjust = 0, nudge_x = 0.5, nudge_y = 0.5, check_overlap = TRUE) +
  labs(
    title = "PCA of Gene Expression",
    x = paste0("PC1 (", round(var_explained[1], 2), "%)"),
    y = paste0("PC2 (", round(var_explained[2], 2), "%)")
  ) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

print(pca_plot)
  
# Save PCA plot
ggsave(paste0(output_dir, "PCA_plot_", matrix_string, "_SampleStyle_Tissue.pdf"), pca_plot, width = 12, height = 8)
cat("PCA plot saved as PCA_plot_", matrix_string, ".pdf\n")


# %% UMAP Analysis
message("UMAP")
message("UMAP")
message("UMAP")

umap_result <- umap(t(matrix_pca), n_neighbors = 15, min_dist = 0.1, n_components = 2)

# Create UMAP data frame
umap_data <- data.frame(
  Sample = colnames(matrix_pca),
  UMAP1 = umap_result[, 1],
  UMAP2 = umap_result[, 2]
)
write.table(umap_data, file = paste0(output_dir, "umap_coordinates_", matrix_string, ".csv"), sep = "\t", quote = FALSE)
cat("\nUMAP completed\n")

umap_data <- umap_data |> dplyr::left_join(md_df, by = c("Sample" = "Run")) # Join metadata to UMAP data for better interpretation
head(umap_data)
unique(umap_data$BioProject) # Check unique Bioprojects in the metadata
unique(umap_data$SampleStyle) # Check unique SampleStyles in the metadata
# %% Plot UMAP
umap_plot <- ggplot(umap_data, aes(x = UMAP1, y = UMAP2, label = Sample,color=SampleTreatment)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_text(size = 2, hjust = 0, vjust = 0, nudge_x = 0.5, nudge_y = 0.5, check_overlap = TRUE) +
  labs(
    title = "UMAP of Gene Expression",
    x = "UMAP1",
    y = "UMAP2"
  ) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

print(umap_plot)

# Save UMAP plot
ggsave(paste0(output_dir, "UMAP_plot_", matrix_string, "_SampleTreatment.pdf"), umap_plot, width = 12, height = 8)
cat("UMAP plot saved as UMAP_plot_", matrix_string, ".pdf\n")

# %% Sample-to-Sample Correlation Analysis
# Calculate correlation matrix
message("Calculating sample-to-sample correlation matrix...")
message("Calculating sample-to-sample correlation matrix...")
message("Calculating sample-to-sample correlation matrix...")
message("using matrix_pca variable which is defined above!")
cor_matrix <- cor(matrix_pca, method = "spearman")

cat("\nCorrelation matrix calculated\n")
cat("Dimensions:", dim(cor_matrix), "\n")

# %% Plot correlation heatmap
# Create correlation-based distance for clustering
# Use 1 - correlation as distance (ranges from 0 to 2)
cor_dist <- as.dist(1 - cor_matrix)

# Create heatmap with correlation-based clustering
# Note: pheatmap doesn't directly accept dist objects, so we use clustering_callback
pdf(paste0(output_dir, "correlation_heatmap_",matrix_string,".pdf"), width = 19, height = 24)
pheatmap(cor_matrix,
         clustering_distance_rows = cor_dist,
         clustering_distance_cols = cor_dist,
         clustering_method = "average",  # UPGMA - recommended for RNA-seq
         color = colorRampPalette(c("blue", "white", "red"))(100),
         main = "Sample-to-Sample Correlation (Spearman)",
         fontsize = 8,
         display_numbers = FALSE)
dev.off()
cat("Correlation heatmap saved as correlation_heatmap_", matrix_string, ".pdf\n")

# %% Hierarchical clustering based on correlation
sample_dist <- as.dist(1 - cor_matrix)

# Try different linkage methods:
# - "average" (UPGMA): balanced, recommended for RNA-seq
# - "ward.D2": creates compact clusters, good for identifying sample groups
# - "complete": tends to create elongated clusters
# - "single": sensitive to outliers (not recommended)

hc <- hclust(sample_dist, method = "average")  # Using average (UPGMA)

pdf(paste0(output_dir, "hierarchical_clustering_", matrix_string, ".pdf"), width = 14, height = 8)
plot(hc, main = "Hierarchical Clustering of Samples (Average Linkage)", 
     xlab = "", sub = "", hang = -1)
dev.off()
cat("Hierarchical clustering plot saved as hierarchical_clustering_", matrix_string, ".pdf\n")

# Optional: Try Ward's method for comparison
hc_ward <- hclust(sample_dist, method = "ward.D2")
pdf(paste0(output_dir, "hierarchical_clustering_ward_", matrix_string, ".pdf"), width = 14, height = 8)
plot(hc_ward, main = "Hierarchical Clustering of Samples (Ward's Method)", 
     xlab = "", sub = "", hang = -1)
dev.off()
cat("Ward clustering plot saved as hierarchical_clustering_ward_", matrix_string, ".pdf\n")

# %% Summary statistics
cat("\n=== ANALYSIS SUMMARY ===\n")
cat("Total samples:", ncol(count_matrix), "\n")
cat("Genes after filtering:", nrow(count_matrix_filtered), "\n")
cat("Normalization method: TMM\n")
cat("PCA variance explained (PC1-PC3):", paste(round(var_explained[1:3], 2), collapse = "%, "), "%\n")
cat("\nGenerated files:\n")
cat("  - PCA_plot_", matrix_string, ".pdf\n")
cat("  - UMAP_plot_", matrix_string, ".pdf\n")
cat("  - correlation_heatmap_", matrix_string, ".pdf\n")
cat("  - hierarchical_clustering_", matrix_string, ".pdf\n")
cat("  - hierarchical_clustering_ward_", matrix_string, ".pdf\n")

# %% Save processed data
# save(count_matrix, count_matrix_filtered, dge, logCPM, 
#      pca_result, pca_data, umap_result, umap_data, cor_matrix,
#      file = paste0(output_dir, "processed_counts_data_", matrix_string, ".RData"))
# cat("\nAll processed data saved to processed_counts_data_", matrix_string, ".RData\n")
# %% Load metadata (if available) and add to PCA/UMAP plots for better interpretation
message("Adding metadata")
message("Adding metadata")
message("Adding metadata")
message("Adding metadata")

metadata_file <- "/storage/research/vetsuisse_ivi/Summerfield/Marius/Pig/metadata/all_sra_projects_metadata_combined.csv"
md_df <- read.csv(metadata_file, stringsAsFactors = FALSE)
head(md_df)

#unique(md_df$Experiment_Title)
#unique(md_df$Sample_Title)
#unique(md_df$Layout)
unique(md_df$BioProject)

md_df <- md_df |> dplyr::mutate(SampleStyle = "Control")
md_df <- md_df |> dplyr::mutate(SampleTissue = "blood")
md_df <- md_df |> dplyr::mutate(SampleTreatment = "none")
md_df <- md_df |> dplyr::mutate(SampleTimepoint = "baseline")
md_df <- md_df |> dplyr::mutate(SampleData = "RNA-seq")
md_df <- md_df |> dplyr::mutate(SampleAge = "none")




md_df <- md_df |> dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA1200777", BioProject), "control,influenza", SampleStyle),
                                 SampleTissue = ifelse(grepl("PRJNA1200777",BioProject), "pbmc",SampleTissue),
                                 SampleTreatment = ifelse(grepl("PRJNA1200777",BioProject), "vacc_influenza",SampleTreatment),
                                 SampleTimepoint = ifelse(grepl("PRJNA1200777",BioProject), "baseline&21dpv",SampleTimepoint),
                                 SampleData = ifelse(grepl("PRJNA1200777",BioProject), "RNA-seq",SampleData),
                                 SampleAge = ifelse(grepl("PRJNA1200777",BioProject), "4_weeks",SampleAge)) |> 
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA1162510", BioProject), "control", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA1162510",BioProject), "lung_pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA1162510",BioProject), "wt-cystic",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA1162510",BioProject), "baseline",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA1162510",BioProject), "scRNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA1162510",BioProject), "piglets",SampleAge)) |> 
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA1197383", BioProject), "control", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA1197383",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA1197383",BioProject), "none",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA1197383",BioProject), "baseline",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA1197383",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA1197383",BioProject), "none",SampleAge)) |>
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA909593", BioProject), "control", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA909593",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA909593",BioProject), "none",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA909593",BioProject), "baseline",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA909593",BioProject), "scRNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA909593",BioProject), "none",SampleAge)) |> 
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA1163897", BioProject), "control,prrsv", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA1163897",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA1163897",BioProject), "caesalpinia_sappan",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA1163897",BioProject), "baseline&6h",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA1163897",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA1163897",BioProject), "4_weeks",SampleAge)) |> 
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA1107598", BioProject), "control,asfv", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA1107598",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA1107598",BioProject), "BA71ACD2",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA1107598",BioProject), "baseline&3&7dpc",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA1107598",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA1107598",BioProject), "27_weeks",SampleAge)) |> 
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA982050", BioProject), "gestation", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA982050",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA982050",BioProject), "enriched,concrete",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA982050",BioProject), "Gestation98,Lactation11",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA982050",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA982050",BioProject), "none",SampleAge)) |> 
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA805111", BioProject), "control,asfv", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA805111",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA805111",BioProject), "BA71ACD2",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA805111",BioProject), "21dpv_baseline&10h",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA805111",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA805111",BioProject), "none",SampleAge)) |> 
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA812000", BioProject), "control", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA812000",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA812000",BioProject), "none",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA812000",BioProject), "baseline",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA812000",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA812000",BioProject), "adult",SampleAge)) |> 
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA802582", BioProject), "control", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA802582",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA802582",BioProject), "none",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA802582",BioProject), "baseline",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA802582",BioProject), "scRNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA802582",BioProject), "7_weeks",SampleAge)) |> 
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA844377", BioProject), "control", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA844377",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA844377",BioProject), "soil",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA844377",BioProject), "baseline&11&20&56d",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA844377",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA844377",BioProject), "4_day",SampleAge)) |>                   
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA832220", BioProject), "control,bacteria", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA832220",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA832220",BioProject), "Staphylococcus_epidermidis&glucose",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA832220",BioProject), "baseline&3&6&12h",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA832220",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA832220",BioProject), "106_days_gestation_preterm",SampleAge)) |>                          
                  #dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA1200778", BioProject), "control", SampleStyle),

                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA798674", BioProject), "control", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA798674",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA798674",BioProject), "none",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA798674",BioProject), "baseline",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA798674",BioProject), "scRNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA798674",BioProject), "none",SampleAge)) |>
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA782748", BioProject), "control,bacteria", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA782748",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA782748",BioProject), "Salmonella_typhimurium&miR124",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA782748",BioProject), "baseline&12h",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA782748",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA782748",BioProject), "4_weeks",SampleAge)) |>                          
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA705952", BioProject), "control,csf", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA705952",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA705952",BioProject), "classic_swine_fever_virus",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA705952",BioProject), "baseline&7dpv",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA705952",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA705952",BioProject), "12_weeks",SampleAge)) |>                                            
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA982050", BioProject), "control,bacteria", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA982050",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA982050",BioProject), "Actinobacillus_pleuropneumoniae",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA982050",BioProject), "baseline&24&124hpi",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA982050",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA982050",BioProject), "5_weeks",SampleAge)) |>                                            
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA699465", BioProject), "control,disease", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA699465",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA699465",BioProject), "Nectrotizing_enterocolitis",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA699465",BioProject), "5days",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA699465",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA699465",BioProject), "5_days_preterm",SampleAge)) |>                                            
                
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA692626", BioProject), "control,disease", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA692626",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA692626",BioProject), "myocardial_infarction",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA692626",BioProject), "baseline&1h&3h&2d&14dpi",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA692626",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA692626",BioProject), "none",SampleAge)) |>                                            
                
                
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA689738", BioProject), "control,bacteria", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA689738",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA689738",BioProject), "Diphteria_pertussis",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA689738",BioProject), "6monthspostimmunization_baseline&16h",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA689738",BioProject), "scRNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA689738",BioProject), "6_days",SampleAge)) |>                                            
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA513475", BioProject), "control", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA513475",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA513475",BioProject), "none",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA513475",BioProject), "baseline",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA513475",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA513475",BioProject), "adult",SampleAge)) |>                                            
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA510331", BioProject), "control", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA510331",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA510331",BioProject), "none",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA510331",BioProject), "baseline&7&28&180&730d",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA510331",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA510331",BioProject), "adult",SampleAge)) |>
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA512863", BioProject), "control,disease", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA512863",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA512863",BioProject), "intracerreblar_hemorragy",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA512863",BioProject), "baseline&6h",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA512863",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA512863",BioProject), "10_weeks",SampleAge)) |>
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA484712", BioProject), "control,bacteria", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA484712",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA484712",BioProject), "Salmonella_typhimurium",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA484712",BioProject), "baseline&2&7dpi",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA484712",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA484712",BioProject), "10_weeks",SampleAge)) |>
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA479928", BioProject), "control,disease", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA479928",BioProject), "pbmc",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA479928",BioProject), "leptin_over_expression_vs_WT",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA479928",BioProject), "6month",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA479928",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA479928",BioProject), "28_weeks",SampleAge)) |>
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA381548", BioProject), "control", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA381548",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA381548",BioProject), "none",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA381548",BioProject), "baseline",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA381548",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA381548",BioProject), "8_weeks",SampleAge)) |>

                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA311061", BioProject), "control,prrsv", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA311061",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA311061",BioProject), "prrsv",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA311061",BioProject), "baseline&4&7&10&14&21&28dpi",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA311061",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA311061",BioProject), "none",SampleAge)) |>
                  
                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA313448", BioProject), "control,prrsv", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA313448",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA313448",BioProject), "prrsv",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA313448",BioProject), "baseline&4&7&10&14dpi",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA313448",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA313448",BioProject), "none",SampleAge)) |>

                  dplyr::mutate(SampleStyle = ifelse(grepl("PRJNA311061", BioProject), "control,prrsv", SampleStyle),
                                SampleTissue = ifelse(grepl("PRJNA311061",BioProject), "blood",SampleTissue),
                                SampleTreatment = ifelse(grepl("PRJNA311061",BioProject), "prrsv",SampleTreatment),
                                SampleTimepoint = ifelse(grepl("PRJNA311061",BioProject), "baseline&2&6dpi",SampleTimepoint),
                                SampleData = ifelse(grepl("PRJNA311061",BioProject), "RNA-seq",SampleData),
                                SampleAge = ifelse(grepl("PRJNA311061",BioProject), "none",SampleAge))
head(md_df)
write.table(md_df, file = paste0(output_dir, "metadata_with_sample_annotations.csv"), sep = "\t", quote = FALSE)            



# Done continue with modules generation



                  



                