# %%
.libPaths() # --> kind of already established in the R_Home
.libPaths("/data/users/mbotos/RLibs_v2/")
#.libPaths("/data/users/mbotos/RLibs_v2/")
#install.packages("remotes")
#remotes::install_github("nx10/httpgd")

# %% test
print(mtcars)

# %%
#BiocManager::install(c("edgeR","scales","ggplot2","dplyr","clusterProfiler"),force = TRUE,update = TRUE,ask = FALSE)
library("dplyr")
library("edgeR")
library("ggplot2")

# Install uwot if needed: install.packages("uwot")
# if (!requireNamespace("uwot", quietly = TRUE)) {
#   cat("Installing uwot package for UMAP...\n")
#   install.packages("uwot")
# }
#library("uwot")
library("pheatmap")


output_dir <- "/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/03_network/"
dir.create(output_dir, showWarnings = FALSE)
# %% read in the counts matrix and the md
path_to_counts <- "/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts/logCPM_matrix_filtered_samples.csv"
logcpm_counts <- read.table(path_to_counts, row.names = 1, check.names = FALSE,header = TRUE)
dim(logcpm_counts)
head(logcpm_counts)
path_to_md <- "/data/users/mbotos/Environments/2026_2_25_PIGS_BTMS+/workingEnvironment/02_counts/metadata_with_sample_annotations.csv"
md_df <- read.table(path_to_md, check.names = FALSE,header = TRUE, sep = "\t")
dim(md_df)
head(md_df)

# %% 
#BiocManager::install("minet")
library("minet")

compute_clr_network <- function(expr_mat) {
  # expr_mat: genes x samples
  message("Computing CLR network...")
  expr_mat <- t(expr_mat)  # minet expects samples x genes
  #mi <- build.mim(expr_mat, estimator = "spearman")  # or "mi.empirical"
  mi <- build.mim(expr_mat, estimator = "mi.mm",disc="equalfreq",nbins=5)  # or "mi.empirical" using mutual information!
  
  clr_net <- clr(mi)
  clr_net
}

# %% threshold the CLR network to get an adjacency matrix. This is a simple approach and can be optimized by using more sophisticated methods for thresholding or clustering.

# IF we want to run per project and select consensus modules, we can do this by splitting the counts matrix by project and applying the function to each subset. For now, we will just run it on the whole matrix.
#clr_list <- lapply(list(logcpm_counts), compute_clr_network)# this splits the counts matrix into a list of one element, which is the counts matrix, and applies the function to it.

#CLR (Context Likelihood of Relatedness)
clr_net <- compute_clr_network(logcpm_counts)
# save the CLR network for downstream analysis
#saveRDS(clr_net, paste0(output_dir,"CLR_network_spearman.rds"))
#load the CLR network for downstream analysis
#clr_net <- readRDS(paste0(output_dir,"CLR_network_spearman.rds"))
saveRDS(clr_net, paste0(output_dir,"CLR_network_mi_mm.rds"))

thr <- quantile(clr_net[upper.tri(clr_net)], 0.95)
adj <- (clr_net >= thr) * 1
diag(adj) <- 0
# %% set row and column names for the adjacency matrix
rownames(adj) <- rownames(clr_net)
colnames(adj) <- colnames(clr_net)

# cluster the network using a simple thresholding approach (e.g., top 5% of edges) and then apply a clustering algorithm like Louvain or Leiden to identify modules. For simplicity, we will just use hierarchical clustering here.
#threshold <- quantile(clr_net[upper.tri(clr_net)], 0.95) # get the 95th percentile of the edge weights
#adj_matrix <- clr_net > threshold
#dist_matrix <- as.dist(1 - adj_matrix) # convert to distance matrix
#hc <- hclust(dist_matrix, method = "average")
#modules <- cutree(hc, k = 10) # cut into 10 modules, this is arbitrary and can be optimized
#table(modules)

# %%
# save the adjacency matrix for downstream analysis, using Matrix::
library("Matrix")
#Matrix::writeMM(Matrix::Matrix(adj, sparse = TRUE), paste0(output_dir,"CLR_adjacency_matrix_spearman.mtx"))
Matrix::writeMM(Matrix::Matrix(adj, sparse = TRUE), paste0(output_dir,"CLR_adjacency_matrix_mi_mm.mtx"))

#write.table(adj, paste0(output_dir,"CLR_adjacency_matrix.tsv"), sep="\t", quote=FALSE, row.names=TRUE, col.names=TRUE)
# Convert adjacency matrix to edge list for Cytoscape
# %% Load the matrix saved in Matrix Market format
#adj <- as.matrix(Matrix::readMM(paste0(output_dir,"CLR_adjacency_matrix.mtx")))
dim(adj)
# %%
library("reshape2")
# Method 1: Binary edges only (connected/not connected)
edgelist <- melt(adj, varnames = c("source", "target"), value.name = "connected")
edgelist <- edgelist[edgelist$connected == 1, ]  # Keep only edges that exist
edgelist <- edgelist[, c("source", "target")]  # Drop the 'connected' column

# Save for Cytoscape
#write.table(edgelist, paste0(output_dir, "CLR_network_edgelist_spearman.txt"), sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)
write.table(edgelist, paste0(output_dir, "CLR_network_edgelist_mi_mm.txt"), sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)

# Method 2: Include edge weights (CLR scores)
upper_tri_mask <- upper.tri(clr_net)
above_thr <- clr_net >= thr
# Combine conditions
valid_edges <- upper_tri_mask & above_thr
# Extract only valid edges (much smaller!)
edge_indices <- which(valid_edges, arr.ind = TRUE)
edgelist_weighted <- data.frame(
  source = rownames(clr_net)[edge_indices[,1]],
  target = colnames(clr_net)[edge_indices[,2]],
  CLR_score = clr_net[valid_edges],
  stringsAsFactors = FALSE)
#write.table(edgelist_weighted, paste0(output_dir, "CLR_network_weighted_spearman.txt"), sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)
write.table(edgelist_weighted, paste0(output_dir, "CLR_network_weighted_mi_mm.txt"), sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)


# %% convert adjacency matrix to an igraph object and cluster it using Louvain
library("igraph")
g <- graph_from_adjacency_matrix(adj, mode = "undirected", diag = FALSE)
# save graph in graphml format for Cytoscape
#write_graph(g, paste0(output_dir, "CLR_network_spearman.graphml.gz"), format = "graphml")
write_graph(g, paste0(output_dir, "CLR_network_mi_mm.graphml"), format = "graphml")

# %% Cluster the graph into modules
#cl <- igraph::cluster_louvain(g)
cl <- cluster_louvain(g)
modules <- split(V(g)$name, membership(cl))
#modules <- split(names(membership(cl)), membership(cl))

# Create node table with module membership
node_attr <- data.frame(gene = names(membership(cl)),
                        module = membership(cl),
                        stringsAsFactors = FALSE)
#write.table(node_attr, paste0(output_dir, "node_modules_spearman.txt"),sep="\t", quote=FALSE, row.names=FALSE)
write.table(table(node_attr$module), paste0(output_dir, "node_modules_mi_mm.txt"),sep="\t", quote=FALSE, row.names=FALSE)

# %% alternative use -kNN into 10-30 neighbors and then cluster with Leiden, this is more robust to noise and can capture more complex structures in the data. We can use the knn function from the FNN package to compute the k-nearest neighbors and then create a graph based on these neighbors before applying the Leiden algorithm for clustering.
#BiocManager::install("FNN")
library("FNN")
# Compute kNN graph on GENES (rows = genes, columns = samples)
# Do NOT transpose: knn.index expects observations x features,
# so we pass logcpm_counts directly (genes x samples → each gene is an observation)
k <- 20  # number of neighbors, can be optimized
knn_result <- knn.index(logcpm_counts, k = k)  # genes x samples, no transpose
# Create adjacency matrix from kNN result (gene x gene)
n_genes <- nrow(logcpm_counts)
knn_adj <- matrix(0, nrow = n_genes, ncol = n_genes)
for (i in 1:n_genes) {
  knn_adj[i, knn_result[i, ]] <- 1
}
knn_adj <- knn_adj + t(knn_adj)  # make symmetric
knn_adj[knn_adj > 1] <- 1  # ensure binary
diag(knn_adj) <- 0  # remove self-loops
# Create graph from kNN adjacency matrix (one vertex per gene)
g_knn <- graph_from_adjacency_matrix(knn_adj, mode = "undirected", diag = FALSE)
V(g_knn)$name <- rownames(logcpm_counts)  # gene IDs as vertex names
# Cluster with Leiden
#cl_leiden <- igraph::cluster_leiden(g_knn)
cl_louvain <- igraph::cluster_louvain(g_knn)
modules_louvain <- split(V(g_knn)$name, membership(cl_louvain)) 
# %% revise from here...






# %% check the size of the modules
module_sizes <- sapply(modules, length)
large_modules <- names(module_sizes[module_sizes > 100])

head(sort(module_sizes, decreasing = TRUE))
length(modules[[which.max(module_sizes)]])

# %% annoate the modules using clusterProfiler and org.Ss.eg.db for pig annotation. This will give us the biological functions associated with each module, which can help us interpret the results and identify key pathways or processes that are relevant to our study. We will use the enrichGO function to perform GO enrichment analysis for each module, and we will adjust the p-values using the Benjamini-Hochberg method to control for false discovery rate.
library("clusterProfiler")
#BiocManager::install(c("clusterProfiler","org.Ss.eg.db"))  # pig annotation
library("org.Ss.eg.db")
#keytypes(org.Ss.eg.db)
#does not support ensembl..
# modules_entrez <- lapply(modules, function(genes) {
#   bitr(genes, fromType = "ENSEMBL",
#               toType = "ENTREZID",
#               OrgDb = org.Ss.eg.db)
# })
#use biomart to convert ensembl to entrez
library("biomaRt")
all_ens <- unique(unlist(modules))
mart <- useMart("ensembl", dataset = "sscrofa_gene_ensembl")
ens2entrez <- getBM(
  attributes = c("ensembl_gene_id", "entrezgene_id"),
  filters = "ensembl_gene_id",
  values = all_ens,
  mart = mart)
modules_entrez <- lapply(modules, function(genes) {
  subset(ens2entrez, ensembl_gene_id %in% genes)})

# mart <- useMart("ensembl", dataset = "sscrofa_gene_ensembl")

# modules_entrez <- lapply(modules, function(genes) {
#   getBM(
#     attributes = c("ensembl_gene_id", "entrezgene_id"),
#     filters = "ensembl_gene_id",
#     values = genes,
#     mart = mart)})
# check na coverage per module
sapply(modules_entrez, function(df) sum(!is.na(df$entrezgene_id)))

modules_entrez_clean <- lapply(modules_entrez, function(df) {
  df[!is.na(df$entrezgene_id), ]})

# check for those modules that have at least 5 genes with valid entrez ids, as enrichGO requires at least 5 genes to perform enrichment analysis. We will filter out modules that do not meet this criterion before running enrichGO.
#sapply(modules_entrez_clean, nrow)
#annot_list <- lapply(modules_entrez_clean, function(df) {
annot_list <- lapply(modules_entrez_clean, function(df) {
  if (nrow(df) >= 5) {
    enrichGO(
      gene          = df$entrezgene_id,
      OrgDb         = org.Ss.eg.db,
      keyType       = "ENTREZID",
      ont           = "BP",
      pAdjustMethod = "BH",
      readable      = TRUE,
      qvalueCutoff  = 0.05
    )
  } else {
    NULL
  }
})


# %% Use gprofiler directly works on ensembl?library(gprofiler2)
library("gprofiler2")
#BiocManager::install("gprofiler2")
annot_list_ens <- lapply(modules, function(genes) {
  gost(query = genes,
       organism = "sscrofa",
       correction_method = "fdr",
       significant = TRUE,
       sources = c("GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"))})



# %% Export the modules to a file for downstream analysis..
btm_df <- stack(modules)
colnames(btm_df) <- c("Gene", "Module")
#write.table(btm_df, paste0(output_dir, "BTM_modules_spearman.tsv"), sep="\t", quote=FALSE, row.names=FALSE)
write.table(btm_df, paste0(output_dir, "BTM_modules_mi_mm.tsv"), sep="\t", quote=FALSE, row.names=FALSE)


# %% Prepare the largest modules for subclustering.
# %% check the size of the modules
module_sizes <- sapply(modules, length)
large_modules <- names(module_sizes[module_sizes > 100])

# %% sub cluster the large modules to get more refined modules. 
#names(large_modules) <- large_modules
# subgraphs <- lapply(large_modules, function(m) {
#   induced_subgraph(g, vids = modules[[m]])
# })
subgraphs <- list()
for (m in large_modules) {
  subgraphs[[paste0("M.",m)]] <- igraph::induced_subgraph(g, vids = modules[[m]])}
names(subgraphs)

# run later, this takes a while...
# for (m in names(subgraphs)) {
#   pdf(paste0(output_dir, "module_", m, ".pdf"), width = 10, height = 10)
#   sg <- subgraphs[[m]]
#   plot(sg, vertex.size = 3, vertex.label = NA)
#   dev.off()
# }



# %% Recluster the subgraphs using Leiden algorithm.
library("leiden")
# submodules <- lapply(subgraphs, function(sg) {
#   cl <- leiden(sg)
#   split(V(sg)$name, cl)
# })
submodules <- list()
for (m in names(subgraphs)) {
  sg <- subgraphs[[m]]
  # cl <- igraph::cluster_leiden(sg)
  cl <- igraph::cluster_louvain(sg) 
  sm <- split(V(sg)$name, cl$membership)
  # submodules[[paste0(m, ".", sm)]] <- sm
  submodules[[m]] <- sm
}
names(submodules)

# %%
submodule_counts <- sapply(submodules[["M.5"]], length)
submodule_counts
#length(submodules$M.1[[1]])
#length(submodules$M.1[[2]])

# %% Save the Modules with their respective submodules for downstream analysis. This will allow us to analyze the modules at a finer resolution and identify more specific biological functions or pathways that are associated with each submodule. We can save the submodules in a similar format as the original modules, with a gene-to-submodule mapping that can be used for enrichment analysis or other downstream analyses.
# separate clearly Module 1 with submodules M.1.1, M.1.2, etc. and so on..

submodule_df <- do.call(rbind, lapply(names(submodules), function(m) {
  sm_list <- submodules[[m]]
  do.call(rbind, lapply(names(sm_list), function(sm) {
    data.frame(Gene = sm_list[[sm]], Submodule = paste0(m, ".", sm), stringsAsFactors = FALSE)
  }))
}))

head(submodule_df)
# %% Check length of submodules
table(submodule_df$Submodule)

# %%
submodule_df  |> dplyr::filter(Submodule == "M.1.1")

#colnames(submodule_df) <- c("Gene", "Submodule")
write.table(submodule_df,
            paste0(output_dir, "submodules_spearman_louvain_leiden.tsv"),
            sep="\t", quote=FALSE, row.names=FALSE)

# %% Save the submodules plot in a pdf for each Module.
for (m in names(submodules)) {
  list_of_modules <- submodules[[m]]
  for (subm in names(list_of_modules)) {
    submodule_genes <- list_of_modules[[subm]]
    submodule_sg <- induced_subgraph(g, vids = submodule_genes)
    pdf(paste0(output_dir, "submodule_", m, "_", subm, ".pdf"), width = 10, height = 10)
    plot(submodule_sg, vertex.size = 3, vertex.label = NA)
    dev.off()
}}


# for (m in names(submodules)) {
#   pdf(paste0(output_dir, "submodule_", m, ".pdf"), width = 10, height = 10)
#   sg <- subgraphs[[m]]
#   plot(sg, vertex.size = 3, vertex.label = NA)
#   dev.off()
# }
# %% Save the submodules in a graphml format for Cytoscape.
for (m in names(submodules)) {
  list_of_modules <- submodules[[m]]
  for (subm in names(list_of_modules)) {
    submodule_genes <- list_of_modules[[subm]]
    submodule_sg <- induced_subgraph(g, vids = submodule_genes)
    write_graph(submodule_sg, paste0(output_dir, "submodule_", m, "_", subm, ".graphml"), format = "graphml")
}}  

# %% Convert submodule genes to symbols for gprofiler2 enrichment
# This allows us to see which specific genes map to each enriched term
library("gprofiler2")
library("biomaRt")

# Collect all unique genes across submodules
all_submodule_genes <- unique(unlist(submodules, recursive = TRUE))
message("Converting ", length(all_submodule_genes), " genes to symbols...")

# biomaRt mapping
mart <- useMart("ensembl", dataset = "sscrofa_gene_ensembl")
gene_mapping_raw <- getBM(
  attributes = c("ensembl_gene_id", "external_gene_name", "entrezgene_id"),
  filters = "ensembl_gene_id",
  values = all_submodule_genes,
  mart = mart
)

# Clean mapping
gene_mapping <- gene_mapping_raw[
  gene_mapping_raw$external_gene_name != "" &
  !is.na(gene_mapping_raw$external_gene_name), ]

gene_mapping <- gene_mapping[!duplicated(gene_mapping$ensembl_gene_id), ]

message("Mapped ", nrow(gene_mapping), " genes (",
        round(nrow(gene_mapping)/length(all_submodule_genes)*100, 1), "% success)")


# Save the mapping for reference
write.table(gene_mapping, 
            paste0(output_dir, "gene_id_mapping.tsv"),
            sep="\t", quote=FALSE, row.names=FALSE)

# %% Enrich submodules using gprofiler2 with gene symbols
# Using symbols gives better results and allows gene-level resolution
submodule_annot <- lapply(names(submodules), function(m) {
  list_of_modules <- submodules[[m]]
  
  lapply(names(list_of_modules), function(subm) {
    submodule_genes_ens <- list_of_modules[[subm]]
    
    # Convert ENSEMBL → SYMBOL
    submodule_genes_symbols <- gene_mapping$external_gene_name[
      match(submodule_genes_ens, gene_mapping$ensembl_gene_id)
    ]
    submodule_genes_symbols <- submodule_genes_symbols[!is.na(submodule_genes_symbols)]
    
    if (length(submodule_genes_symbols) < 5) {
      message("Skipping ", m, ".", subm, ": only ", length(submodule_genes_symbols), " mapped genes")
      return(NULL)
    }
    
    message("Enriching ", m, ".", subm, ": ", length(submodule_genes_symbols), " genes")
    
    gost_result <- gost(
      query = submodule_genes_symbols,
      organism = "sscrofa",
      correction_method = "fdr",
      significant = TRUE,
      sources = c("GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC")
    )
    Sys.sleep(1)  # avoid API throttling
    
    if (is.null(gost_result) || is.null(gost_result$result)) { return(NULL) } 
    # Add parsed intersection genes safely 
    if ("intersection" %in% colnames(gost_result$result)) {
      gost_result$result$intersection_genes <- strsplit( as.character(gost_result$result$intersection), "," ) 
    } else {
        gost_result$result$intersection_genes <- vector("list", nrow(gost_result$result))
    } # Add mapping table for traceability 
    gost_result$gene_mapping <- data.frame( module = paste0(m, ".", subm),
        ensembl_gene_id = submodule_genes_ens,
        gene_symbol = gene_mapping$external_gene_name[
          match(submodule_genes_ens, gene_mapping$ensembl_gene_id) ],
        stringsAsFactors = FALSE)
    return(gost_result)
  })
})


# %%
submodule_annot[[1]][[2]]$result
colnames(submodule_annot[[1]][[1]]$result)
head(submodule_annot[[1]][[1]]$result)

# %% Example: View enrichment results with gene-level detail
# Show first submodule's top enriched terms
if (!is.null(submodule_annot[[1]][[1]])) {
  cat("\n========== Example Enrichment Results ==========\n")
  cat("Submodule: M.1.1\n")
  cat("Number of mapped genes:", nrow(submodule_annot[[1]][[1]]$gene_mapping), "\n")
  cat("\nTop enriched terms:\n")
  print(head(submodule_annot[[1]][[1]]$result[, c("term_name", "p_value", 
                                                     "term_size", "intersection_size")], 10))
  
  cat("\nGenes in this submodule:\n")
  print(head(submodule_annot[[1]][[1]]$gene_mapping, 10))
}

# %% Save enrichment results to files
for (i in seq_along(names(submodules))) {
  m <- names(submodules)[i]
  list_of_modules <- submodules[[m]]
  
  for (j in seq_along(names(list_of_modules))) {
    subm <- names(list_of_modules)[j]
    annot <- submodule_annot[[i]][[j]]
    
    if (!is.null(annot) && !is.null(annot$result) && nrow(annot$result) > 0) {
      # Save enrichment results
      outfile <- paste0(output_dir, "enrichment_", m, "_", subm, ".tsv")
      write.table(annot$result, outfile, sep="\t", quote=FALSE, row.names=FALSE)
      
      # Save gene mapping for this submodule
      gene_file <- paste0(output_dir, "genes_", m, "_", subm, ".tsv")
      write.table(annot$gene_mapping, gene_file, sep="\t", quote=FALSE, row.names=FALSE)
    }
  }
}

cat("\nEnrichment analysis complete! Files saved to:", output_dir, "\n")

# %% Helper function: Get genes for a specific enriched term
# This answers "which genes in my module are in pathway X?"
get_genes_in_term <- function(submodule_idx, submodule_subidx, term_id, 
                                annot_list = submodule_annot, 
                                mapping = gene_mapping) {
  # Get the enrichment result for this submodule
  result <- annot_list[[submodule_idx]][[submodule_subidx]]
  
  if (is.null(result) || is.null(result$result)) {
    cat("No enrichment results for this submodule\n")
    return(NULL)
  }
  
  # Find the term
  term_row <- result$result[result$result$term_id == term_id, ]
  
  if (nrow(term_row) == 0) {
    cat("Term ID not found in results\n")
    return(NULL)
  }
  
  # Get the genes that were tested
  tested_genes_symbols <- result$gene_mapping$gene_symbol
  tested_genes_ensembl <- result$gene_mapping$ensembl_gene_id
  
  # Use gprofiler2 to get genes in this specific term
  # Query the same genes again but retrieve the intersections
  term_genes_result <- gost(
    query = tested_genes_symbols,
    organism = "sscrofa",
    sources = strsplit(term_id, ":")[[1]][1],  # Get source (GO, KEGG, etc.)
    evcodes = TRUE  # This returns gene evidence codes
  )
  
  # Find the matching term and extract genes
  if (!is.null(term_genes_result) && term_id %in% term_genes_result$result$term_id) {
    # The genes are in the evcodes if requested, but easier is to match back
    cat("Term:", term_row$term_name, "\n")
    cat("Term ID:", term_id, "\n")
    cat("P-value:", format(term_row$p_value, digits=3), "\n")
    cat("Genes in term:", term_row$intersection_size, "/", term_row$term_size, "\n\n")
    
    # Create a results dataframe
    genes_df <- result$gene_mapping
    genes_df$in_pathway <- "Check with publish_gosttable or publish_gostplot"
    
    return(genes_df)
  } else {
    cat("Could not retrieve detailed gene mapping\n")
    return(result$gene_mapping)
  }
}

# %% Example usage: View genes in a specific enriched term
# Example: Get genes from first submodule's top enriched pathway
if (!is.null(submodule_annot[[1]][[1]]) && nrow(submodule_annot[[1]][[1]]$result) > 0) {
  cat("\n========== Example: Genes in Top Enriched Term ==========\n")
  top_term_id <- submodule_annot[[1]][[1]]$result$term_id[1]
  top_term_name <- submodule_annot[[1]][[1]]$result$term_name[1]
  
  cat("Extracting genes for:", top_term_name, "\n")
  cat("Term ID:", top_term_id, "\n\n")
  
  genes_in_top_term <- get_genes_in_term(1, 1, top_term_id)
  cat("\nAll genes in this submodule:\n")
  print(head(genes_in_top_term, 20))
}

# %% Better approach: Use gprofiler2's built-in gene retrieval
# Export detailed results with gene lists using publish_gosttable
if (!is.null(submodule_annot[[1]][[1]])) {
  cat("\n========== Exporting Detailed Enrichment Tables ==========\n")
  
  # For each submodule, create detailed enrichment table
  for (i in seq_along(names(submodules))) {
    m <- names(submodules)[i]
    list_of_modules <- submodules[[m]]
    
    for (j in seq_along(names(list_of_modules))) {
      subm <- names(list_of_modules)[j]
      annot <- submodule_annot[[i]][[j]]
      
      if (!is.null(annot) && !is.null(annot$result) && nrow(annot$result) > 0) {
        # Create detailed table
        detailed_file <- paste0(output_dir, "enrichment_detailed_", m, "_", subm, ".tsv")
        
        tryCatch({
          # Get genes that were queried
          genes_tested <- annot$gene_mapping$gene_symbol
          genes_tested <- genes_tested[!is.na(genes_tested)]
          
          # Add column showing which genes were tested
          annot$result$genes_tested <- paste(genes_tested, collapse = ";")
          annot$result$n_genes_tested <- length(genes_tested)
          
          write.table(annot$result, detailed_file, sep="\t", quote=FALSE, row.names=FALSE)
          
          cat("Saved:", detailed_file, "\n")
        }, error = function(e) {
          cat("Error saving detailed results for", m, subm, ":", e$message, "\n")
        })
      }
    }
  }
}

