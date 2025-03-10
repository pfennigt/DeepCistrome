# This tool tests the enrichment of IPMs to annotated genome features
# SMZ 2024-09-01
# Load the necessary packages
library(dplyr)
library(tidyverse)
library(tidyr)
library(stringr)
library(rtracklayer)
library(GenomicRanges)
library(data.table)
library(ggrepel)
library(igraph)
library(pheatmap)
library(ggraph)

########################################
output_file_path <- "IPM_TFBS_summary_statistics.csv"
# Set the working directory to the location of the BED file
setwd("/home/ibg-4/Desktop/deepCIS_calc/")
# Define the path to your CSV file
file_path <- "./studies/deepCIS/ipmArthDAPmotifs/extracted_sites_predictions_chrom_1.csv"
# Define the file that contains similar ids
file_path11 <- "./studies/deepCIS/ipmArthDAPmotifs/extracted_ranges_1.txt"
################################################################
file_path22 <- "./studies/deepCIS/ipmArthDAPmotifs/predicted_percentages_per_family.csv"
# Import the CSV file with accuracies
predicted_percentages <- read.csv(file_path22)

head(predicted_percentages)
################################################################
file_path33 <- "./studies/deepCIS/ipmArthDAPmotifs/all_binding_sites.csv"
# Import the CSV file with all the bed-files TRUE TARGETS
true_binding_sites <- read.csv(file_path33)
################################################################

# View the first few rows of the updated dataframe
head(true_binding_sites)
true_binding_sites <- true_binding_sites %>%
  mutate(
    center = (end + start) / 2 + 0.5,
    wstart = center - 125,
    wend = center + 125,
    wlength = wend - wstart,
    seq_id = paste0(chr, ":", wstart, "-", wend)
  )

# Define the chunk size (number of rows to read at a time)
chunk_size <- 100000  # Adjust this number based on your memory capacity
# Count the total number of rows in the file
total_rows <- fread(file_path, select = 1)[, .N]
# Initialize an empty list to store chunks
chunks <- list()
chunk_number <- 1
skip <- 0
while (skip < total_rows) {
  # Calculate the number of rows to read
  rows_left <- total_rows - skip
  nrows_to_read <- min(chunk_size, rows_left)
  # Read a chunk of the file
  chunk <- fread(file_path, nrows = nrows_to_read, skip = skip, header = (chunk_number == 1))
  # Store the chunk in the list
  chunks[[chunk_number]] <- chunk
  # Increment the chunk number and skip value
  chunk_number <- chunk_number + 1
  skip <- skip + nrows_to_read
}
# Combine all chunks into one data.table (optional)
extracted_sites <- rbindlist(chunks)
# Display the first few rows of the combined data
head(extracted_sites)

# Process extracted_sites
extracted_sites <- extracted_sites %>%
  mutate(
    start = as.numeric(start),
    end = as.numeric(end) + 1,
    seq_id = paste(chr, paste(start, end, sep = "-"), sep = ":")
  )

# Extract IDs
comm_ids_only <- common_ids$seq_id
extract_ids_only <- extracted_sites$seq_id

# Find common and false IDs
true_ids <- intersect(extract_ids_only, comm_ids_only)
false_ids0 <- setdiff(extract_ids_only, comm_ids_only)
false_ids1 <- setdiff(comm_ids_only, extract_ids_only)
false_ids01 <- union(false_ids1, false_ids0)

# Filter data
extracted_sites_filtered <- extracted_sites %>% filter(seq_id %in% true_ids)
common_ids_filtered <- common_ids %>% filter(seq_id %in% true_ids)

# Create a dataframe to store results
res_gen_wide_binding_stats <- data.frame(
  column_name = character(),
  total_extracted_sites = numeric(),
  total_common_ids = numeric(),
  predicted_binding = numeric(),
  motif_binding_context_value = numeric(),
  motif_redundancy_value = numeric(),
  true_binding_count = numeric(),
  true_per_total_extracted = numeric(),
  TDR_IPM = numeric(),
  FPFN_IPM_plus_binding_prediction = numeric(),
  stringsAsFactors = FALSE
)

# Loop over each column except 'seq_id'
for (col_name in colnames(extracted_sites_filtered)) {
  if (col_name != "seq_id") {
    # Filter matching_common_ids for the current factor
    matching_common_ids <- common_ids_filtered %>%
      filter(seq_id %in% extracted_sites_filtered$seq_id &
               grepl(paste0("*", gsub("_tnt", "", col_name), "*"), label))
    
    # Filter the relevant column from extracted_sites_filtered
    extracted_sites_filtered_col <- extracted_sites_filtered %>%
      select(seq_id, all_of(col_name)) %>%
      distinct()
    
    # Perform semi-join to filter matching seq_ids
    extracted_sites_doublefiltered <- extracted_sites_filtered_col %>%
      semi_join(matching_common_ids, by = "seq_id")
    
    # Calculate the total number of extracted sites
    total_extracted_sites <- nrow(extracted_sites_doublefiltered)
    
    # Calculate the total number of common ids
    total_common_ids <- nrow(matching_common_ids)
    
    # Count values greater than 0.5 in the current column
    predicted_binding <- sum(extracted_sites_doublefiltered[[col_name]] > 0.5)
    
    # Calculate the motif binding context value
    motif_binding_context_value <- 1 - (predicted_binding / total_extracted_sites)
    
    # Calculate the ratio of total_common_ids to total_extracted_sites
    motif_redundancy_value <- ifelse(total_extracted_sites > 0, total_common_ids / total_extracted_sites, NA)
    
    # Calculate true_binding_count
    true_binding_count <- sum(
      unique(extracted_sites_filtered$seq_id[extracted_sites_filtered[[col_name]] > 0.5]) %in% true_binding_sites$seq_id
    )
    
    # Calculate true_per_total_extracted
    true_per_total_extracted <- sum(extracted_sites_doublefiltered$seq_id %in% true_binding_sites$seq_id)
    
    TDR_IPM <- true_per_total_extracted / total_extracted_sites
    TDR_IPM_plus_binding_prediction <- true_binding_count / predicted_binding
    
    # Add the results to the dataframe
    res_gen_wide_binding_stats <- rbind(res_gen_wide_binding_stats, data.frame(
      column_name = col_name,
      total_extracted_sites = total_extracted_sites,
      total_common_ids = total_common_ids,
      predicted_binding = predicted_binding,
      motif_binding_context_value = motif_binding_context_value,
      motif_redundancy_value = motif_redundancy_value,
      true_binding_count = true_binding_count,
      true_per_total_extracted = true_per_total_extracted,
      TDR_IPM = TDR_IPM,
      FPFN_IPM_plus_binding_prediction = TDR_IPM_plus_binding_prediction
    ))
  }
}

# Print the results
head(res_gen_wide_binding_stats)

# Rename column in predicted_percentages and merge with results
names(predicted_percentages)[1] <- "column_name"
merged_stats <- merge(res_gen_wide_binding_stats, predicted_percentages, by = "column_name", all.x = TRUE)

# Remove rows with NA values in specific columns
merged_stats_cleaned <- merged_stats %>%
  filter(!is.na(motif_binding_context_value) & !is.na(Predicted.Percentage))

# Fit a polynomial model (quadratic)
fit_poly <- lm(motif_binding_context_value ~ poly(Predicted.Percentage, 2), data = merged_stats_cleaned)

# Get the R-squared value
r_squared <- summary(fit_poly)$r.squared
print(paste("R-squared:", r_squared))

# Generate predictions using the model
merged_stats_cleaned$predicted_values <- predict(fit_poly, newdata = data.frame(Predicted.Percentage = merged_stats_cleaned$Predicted.Percentage))

# Calculate the residuals
merged_stats_cleaned$residuals <- abs(merged_stats_cleaned$motif_binding_context_value - merged_stats_cleaned$predicted_values)

# Define outliers
threshold <- mean(merged_stats_cleaned$residuals) + 0.5 * sd(merged_stats_cleaned$residuals)
merged_stats_cleaned$outlier <- merged_stats_cleaned$residuals > threshold

# Create the ggplot
p <- ggplot(merged_stats_cleaned, aes(x = Predicted.Percentage, y = (1 / motif_binding_context_value) - 1)) +
  geom_point(aes(color = outlier, alpha = ifelse(Predicted.Percentage < 0.6, 0.4, 1)), size = 3) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = FALSE, color = "darkgrey") +
  geom_text_repel(aes(label = str_remove(column_name, "_tnt$"), alpha = ifelse(Predicted.Percentage < 0.6, 0.4, 1)),
                  size = 4, box.padding = 0.6, point.padding = 0.3, max.overlaps = 15) +
  scale_alpha_identity() +
  scale_color_manual(values = c("black", "orange"), guide = FALSE) +
  labs(title = "", x = "Sensitivity", y = "Predictability (1/IPMciv)-1") +
  annotate("text", x = min(merged_stats_cleaned$Predicted.Percentage), y = max((1 / merged_stats_cleaned$motif_binding_context_value) - 1),
           label = paste("R-squared:", round(r_squared, 3)), color = "darkgrey", hjust = 0) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    axis.line = element_line(size = 0.8),
    axis.ticks = element_line(size = 0.8),
    plot.title = element_text(size = 16, face = "bold")
  )

# Print the plot in a new window
print(p)

# Save the plot as PNG
ggsave(filename = "./motif_binding_context_vs_predicted_percentage.png", plot = p, width = 14, height = 6, dpi = 900)

# Filter seq_ids for values greater than 0.5 per column
seq_id_lists <- lapply(extracted_sites_filtered %>% select(-seq_id), function(col) {
  extracted_sites_filtered$seq_id[which(col > 0.5)]
})

# Find overlapping seq_ids between columns
overlap_count <- matrix(0, ncol = length(seq_id_lists), nrow = length(seq_id_lists),
                        dimnames = list(names(seq_id_lists), names(seq_id_lists)))

for (i in 1:length(seq_id_lists)) {
  for (j in 1:length(seq_id_lists)) {
    if (i != j) {
      overlap_count[i, j] <- length(intersect(seq_id_lists[[i]], seq_id_lists[[j]]))
    }
  }
}

# Display the overlap count matrix
print(overlap_count)
shared_seq_id <- as.data.frame(overlap_count)

# Z-Score Normalization function
z_score_normalize <- function(x) {
  return((x - mean(x)) / sd(x))
}

# Apply Z-Score normalization
shared_seq_id_z_normalized <- shared_seq_id %>%
  mutate(across(everything(), z_score_normalize))

# Filter rows where at least one value is above 1
filtered_shared_seq_id <- shared_seq_id_z_normalized %>%
  filter(apply(., 1, function(x) any(x > 1)))

# Modify column names to remove "_tnt"
colnames(filtered_shared_seq_id) <- str_remove(colnames(filtered_shared_seq_id), "_tnt$")
row.names(filtered_shared_seq_id) <- str_remove(row.names(filtered_shared_seq_id), "_tnt$")

# Convert to matrix for clustering
shared_seq_matrix <- as.matrix(filtered_shared_seq_id)

# Calculate distance and perform hierarchical clustering
dist_matrix <- dist(shared_seq_matrix, method = "euclidean")
hc_rows <- hclust(dist_matrix, method = "complete")
hc_cols <- hclust(dist(t(shared_seq_matrix)), method = "complete")

# Generate the heatmap
h <- pheatmap(shared_seq_matrix,
              cluster_rows = hc_rows,
              cluster_cols = hc_cols,
              display_numbers = TRUE,
              fontsize_number = 8,
              main = "",
              fontsize = 10,
              color = colorRampPalette(c("white", "darkorange"))(100))

print(h)

# Save the heatmap to a file
ggsave(filename = "./low_motif_binding_outliers_heatmap.png", plot = h, width = 6, height = 6, dpi = 900)

# Define a safe Z-score normalization function
z_score_normalize_safe <- function(x) {
  if (sd(x, na.rm = TRUE) == 0) {
    return(rep(0, length(x)))
  } else {
    return((x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
  }
}

# Apply Z-Score normalization
shared_ALL_seq_id_id_z_normalized <- shared_ALL_seq_id %>%
  mutate(across(everything(), z_score_normalize_safe))

# Replace values greater than 1 with the original value, others with NA
filtered_data <- shared_ALL_seq_id_id_z_normalized %>%
  mutate(across(everything(), ~ ifelse(. > 1, ., NA)))

# Flatten the data for edge creation
edges <- filtered_data %>%
  as.data.frame() %>%
  rownames_to_column(var = "from") %>%
  pivot_longer(cols = -from, names_to = "to", values_to = "weight") %>%
  filter(!is.na(weight))

# Create a list of all unique nodes (TFs) from the edges and all available TFs
all_nodes <- unique(c(edges$from, edges$to))

# Create the graph object, ensuring all nodes are included, even those without edges
graph <- graph_from_data_frame(edges, vertices = data.frame(name = all_nodes))

# Check if all nodes are present in shared_seq_id_with_percentage
missing_nodes <- setdiff(V(graph)$name, shared_seq_id_with_percentage$TF)
if (length(missing_nodes) > 0) {
  message("Missing nodes in predicted percentages:", paste(missing_nodes, collapse = ", "))
}

# Merge the predicted percentage data into the graph's vertices
V(graph)$Predicted_Percentage <- shared_seq_id_with_percentage$Predicted.Percentage[match(V(graph)$name, shared_seq_id_with_percentage$TF)]

# Check the length of V(graph)$Predicted_Percentage
if (length(V(graph)$Predicted_Percentage) != length(V(graph))) {
  stop("Mismatch in length of Predicted_Percentage and number of vertices in the graph.")
}

# Map Predicted_Percentage to node color using a gradient
V(graph)$node_color <- scales::col_numeric(palette = c("red", "blue"), domain = c(0, 1))(V(graph)$Predicted_Percentage)

# Remove the "_tnt" suffix from the labels
V(graph)$label <- str_remove(V(graph)$name, "_tnt$")

# Compute degree (number of connections) for each node
degree_values <- degree(graph)

# Set the node size based on degree
node_size <- scales::rescale(degree_values, to = c(2, 10))

# Plot the graph with the new coloring logic for nodes and light grey edges
N <- ggraph(graph, layout = "fr") +
  geom_edge_link(aes(width = weight), color = "darkgrey", alpha = 0.3) +
  geom_node_point(aes(color = Predicted_Percentage, size = node_size)) +
  geom_node_text(aes(label = label), repel = TRUE, size = 4, box.padding = 0.4) +
  theme_void() +
  scale_color_gradient(low = "red", high = "blue", name = "Sensitivity") +
  scale_size_continuous(name = "Centrality", range = c(2, 10)) +
  ggtitle("") +
  guides(color = guide_colorbar(title = "Sensitivity", barwidth = 1, barheight = 6),
         size = guide_legend(title = "Centrality", override.aes = list(color = "black")))

# Print the plot
print(N)

# Save the plot to a file
ggsave(filename = "Filtered_Zscore_Network_Optimized_with_legend.png", plot = N, width = 7, height = 9, dpi = 900)

# Create a gradient from red to blue with purple in the middle
color_palette <- colorRampPalette(c("red", "purple", "blue"))

# Generate the gradient image
gradient <- matrix(seq(0, 85, length.out = 256), nrow = 1)

# Plot the gradient
image(gradient, col = color_palette(256), axes = FALSE, xlab = "", ylab = "")
axis(1, at = c(0, 1), labels = c("0", "85"))

# Define central nodes and trim "_tnt"
central_node <- c("BZR_tnt", "BES1_tnt", "bHLH_tnt", "bZIP_tnt", "SBP_tnt") %>%
  str_remove("_tnt$")

# Create edges connecting only to central nodes with Z > 1 and assign edge colors
edges <- bind_rows(
  data.frame(
    from = "BZR",
    to = node_names[!is.na(filtered_data$BZR_tnt)],
    weight = filtered_data$BZR_tnt[!is.na(filtered_data$BZR_tnt)]
  ),
  data.frame(
    from = "BES1",
    to = node_names[!is.na(filtered_data$BES1_tnt)],
    weight = filtered_data$BES1_tnt[!is.na(filtered_data$BES1_tnt)]
  ),
  data.frame(
    from = "bHLH",
    to = node_names[!is.na(filtered_data$bHLH_tnt)],
    weight = filtered_data$bHLH_tnt[!is.na(filtered_data$bHLH_tnt)]
  ),
  data.frame(
    from = "bZIP",
    to = node_names[!is.na(filtered_data$bZIP_tnt)],
    weight = filtered_data$bZIP_tnt[!is.na(filtered_data$bZIP_tnt)]
  ),
  data.frame(
    from = "SBP",
    to = node_names[!is.na(filtered_data$SBP_tnt)],
    weight = filtered_data$SBP_tnt[!is.na(filtered_data$SBP_tnt)]
  )
)

# Assign colors to edges based on connection type
edges <- edges %>%
  mutate(
    color = case_when(
      from == "BZR" | to == "BZR" ~ "orange",
      from == "bHLH" | to == "bHLH" ~ "aquamarine",
      from == "BES1" | to == "BES1" ~ "plum",
      TRUE ~ "grey"
    )
  )

# Ensure node names are unique and valid
all_node_names <- unique(c(central_node, edges$to))

# Create the graph object
graph <- graph_from_data_frame(edges, vertices = data.frame(name = all_node_names), directed = FALSE)

# Plot the graph
N <- ggraph(graph, layout = "fr") +
  geom_edge_link(aes(width = weight, color = color), alpha = 0.6) +
  geom_node_point(size = 5, aes(color = case_when(
    name == "bHLH" ~ "aquamarine3",
    name == "BES1" ~ "plum3",
    name %in% c("SBP", "BZR") ~ "darkorange",
    TRUE ~ "black"
  ))) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3) +
  scale_edge_color_manual(values = c("orange" = "orange", "aquamarine" = "aquamarine", "plum" = "plum", "grey" = "grey")) +
  scale_color_identity() +
  theme_void() +
  theme(legend.title = element_blank()) +
  ggtitle("")

print(N)

# Save the plot
ggsave(filename = "G-Box_TF_Network_Relevant_Nodes_Full_Network.png", plot = N, width = 4, height = 3, dpi = 900)

# Filter data for Z-scores > 1
filtered_data <- shared_ALL_seq_id_id_z_normalized %>%
  mutate(across(everything(), ~ ifelse(. > 1, ., NA)))

# Ensure no values <= 1 remain
print(filtered_data)

# Flatten into a long-format table for edges
edges <- filtered_data %>%
  as.data.frame() %>%
  rownames_to_column(var = "from") %>%
  pivot_longer(cols = -from, names_to = "to", values_to = "weight") %>%
  filter(!is.na(weight))

# Check for invalid edges
invalid_edges <- edges %>% filter(from == "HB_tnt" & to == "CPP")
print(invalid_edges)

# Clear old graph objects if necessary
rm(graph)

# Modify edge and node labels by removing the "_tnt" suffix
edges <- edges %>%
  mutate(
    from = str_remove(from, "_tnt$"),
    to = str_remove(to, "_tnt$")
  )

# Update node names in the graph object to remove "_tnt"
all_node_names <- unique(c(edges$from, edges$to))
graph <- graph_from_data_frame(
  d = edges,
  vertices = data.frame(name = str_remove(all_node_names, "_tnt$")),
  directed = FALSE
)

# Assign colors to edges
relevant_nodes <- c("BZR", "SBP", "HB")
edges <- edges %>%
  mutate(
    color = ifelse(from %in% relevant_nodes | to %in% relevant_nodes, "orange", "grey")
  )

# Create graph object
all_node_names <- unique(c(edges$from, edges$to))
graph <- graph_from_data_frame(edges, vertices = data.frame(name = unique(c(edges$from, edges$to))), directed = FALSE)

M <- ggraph(graph, layout = "fr") +
  geom_edge_link(aes(width = weight, color = color), alpha = 0.6) +
  geom_node_point(size = 2, aes(color = ifelse(name %in% relevant_nodes, "highlighted", "other"))) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3) +
  scale_edge_color_manual(
    name = "Edge Types",
    values = c("orange" = "orange", "grey" = "grey"),
    labels = c("orange" = "low IPMciv outliers", "grey" = "others")
  ) +
  scale_edge_width(
    name = "Z-scores",
    range = c(0.5, 2)
  ) +
  scale_color_manual(
    name = "Node Types",
    values = c("highlighted" = "darkorange", "other" = "black"),
    labels = c("highlighted" = "low IPMciv outliers", "other" = "others")
  ) +
  theme_void() +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 12),
    legend.spacing = unit(1, "lines"),
    plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(size = 12, hjust = 0.5)
  ) +
  ggtitle("")

print(M)
print(edges %>% filter(from == "HB_tnt" & to == "CPP"))

# Save the plot
ggsave(filename = "Filtered_Zscore_Network_Relevant_Nodes_Full_Network.png", plot = M, width = 6, height = 8, dpi = 900)

# Define `node_names` based on the column names in `filtered_data`
node_names <- colnames(filtered_data)

# Trim `_tnt` suffix from `node_names`
node_names <- str_remove(node_names, "_tnt$")

# Define central nodes and trim `_tnt`
central_node <- c("ARID_tnt", "Homeobox_tnt", "Homeobox_ecoli", "ZFHD_tnt", "MYBrelated_tnt", "HB_tnt", "CPP_tnt", "C2C2dof_tnt", "C2C2YABBY_tnt", "REM_tnt") %>%
  str_remove("_tnt$")

# Create edges connecting only to central nodes with Z > 1 and assign edge colors
edges <- bind_rows(
  data.frame(
    from = "ARID",
    to = node_names[!is.na(filtered_data$ARID_tnt)],
    weight = filtered_data$ARID_tnt[!is.na(filtered_data$ARID_tnt)]
  ),
  data.frame(
    from = "Homeobox",
    to = node_names[!is.na(filtered_data$Homeobox_tnt)],
    weight = filtered_data$Homeobox_tnt[!is.na(filtered_data$Homeobox_tnt)]
  ),
  data.frame(
    from = "Homeobox_ecoli",
    to = node_names[!is.na(filtered_data$Homeobox_ecoli)],
    weight = filtered_data$Homeobox_ecoli[!is.na(filtered_data$Homeobox_ecoli)]
  ),
  data.frame(
    from = "ZFHD",
    to = node_names[!is.na(filtered_data$ZFHD_tnt)],
    weight = filtered_data$ZFHD_tnt[!is.na(filtered_data$ZFHD_tnt)]
  ),
  data.frame(
    from = "MYBrelated",
    to = node_names[!is.na(filtered_data$MYBrelated_tnt)],
    weight = filtered_data$MYBrelated_tnt[!is.na(filtered_data$MYBrelated_tnt)]
  ),
  data.frame(
    from = "HB",
    to = node_names[!is.na(filtered_data$HB_tnt)],
    weight = filtered_data$HB_tnt[!is.na(filtered_data$HB_tnt)]
  ),
  data.frame(
    from = "CPP",
    to = node_names[!is.na(filtered_data$CPP_tnt)],
    weight = filtered_data$CPP_tnt[!is.na(filtered_data$CPP_tnt)]
  ),
  data.frame(
    from = "C2C2dof",
    to = node_names[!is.na(filtered_data$C2C2dof_tnt)],
    weight = filtered_data$C2C2dof_tnt[!is.na(filtered_data$C2C2dof_tnt)]
  ),
  data.frame(
    from = "C2C2YABBY",
    to = node_names[!is.na(filtered_data$C2C2YABBY_tnt)],
    weight = filtered_data$C2C2YABBY_tnt[!is.na(filtered_data$C2C2YABBY_tnt)]
  ),
  data.frame(
    from = "REM",
    to = node_names[!is.na(filtered_data$REM_tnt)],
    weight = filtered_data$REM_tnt[!is.na(filtered_data$REM_tnt)]
  )
)

# Assign colors to edges based on connection type
edges <- edges %>%
  mutate(
    color = ifelse(from == "HB" | to == "HB", "orange", "grey")
  )

# Ensure node names are unique and valid
all_node_names <- unique(c(central_node, edges$to))

# Create the graph object
graph <- graph_from_data_frame(edges, vertices = data.frame(name = all_node_names), directed = FALSE)

# Plot the graph
N <- ggraph(graph, layout = "fr") +
  geom_edge_link(aes(width = weight, color = color), alpha = 0.6) +
  geom_node_point(size = 5, color = "black") +
  geom_node_text(aes(label = name), repel = TRUE, size = 3) +
  scale_edge_color_manual(
    name = "Edge Types",
    values = c("orange" = "orange", "grey" = "grey"),
    labels = c("orange" = "Connections to HB", "grey" = "Other Connections")
  ) +
  theme_void() +
  ggtitle("")

print(N)

# Save the plot
ggsave(filename = "Subnetwork_Specified_Factors_HB_Highlighted.png", plot = N, width = 6, height = 6, dpi = 900)
