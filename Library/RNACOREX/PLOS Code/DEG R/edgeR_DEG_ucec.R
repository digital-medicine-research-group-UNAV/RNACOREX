# Load required libraries
library(edgeR)

data_edger <- read.csv("data_deg_ucec.csv")
data_edger <- data_edger[,-1]
row_names <- colnames(data_edger)
data_edger_trans <- t(data_edger)
rownames(data_edger_trans) <- row_names

counts_data <- as.matrix(data_edger_trans[-nrow(data_edger_trans), ])
classes <- data_edger_trans[nrow(data_edger_trans), ]
groups <- data.frame(group = classes)

# Create DGEList object
dge <- DGEList(counts = counts_data, group = groups$group)

# Filter lowly expressed genes
# keep <- rowSums(cpm(dge) > 1) >= 3
# dge <- dge[keep,]

# Normalization (TMM normalizes the gene expression from RNAseq, log-fold change?)
dge <- calcNormFactors(dge)

# Estimate dispersion
dge <- estimateDisp(dge)

# Fit generalized linear model
design <- model.matrix(~ groups$group)
fit <- glmFit(dge, design)

# Perform likelihood ratio test
lrt <- glmLRT(fit, coef = 2)

# Get top differentially expressed genes
top_genes <- topTags(lrt, n = Inf)$table
top_genes <- top_genes[top_genes$PValue < 0.05, ]

deg_genes <- rownames(top_genes)
df <- as.data.frame(deg_genes)
write.csv(df, file = "deg_genes_ucec.csv", row.names = FALSE)