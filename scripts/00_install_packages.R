# ============================================================
# INSTALL DEPENDENCIES 
# ============================================================

packages <- c(
    "data.table", "ggplot2", "glmnet", "ranger",
    "rmarkdown", "knitr"
)

missing <- packages[
    !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing)) {
    install.packages(missing, repos = "https://cloud.r-project.org")
}
cat("[READY] Project R packages installed.\n")

