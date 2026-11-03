# ============================================================
# GENERATE STAKEHOLDER-FACING HTML REPORT
# ============================================================

if (!requireNamespace("rmarkdown", quietly = TRUE)) {
    stop("Install rmarkdown: install.packages('rmarkdown')")
}
report <- file.path("reports", "Final_Research_Report.Rmd")
rmarkdown::render(
    report,
    output_file = "Assessing_the_Gap_Report.html",
    output_dir = file.path(getwd(), "reports"),
    quiet = FALSE,
    envir = new.env(parent = globalenv())
)
cat("[COMPLETE] HTML report written to reports/.\n")
