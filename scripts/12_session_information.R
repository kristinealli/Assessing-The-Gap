# ============================================================
# RECORD REPRODUCIBILITY INFORMATION
# ============================================================

source(file.path("R", "00_config.R"))
capture.output(
    sessionInfo(),
    file = file.path(OUT_DIR, "logs", "R_session_info.txt")
)
cat("[SAVED] outputs/logs/R_session_info.txt\n")
