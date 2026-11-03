# ============================================================
# ASSESSING THE GAP — SHARED UTILITIES
# ============================================================
# Purpose:
#   Provide reusable functions for loading and validating
#   data, managing checkpoints, preparing model features,
#   calculating weighted statistics, and saving outputs.
#
# Usage:
#   Source this file from the RStudio project root.
#
#   These utilities are shared across the analysis scripts
#   to maintain consistent processing and reproducibility.
# ============================================================


# ============================================================
# 01 — LOAD PROJECT CONFIGURATION
# ============================================================

# Load project paths, study variables, model formulas,
# and reproducibility settings.

source(
    file.path(
        "R",
        "00_config.R"
    )
)


# ============================================================
# 02 — VERIFY REQUIRED R PACKAGES
# ============================================================

# Define the packages required for the primary analyses.
required_packages <- c(
    "data.table",
    "ggplot2",
    "glmnet",
    "ranger"
)

# Check whether the specified R packages are available.
#
# The function checks each package without attaching it
# to the search path.
#
# If a package is unavailable, execution stops and
# provides the installation command.

check_packages <- function(packages = required_packages) {
    
    # Identify packages that are not available.
    missing <- packages[
        !vapply(
            packages,
            requireNamespace,
            logical(1),
            quietly = TRUE
        )
    ]
    
    # Stop if any required packages are missing.
    if (length(missing) > 0) {
        
        stop(
            "Missing R packages: ",
            paste(missing, collapse = ", "),
            "\nRun install.packages(c(",
            paste(
                sprintf('"%s"', missing),
                collapse = ", "
            ),
            "))",
            call. = FALSE
        )
    }
}


# ============================================================
# 03 — SAVE ANALYSIS CHECKPOINTS
# ============================================================

# Save completed R objects as compressed RDS files.
#
# Checkpoints allow expensive processing steps to be
# restored without repeating calculations.
#
# A temporary file is written before being renamed to
# the final destination, reducing the risk of leaving
# a partially written checkpoint.
#
# Compression:
#   gzip provides lossless compression to reduce
#   the storage space required by saved R objects.

save_checkpoint <- function(value, path) {
    
    # Ensure the destination directory exists.
    dir.create(
        dirname(path),
        recursive = TRUE,
        showWarnings = FALSE
    )
    
    # Construct the temporary checkpoint path.
    temporary <- paste0(
        path,
        ".writing"
    )
    
    # Save the object using gzip compression.
    saveRDS(
        value,
        temporary,
        compress = "gzip"
    )
    
    # Rename the completed temporary file.
    if (!file.rename(temporary, path)) {
        
        # Remove the temporary file if saving fails.
        unlink(temporary)
        
        stop(
            "Could not save checkpoint: ",
            path
        )
    }
    
    # Confirm successful checkpoint creation.
    message("[SAVED] ", path)
    
    return(invisible(value))
}


# ============================================================
# 04 — LOAD AND VALIDATE THE POOLED DATASET
# ============================================================

# Load the previously constructed BRFSS 2014–2024 dataset.
#
# The dataset contains:
#   4,790,742 respondents
#   25 variables
#   11 survey years
#
# Validation:
#   Confirm that the dataset file exists, required
#   columns are present, and all study years are included.
#
# Important:
#   This function does not rebuild the pooled dataset
#   or independently validate annual CDC source recodes.

load_data <- function() {
    
    # Verify that the pooled dataset exists.
    if (!file.exists(DATA_FILE)) {
        
        stop(
            "Pooled dataset missing: ",
            DATA_FILE
        )
    }
    
    message(
        "[LOAD] Reading the already constructed pooled dataset..."
    )
    
    # Load the processed dataset into memory.
    d <- readRDS(DATA_FILE)
    
    # Convert to data.table for efficient processing.
    data.table::setDT(d)
    
    # Define the variables required for analysis.
    expected <- c(
        "YEAR",
        "STATE",
        "STRAT",
        "PSU",
        "RAW_WEIGHT",
        "POOL_WEIGHT",
        "RACE_ALONE_RAW",
        "AIAN_ALONE",
        "AIAN_NH_ALONE",
        "MENTHLTH_RAW",
        "MENTHLTH_DAYS",
        "FMD",
        FEATURES
    )
    
    # Identify missing required variables.
    absent <- setdiff(
        expected,
        names(d)
    )
    
    # Stop if any required variables are unavailable.
    if (length(absent) > 0) {
        
        stop(
            "Missing required fields: ",
            paste(absent, collapse = ", ")
        )
    }
    
    # Confirm that the dataset includes the study years.
    if (!identical(
        sort(unique(as.integer(d$YEAR))),
        STUDY_YEARS
    )) {
        
        stop(
            "The dataset does not contain exactly ",
            "the study years 2014–2024."
        )
    }
    
    # Report the loaded dataset dimensions.
    message(
        sprintf(
            "[DATA] %s rows × %d columns",
            format(nrow(d), big.mark = ","),
            ncol(d)
        )
    )
    
    return(d)
}


# ============================================================
# 05 — STANDARDIZE PREDICTOR VARIABLES
# ============================================================

# Standardize the data types and categorical levels
# used by the predictive models.
#
# This ensures that training, cross-validation, and
# held-out datasets use consistent feature definitions.
#
# Categorical variables are converted into factors
# with explicitly defined levels.
#
# Health variables are converted into numeric values.
#
# Note:
#   This function does not impute missing values.

normalize_features <- function(d) {
    
    # --------------------------------------------------------
    # 05A — AGE CATEGORIES
    # --------------------------------------------------------
    
    d$AGE_BAND <- factor(
        as.character(d$AGE_BAND),
        levels = c(
            "18-34",
            "35-49",
            "50-64",
            "65+"
        )
    )
    
    
    # --------------------------------------------------------
    # 05B — SEX CLASSIFICATION
    # --------------------------------------------------------
    
    # Use the sex categories recorded in the dataset.
    d$SEX_CLEAN <- factor(
        as.character(d$SEX_CLEAN),
        levels = c(
            "Female",
            "Male"
        )
    )
    
    
    # --------------------------------------------------------
    # 05C — EDUCATIONAL ATTAINMENT
    # --------------------------------------------------------
    
    d$EDU_CLEAN <- factor(
        as.character(d$EDU_CLEAN),
        levels = c(
            "<HS",
            "HS",
            "Some college",
            "College grad"
        )
    )
    
    
    # --------------------------------------------------------
    # 05D — HOUSEHOLD INCOME
    # --------------------------------------------------------
    
    d$INCOME_BAND <- factor(
        as.character(d$INCOME_BAND),
        levels = c(
            "<25k",
            "25-49k",
            "50-74k",
            "75k+"
        )
    )
    
    
    # --------------------------------------------------------
    # 05E — SURVEY PERIOD
    # --------------------------------------------------------
    
    d$PERIOD <- factor(
        as.character(d$PERIOD),
        levels = PERIOD_LEVELS
    )
    
    
    # --------------------------------------------------------
    # 05F — HEALTH VARIABLES
    # --------------------------------------------------------
    
    # Convert self-rated general health to numeric.
    d$GENHLTH_CLEAN <- as.numeric(
        d$GENHLTH_CLEAN
    )
    
    # Convert physically unhealthy days to numeric.
    d$PHYSHLTH_DAYS <- as.numeric(
        d$PHYSHLTH_DAYS
    )
    
    return(d)
}


# ============================================================
# 06 — CLASSIFY STUDY POPULATIONS
# ============================================================

# Convert the existing AI/AN-alone indicator into
# readable population labels for tables and figures.
#
# Expected input:
#   TRUE  = AI/AN alone
#   FALSE = Other identified single-race
#   NA    = Excluded or unclassified
#
# Important:
#   This function does not independently determine race.
#
#   The source indicator must already exclude multiracial
#   and unknown classifications, representing them as NA.
#
#   The comparison group should not be interpreted as
#   containing no respondents with Indigenous ancestry.

two_group <- function(flag) {
    
    return(
        ifelse(
            is.na(flag),
            NA_character_,
            ifelse(
                flag,
                GROUP_LEVELS[1],
                GROUP_LEVELS[2]
            )
        )
    )
}


# ============================================================
# 07 — VALIDATE SURVEY WEIGHTS
# ============================================================

# Identify finite, positive weights.
#
# TRUE:
#   Weight is finite and greater than zero.
#
# FALSE:
#   Weight is missing, non-finite, zero, or negative.
#
# This checks numeric validity, not whether the weighting
# methodology is appropriate for multi-year BRFSS inference.

valid_weight <- function(w) {
    
    return(
        is.finite(w) & w > 0
    )
}


# ============================================================
# 08 — CALCULATE WEIGHTED FMD PREVALENCE
# ============================================================

# Calculate the weighted proportion of respondents
# reporting frequent mental distress (FMD = 1).
#
# Eligible observations:
#   FMD must equal 0 or 1.
#   Weight must be finite and positive.
#
# Returns:
#   A proportion between 0 and 1.
#   NA if no eligible observations are available.
#
# Example:
#   0.18 corresponds to an estimated prevalence of 18%.
#
# Limitation:
#   This function calculates a weighted point estimate.
#   It does not estimate complex-survey standard errors
#   or confidence intervals.

weighted_rate <- function(y, w) {
    
    # Identify valid outcomes and weights.
    ok <- (
        y %in% c(0, 1) &
            valid_weight(w)
    )
    
    # Return NA if no observations are eligible.
    if (!any(ok)) {
        return(NA_real_)
    }
    
    # Calculate weighted FMD prevalence.
    return(
        stats::weighted.mean(
            y[ok],
            w[ok]
        )
    )
}


# ============================================================
# 09 — SAVE ANALYSIS TABLES
# ============================================================

# Export tables as comma-separated value (CSV) files.
#
# Files are saved to the project's tables directory.
# Missing values are written as empty fields.
#
# CSV files can be opened in spreadsheet software
# or imported into reporting and visualization tools.

save_csv <- function(d, filename) {
    
    # Construct the output file path.
    path <- file.path(
        TABLE_DIR,
        filename
    )
    
    # Export the table.
    data.table::fwrite(
        as.data.frame(d),
        path,
        na = ""
    )
    
    # Confirm successful export.
    message("[SAVED] ", path)
}


# ============================================================
# 10 — SAVE ANALYSIS FIGURES
# ============================================================

# Export ggplot2 figures using consistent formatting.
#
# Default settings:
#   Width      = 9 inches
#   Height     = 5.5 inches
#   Resolution = 180 DPI
#   Background = White
#
# Figures are saved to the project's figures directory.
#
# Titles, legends, axis labels, and explanatory captions
# should be defined in the plot before calling this function.

save_figure <- function(
        p,
        filename,
        width = 9,
        height = 5.5
) {
    
    # Construct the figure output path.
    path <- file.path(
        FIGURE_DIR,
        filename
    )
    
    # Save the figure.
    ggplot2::ggsave(
        filename = path,
        plot = p,
        width = width,
        height = height,
        dpi = 180,
        bg = "white"
    )
    
    # Confirm successful export.
    message("[SAVED] ", path)
}


# ============================================================
# 11 — CREATE MACHINE-LEARNING DESIGN MATRIX
# ============================================================

# Convert predictor variables into a numeric matrix.
#
# The DESIGN_FORMULA specifies the predictor variables
# used for machine learning.
#
# model.matrix() converts categorical predictors into
# numeric indicator columns.
#
# The intercept column is removed because the relevant
# modeling algorithms handle intercepts separately.
#
# Column names are standardized for consistent use
# across model training, validation, and testing.
#
# This function requires complete predictor values.

make_matrix <- function(dat) {
    
    # Generate the numeric predictor matrix.
    X <- stats::model.matrix(
        DESIGN_FORMULA,
        data = dat
    )
    
    # Remove the intercept column.
    X <- X[
        ,
        colnames(X) != "(Intercept)",
        drop = FALSE
    ]
    
    # Standardize column names.
    colnames(X) <- make.names(
        colnames(X),
        unique = TRUE
    )
    
    # Stop if missing values remain.
    if (anyNA(X)) {
        
        stop(
            "Missing values in design matrix."
        )
    }
    
    return(X)
}


# ============================================================
# 12 — NORMALIZE MODEL TRAINING WEIGHTS
# ============================================================

# Normalize model weights so their mean equals 1.
#
# Normalization preserves the relative weight assigned
# to each respondent while adjusting the overall scale.
#
# ML_WEIGHT should already incorporate any adjustments
# for respondent sampling or training-set inclusion.
#
# This function does not calculate inclusion probabilities
# or validate the original pooled survey-weight formula.

model_weight <- function(dat) {
    
    # Retrieve the model training weights.
    w <- dat$ML_WEIGHT
    
    # Verify that all weights are finite and positive.
    if (any(!valid_weight(w))) {
        
        stop(
            "Invalid model weights."
        )
    }
    
    # Normalize weights to an average of 1.
    return(
        w / mean(w)
    )
}
