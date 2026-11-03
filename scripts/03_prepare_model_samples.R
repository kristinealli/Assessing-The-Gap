# ============================================================
# ASSESSING THE GAP — MODEL SAMPLE PREPARATION
# ============================================================
# Purpose:
#   Prepare reproducible training and held-out evaluation
#   samples from the pooled BRFSS 2014–2024 dataset.
#
# Research Objective:
#   Develop classification models for frequent mental
#   distress (FMD) and compare predictive performance
#   between AI/AN-alone and other single-race adults.
#
# Sampling Strategy:
#   Stratify respondents by survey year, population group,
#   and FMD outcome.
#
#   Every study year contributes observations to both
#   model training and held-out evaluation.
#
#   All eligible AI/AN-alone training observations are
#   retained. The larger comparison group is capped
#   for computational feasibility.
#
# Weighting:
#   ML_WEIGHT adjusts the pooled survey weights for
#   training-set selection and computational subsampling.
#
# Cross-Validation:
#   Generate one common, reproducible set of folds
#   shared by all classification models.
#
# Important:
#   Held-out outcomes are not used during model tuning.
#   Existing checkpoints are reused when available.
# 
# This script produces: 
#   output/checkpoints/training_sample.rds
#   output/checkpoints/heldout_split.rds
#   output/checkpoints/common_cv_folds.rds
# ============================================================


# ============================================================
# 01 — LOAD PROJECT UTILITIES
# ============================================================

# Import shared configuration, data validation functions,
# feature preparation utilities, and checkpoint management.

source(
    file.path(
        "R",
        "01_utilities.R"
    )
)

# Verify that data.table is installed.
check_packages(
    c("data.table")
)

# Load data.table without displaying startup messages.
suppressPackageStartupMessages(
    library(data.table)
)


# ============================================================
# 02 — DEFINE CHECKPOINT PATHS
# ============================================================

# Define the file paths used to save and retrieve:
#   Training dataset.
#   Held-out respondent indices and inclusion probabilities.
#   Common cross-validation fold assignments.

train_file <- file.path(
    CHECKPOINT_DIR,
    "training_sample.rds"
)

test_file <- file.path(
    CHECKPOINT_DIR,
    "heldout_split.rds"
)

fold_file <- file.path(
    CHECKPOINT_DIR,
    "common_cv_folds.rds"
)


# ============================================================
# 03 — CHECK FOR EXISTING MODEL SAMPLES
# ============================================================

# Reuse previously generated samples if all three
# checkpoint files are available.
#
# This prevents unnecessary resampling and ensures
# consistency across repeated analysis runs.
#
# To change the sampling settings, archive or remove
# the existing checkpoints before rerunning this script.

if (all(file.exists(c(
    train_file,
    test_file,
    fold_file
)))) {
    
    message(
        "[SKIP] Samples and folds already exist. To change",
        " sampling settings, archive the checkpoints first."
    )
    
} else {
    
    
    # ========================================================
    # 04 — LOAD POOLED BRFSS DATASET
    # ========================================================
    
    # Load the previously constructed pooled dataset.
    d <- load_data()
    
    
    # ========================================================
    # 05 — IDENTIFY ELIGIBLE RESPONDENTS
    # ========================================================
    
    # Select respondents meeting all model eligibility rules.
    #
    # Requirements:
    #   FMD outcome must equal 0 or 1.
    #   AI/AN-alone classification must be available.
    #   Pooled survey weight must be finite and positive.
    #   All model predictor variables must be complete.
    #
    # Multiracial and unknown classifications are excluded
    # from the primary two-group model comparison.
    
    eligible <- which(
        d$FMD %in% c(0, 1) &
            !is.na(d$AIAN_ALONE) &
            valid_weight(d$POOL_WEIGHT) &
            stats::complete.cases(
                d[
                    ,
                    ..FEATURES
                ]
            )
    )
    
    # Verify that enough respondents are available
    # to construct the training and test samples.
    if (length(eligible) < 10000L) {
        
        stop(
            "Insufficient eligible cases for pooled modeling."
        )
    }
    
    
    # ========================================================
    # 06 — CONSTRUCT STRATIFIED SAMPLING GROUPS
    # ========================================================
    
    # Stratify eligible respondents by:
    #   Survey year.
    #   AI/AN-alone classification.
    #   Observed FMD outcome.
    #
    # This helps preserve the representation of each
    # year, population group, and outcome during splitting.
    
    strata <- interaction(
        d$YEAR[eligible],
        d$AIAN_ALONE[eligible],
        d$FMD[eligible],
        drop = TRUE
    )
    
    # Divide eligible respondent indices into strata.
    groups <- split(
        eligible,
        strata
    )
    
    # Set the random seed for reproducible splitting.
    set.seed(SEED)
    
    
    # ========================================================
    # 07 — INITIALIZE HELD-OUT SAMPLE STORAGE
    # ========================================================
    
    # Store respondent indices selected for evaluation.
    held_ids <- integer(0)
    
    # Store each held-out respondent's probability
    # of selection into the evaluation sample.
    held_pi <- numeric(0)
    
    # Store the training inclusion fraction
    # associated with each sampling stratum.
    train_fraction <- numeric(
        length(groups)
    )
    
    names(train_fraction) <- names(groups)
    
    
    # ========================================================
    # 08 — CREATE STRATIFIED TRAIN / TEST SPLIT
    # ========================================================
    
    # Select approximately TEST_FRACTION of respondents
    # from each stratum for held-out evaluation.
    #
    # The default test fraction is 20%.
    #
    # At least one observation is selected per stratum.
    # A stratum must contain enough observations to leave
    # at least one respondent for training.
    
    for (name in names(groups)) {
        
        # Retrieve respondent indices for this stratum.
        ids <- groups[[name]]
        
        # Calculate the number of held-out respondents.
        n_test <- max(
            1L,
            round(
                length(ids) * TEST_FRACTION
            )
        )
        
        # Prevent a stratum from being assigned
        # entirely to the held-out sample.
        if (n_test >= length(ids)) {
            
            stop(
                "Cannot split an undersized stratum: ",
                name
            )
        }
        
        # Randomly select held-out respondents.
        chosen <- sample(
            ids,
            n_test
        )
        
        # Append the selected respondent indices.
        held_ids <- c(
            held_ids,
            chosen
        )
        
        # Record held-out selection probabilities.
        held_pi <- c(
            held_pi,
            rep(
                n_test / length(ids),
                n_test
            )
        )
        
        # Record the remaining training fraction.
        train_fraction[name] <- (
            1 - n_test / length(ids)
        )
    }
    
    
    # ========================================================
    # 09 — ORGANIZE HELD-OUT RESPONDENTS
    # ========================================================
    
    # Sort the held-out indices into their original
    # dataset order.
    ord <- order(held_ids)
    
    held_ids <- held_ids[ord]
    held_pi <- held_pi[ord]
    
    # Create a logical indicator for all pooled respondents.
    #
    # TRUE:
    #   Selected for held-out evaluation.
    #
    # FALSE:
    #   Not selected for held-out evaluation.
    held_flag <- rep(
        FALSE,
        nrow(d)
    )
    
    # Mark held-out respondents.
    held_flag[held_ids] <- TRUE
    
    
    # ========================================================
    # 10 — IDENTIFY TRAINING CANDIDATES
    # ========================================================
    
    # Retain eligible respondents not assigned to
    # held-out evaluation.
    candidate <- eligible[
        !held_flag[eligible]
    ]
    
    # Separate AI/AN-alone training candidates.
    aian <- candidate[
        d$AIAN_ALONE[candidate] %in% TRUE
    ]
    
    # Separate comparison-group training candidates.
    other <- candidate[
        d$AIAN_ALONE[candidate] %in% FALSE
    ]
    
    
    # ========================================================
    # 11 — SUBSAMPLE THE COMPARISON GROUP
    # ========================================================
    
    # Retain all eligible AI/AN-alone training respondents.
    #
    # Limit the much larger comparison training population
    # to OTHER_TRAIN_CAP observations.
    #
    # The default cap is 90,000 respondents.
    
    n_other <- min(
        length(other),
        OTHER_TRAIN_CAP
    )
    
    # Randomly sample the comparison group only
    # when its eligible size exceeds the training cap.
    sampled_other <- if (n_other < length(other)) {
        
        sample(
            other,
            n_other
        )
        
    } else {
        
        other
    }
    
    # Combine AI/AN-alone and comparison-group indices.
    take <- sort(
        c(
            aian,
            sampled_other
        )
    )
    
    # Calculate the comparison group's computational
    # inclusion probability.
    other_pi <- n_other / length(other)
    
    
    # ========================================================
    # 12 — CALCULATE TRAINING INCLUSION PROBABILITIES
    # ========================================================
    
    # Identify the original sampling stratum associated
    # with each selected training respondent.
    selected_strata <- as.character(
        interaction(
            d$YEAR[take],
            d$AIAN_ALONE[take],
            d$FMD[take],
            drop = TRUE
        )
    )
    
    # Retrieve the probability of remaining
    # in training after the held-out split.
    split_pi <- as.numeric(
        train_fraction[selected_strata]
    )
    
    # Assign computational sampling probabilities.
    #
    # AI/AN alone:
    #   All eligible training respondents retained.
    #   Probability = 1.
    #
    # Comparison group:
    #   Probability equals the proportion retained
    #   after applying the computational training cap.
    computational_pi <- ifelse(
        d$AIAN_ALONE[take] %in% TRUE,
        1,
        other_pi
    )
    
    # Verify that training inclusion probabilities
    # are available and greater than zero.
    if (
        anyNA(split_pi) ||
        any(split_pi <= 0)
    ) {
        
        stop(
            "Invalid training inclusion probabilities."
        )
    }
    
    
    # ========================================================
    # 13 — BUILD THE TRAINING DATASET
    # ========================================================
    
    # Select the variables required for model training.
    #
    # Includes:
    #   Survey year.
    #   Population classification.
    #   FMD outcome.
    #   Primary pooled survey weight.
    #   All model predictors.
    
    columns <- unique(
        c(
            "YEAR",
            "AIAN_ALONE",
            "FMD",
            "POOL_WEIGHT",
            FEATURES
        )
    )
    
    # Extract the selected training observations
    # and required variables.
    train <- as.data.frame(
        d[
            take,
            ..columns
        ]
    )
    
    # Standardize categorical factor levels
    # and numeric predictor variables.
    train <- normalize_features(
        train
    )
    
    
    # ========================================================
    # 14 — CALCULATE MODEL TRAINING WEIGHTS
    # ========================================================
    
    # Adjust pooled survey weights for both:
    #   Probability of inclusion in the training split.
    #   Probability of computational subsampling.
    #
    # Formula:
    #   ML_WEIGHT = POOL_WEIGHT /
    #               (split_pi * computational_pi)
    #
    # This preserves the relative contribution intended
    # by the original pooled weights after subsampling.
    #
    # It does not independently validate the original
    # pooled survey-weight methodology.
    
    train$ML_WEIGHT <- train$POOL_WEIGHT / (
        split_pi * computational_pi
    )
    
    # Verify that every training weight is valid.
    if (any(!valid_weight(train$ML_WEIGHT))) {
        
        stop(
            "Invalid computationally corrected weights."
        )
    }
    
    
    # ========================================================
    # 15 — CREATE COMMON CROSS-VALIDATION FOLDS
    # ========================================================
    
    # Generate reproducible cross-validation assignments
    # shared by all machine-learning model families.
    #
    # The default number of folds is three.
    #
    # Stratification uses:
    #   Survey year.
    #   Population classification.
    #   FMD outcome.
    #
    # Each model uses the same fold assignments,
    # supporting comparable model evaluation.
    
    
    # --------------------------------------------------------
    # 15A — INITIALIZE FOLD ASSIGNMENTS
    # --------------------------------------------------------
    
    # Set a separate random seed for CV fold allocation.
    set.seed(
        SEED + 1L
    )
    
    # Create an empty fold assignment for each
    # training respondent.
    fold <- integer(
        nrow(train)
    )
    
    
    # --------------------------------------------------------
    # 15B — DEFINE CROSS-VALIDATION STRATA
    # --------------------------------------------------------
    
    # Group training-row positions by survey year,
    # population classification, and FMD outcome.
    fold_groups <- split(
        seq_len(nrow(train)),
        interaction(
            train$YEAR,
            train$AIAN_ALONE,
            train$FMD,
            drop = TRUE
        )
    )
    
    
    # --------------------------------------------------------
    # 15C — ASSIGN RESPONDENTS TO FOLDS
    # --------------------------------------------------------
    
    # Shuffle observations within each stratum.
    #
    # Assign fold numbers cyclically so the strata
    # are distributed across cross-validation folds.
    for (ids in fold_groups) {
        
        # Randomize respondent positions.
        shuffled <- ids[
            sample.int(length(ids))
        ]
        
        # Assign fold identifiers.
        fold[shuffled] <- rep(
            seq_len(CV_FOLDS),
            length.out = length(ids)
        )
    }
    
    
    # --------------------------------------------------------
    # 15D — VALIDATE FOLD ASSIGNMENTS
    # --------------------------------------------------------
    
    # Confirm every training respondent was assigned
    # to a cross-validation fold.
    if (any(fold == 0L)) {
        
        stop(
            "One or more training rows are missing a CV fold."
        )
    }
    
    
    # ========================================================
    # 16 — SAVE MODEL PREPARATION CHECKPOINTS
    # ========================================================
    
    # Store held-out indices, their inclusion probabilities,
    # and the random sampling configuration.
    save_checkpoint(
        list(
            indices = held_ids,
            probabilities = held_pi,
            seed = SEED,
            fraction = TEST_FRACTION
        ),
        test_file
    )
    
    # Save the prepared model training dataset.
    save_checkpoint(
        train,
        train_file
    )
    
    # Save the common cross-validation assignments.
    save_checkpoint(
        fold,
        fold_file
    )
    
    
    # ========================================================
    # 17 — REPORT SAMPLE SIZES
    # ========================================================
    
    # Display the number of respondents selected for
    # held-out evaluation and model training.
    #
    # Also report the training sample sizes for
    # the two primary population groups.
    
    cat(
        "[SPLIT] Held-out:",
        length(held_ids),
        "| model training:",
        nrow(train),
        "| AI/AN training:",
        length(aian),
        "| comparison training:",
        n_other,
        "\n"
    )
    
    # Display the number of training respondents
    # assigned to each cross-validation fold.
    cat(
        "[FOLDS]\n"
    )
    
    print(
        table(fold)
    )
    
    
    # ========================================================
    # 18 — RELEASE MEMORY
    # ========================================================
    
    # Remove large datasets and intermediate objects
    # no longer required after checkpoint creation.
    rm(
        d,
        train,
        groups,
        other,
        aian,
        candidate
    )
    
    # Request garbage collection to release unused memory.
    gc(
        verbose = FALSE
    )
}
