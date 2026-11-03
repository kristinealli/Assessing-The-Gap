# ============================================================
# ASSESSING THE GAP — FINAL HELD-OUT EVALUATION
# ============================================================
# Purpose:
#   Evaluate three classification models on the independent
#   BRFSS 2014–2024 held-out samples by population and period.
#
# Primary metric:
#   Weighted AUROC.
#
# Secondary metrics:
#   Average precision, accuracy, Brier score, sensitivity,
#   specificity, PPV, false-negative rate, and calibration.
#
# Important:
#   Results are weighted descriptive comparisons, not
#   complex-survey design-based confidence intervals or
#   evidence of causal bias.
#
# This script produces:
#
# Held-out performance by model, group, and period
#   outputs/tables/09_heldout_model_performance.csv
#
# AI/AN-alone versus comparison-group performance gaps
#   outputs/tables/09_aian_vs_comparison_gaps.csv
#
# Pre/post change in the AUROC performance gap
#   outputs/tables/09_change_in_auc_gap.csv
#
# Weighted AUROC comparison figure
#   outputs/figures/09_auroc_comparison.png
#
# Calibration estimates, including bin support diagnostics
#   outputs/tables/09_calibration_bins.csv
#
# Calibration plot marking sparse or weight-dominated bins
#   outputs/figures/09_calibration.png
#
# Logistic, LASSO, and random forest predictor signals
#   outputs/tables/09_predictor_signals.csv
# ============================================================


# ============================================================
# 01 — LOAD PROJECT UTILITIES
# ============================================================

source(
    file.path(
        "R",
        "01_utilities.R"
    )
)

source(
    file.path(
        "R",
        "02_metrics.R"
    )
)

check_packages(
    c("data.table", "ggplot2", "glmnet")
)

suppressPackageStartupMessages(
    library(data.table)
)


# ============================================================
# 02 — LOAD ANNUAL HELD-OUT PREDICTIONS
# ============================================================

# Use the existing predictions for all eleven survey years.
files <- file.path(
    CHECKPOINT_DIR,
    sprintf(
        "heldout_predictions_%d.rds",
        STUDY_YEARS
    )
)

if (!all(file.exists(files))) {
    stop(
        "Run 08_score_heldout_years.R to create missing predictions."
    )
}

test <- data.table::rbindlist(
    lapply(files, readRDS)
)


# ============================================================
# 03 — DEFINE STUDY PERIODS AND POPULATION GROUPS
# ============================================================

test[
    ,
    study_period := factor(
        ifelse(
            YEAR < 2020,
            "Pre-2020",
            "Post-2020"
        ),
        levels = PERIOD_LEVELS
    )
]

test[
    ,
    cohort := two_group(AIAN_ALONE)
]


# ============================================================
# 04 — VALIDATE HELD-OUT EVALUATION DATA
# ============================================================

if (anyNA(test$cohort)) {
    stop(
        "Unknown or multiracial cases entered the two-group test sample."
    )
}

if (any(!valid_weight(test$ML_WEIGHT))) {
    stop("Invalid test weights.")
}

if (any(!(test$FMD %in% c(0, 1)))) {
    stop("Invalid FMD outcome in held-out predictions.")
}

# Check that every model generated actual probabilities.
# This catches the earlier LASSO log-odds prediction error.
for (family in c("logistic", "lasso", "forest")) {

    column <- paste0("pred_", family)

    if (!column %in% names(test)) {
        stop("Missing prediction column: ", column)
    }

    predicted <- test[[column]]

    if (
        any(!is.finite(predicted)) ||
        any(predicted < 0 | predicted > 1)
    ) {
        stop(
            "Invalid predicted probabilities in ", column,
            ". Check the saved yearly predictions and ",
            "R/02_metrics.R before evaluation."
        )
    }
}


# ============================================================
# 05 — CALCULATE WEIGHTED MODEL PERFORMANCE
# ============================================================

# Compute metrics for each model × period × population group.
results <- list()

for (family in c("logistic", "lasso", "forest")) {

    for (period in PERIOD_LEVELS) {

        for (group in GROUP_LEVELS) {

            one <- test[
                study_period == period &
                cohort == group
            ]

            metrics <- score_metrics(
                one$FMD,
                one[[paste0("pred_", family)]],
                one$ML_WEIGHT
            )

            metrics$family <- family
            metrics$study_period <- period
            metrics$cohort <- group

            results[[length(results) + 1L]] <- metrics

            cat(
                sprintf(
                    "[RESULT] %-8s | %-9s | %-28s | AUROC %.3f\n",
                    family,
                    period,
                    group,
                    metrics$auroc
                )
            )
        }
    }
}

# Combine the twelve subgroup performance results.
perf <- data.table::rbindlist(
    results,
    fill = TRUE
)

data.table::setcolorder(
    perf,
    c(
        "family",
        "study_period",
        "cohort",
        setdiff(
            names(perf),
            c("family", "study_period", "cohort")
        )
    )
)

save_csv(
    perf,
    "09_heldout_model_performance.csv"
)


# ============================================================
# 06 — CALCULATE POPULATION PERFORMANCE GAPS
# ============================================================

# Subgroup gap = AI/AN-alone estimate minus comparison estimate.
aian <- as.data.frame(
    perf[cohort == GROUP_LEVELS[1]]
)

other <- as.data.frame(
    perf[cohort == GROUP_LEVELS[2]]
)

difference <- merge(
    aian[
        ,
        c("family", "study_period", "auroc", "brier", "calibration_gap")
    ],
    other[
        ,
        c("family", "study_period", "auroc", "brier", "calibration_gap")
    ],
    by = c("family", "study_period"),
    suffixes = c("_aian", "_comparison")
)

difference$auc_gap <- (
    difference$auroc_aian -
    difference$auroc_comparison
)

difference$brier_gap <- (
    difference$brier_aian -
    difference$brier_comparison
)

# Compare the subgroup discrimination gap before and after 2020.
shift <- reshape(
    difference[
        ,
        c("family", "study_period", "auc_gap")
    ],
    idvar = "family",
    timevar = "study_period",
    direction = "wide"
)

shift$change_in_auc_gap <- (
    shift[["auc_gap.Post-2020"]] -
    shift[["auc_gap.Pre-2020"]]
)

save_csv(
    difference,
    "09_aian_vs_comparison_gaps.csv"
)

save_csv(
    shift,
    "09_change_in_auc_gap.csv"
)

cat(
    "\n[INTERPRETATION] Negative AUROC gap means lower AI/AN-alone\n",
    "discrimination on this test set. Change in gap = post minus pre.\n",
    sep = ""
)

print(
    shift,
    row.names = FALSE
)


# ============================================================
# 07 — VISUALIZE WEIGHTED AUROC
# ============================================================

plot_data <- perf[
    cohort %in% GROUP_LEVELS
]

p_auc <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
        x = family,
        y = auroc,
        color = cohort
    )
) +
    ggplot2::geom_point(
        position = ggplot2::position_dodge(width = 0.45),
        size = 3
    ) +
    ggplot2::facet_wrap(~study_period) +
    ggplot2::scale_y_continuous(limits = c(0, 1)) +
    ggplot2::labs(
        title = "Held-out discrimination by population and period",
        subtitle = "Models trained on pooled BRFSS respondents",
        x = "Classifier",
        y = "Weighted AUROC (higher is better)",
        color = "Evaluation group",
        caption = paste(
            "Compare points within each panel;",
            "their separation is the subgroup gap.",
            "Weighted point estimates only."
        )
    ) +
    ggplot2::theme_minimal(base_size = 12)

print(p_auc)

save_figure(
    p_auc,
    "09_auroc_comparison.png",
    width = 10,
    height = 5
)


# ============================================================
# 08 — CALCULATE CALIBRATION BINS
# ============================================================

# Keep every respondent and every bin in the output.
# Calculate sample-size and weight-concentration diagnostics.
# No predictions are clipped: the validation above ensures
# they are already valid probabilities between 0 and 1.
cal <- list()

for (family in c("logistic", "lasso", "forest")) {

    # Prepare the observed outcomes, weights, and predictions.
    q <- test[
        ,
        .(
            FMD,
            ML_WEIGHT,
            study_period,
            cohort
        )
    ]

    q[
        ,
        pred := test[[paste0("pred_", family)]]
    ]

    # Divide probabilities into ten fixed-width intervals.
    q[
        ,
        bin := cut(
            pred,
            breaks = seq(0, 1, by = 0.1),
            include.lowest = TRUE
        )
    ]

    # The weight-based effective_n is Kish's approximation.
    # It is NOT a complex-survey design-based effective n.
    bins <- q[
        ,
        .(
            respondents = .N,
            effective_n = sum(ML_WEIGHT)^2 / sum(ML_WEIGHT^2),
            largest_weight_share_pct = 100 *
                max(ML_WEIGHT) / sum(ML_WEIGHT),
            predicted_mean = stats::weighted.mean(
                pred,
                ML_WEIGHT
            ),
            observed_fmd = stats::weighted.mean(
                FMD,
                ML_WEIGHT
            )
        ),
        by = .(
            study_period,
            cohort,
            bin
        )
    ]

    bins[
        ,
        family := family
    ]

    cal[[family]] <- bins
}


# ============================================================
# 09 — FLAG SPARSE OR WEIGHT-DOMINATED BINS
# ============================================================

# IMPORTANT: Create calibration BEFORE assigning bin_quality.
calibration <- data.table::rbindlist(
    cal,
    use.names = TRUE
)

# Thresholds are exploratory DISPLAY rules, not formal
# significance tests or BRFSS survey-design thresholds.
calibration[
    ,
    bin_quality := ifelse(
        respondents >= 20L & effective_n >= 10,
        "Supported bin",
        "Sparse or weight-dominated"
    )
]

# Keep both supported and flagged bins in the exported table.
save_csv(
    calibration,
    "09_calibration_bins.csv"
)

# Report bins with limited information (including the
# three-person post-2020 AI/AN high-risk LASSO/logistic bins).
cat("\n[FLAGGED CALIBRATION BINS]\n")

print(
    calibration[
        bin_quality == "Sparse or weight-dominated",
        .(
            family,
            study_period,
            cohort,
            bin,
            respondents,
            effective_n,
            largest_weight_share_pct
        )
    ]
)


# ============================================================
# 10 — VISUALIZE CALIBRATION WITH BIN-QUALITY MARKERS
# ============================================================

# Circles represent bins meeting the display thresholds.
# Crosses represent sparse or weight-dominated bins.
# Omit connecting lines: a three-person tail bin should not
# appear as a smooth trend extending from neighboring bins.
# All bin estimates are still shown, with no data removed.
calibration[
    ,
    bin_quality := factor(
        bin_quality,
        levels = c(
            "Supported bin",
            "Sparse or weight-dominated"
        )
    )
]

p_cal <- ggplot2::ggplot(
    calibration,
    ggplot2::aes(
        x = predicted_mean,
        y = observed_fmd,
        color = cohort
    )
) +
    ggplot2::geom_abline(
        slope = 1,
        intercept = 0,
        linetype = "dashed"
    ) +
    ggplot2::geom_point(
        ggplot2::aes(shape = bin_quality),
        size = 2.8,
        alpha = 0.85
    ) +
    ggplot2::scale_shape_manual(
        values = c(
            "Supported bin" = 16,
            "Sparse or weight-dominated" = 4
        )
    ) +
    ggplot2::facet_grid(study_period ~ family) +
    ggplot2::coord_equal(
        xlim = c(0, 1),
        ylim = c(0, 1)
    ) +
    ggplot2::labs(
        title = "Calibration of predicted FMD risk",
        subtitle = paste(
            "Weighted observed FMD frequency",
            "versus predicted probability"
        ),
        x = "Mean predicted probability",
        y = "Observed proportion with FMD",
        color = "Evaluation group",
        shape = "Bin support",
        caption = paste(
            "Dashed diagonal indicates perfect calibration.",
            "Crosses mark bins with <20 respondents or",
            "weight-based effective n <10.",
            "All bins retained; thresholds are descriptive."
        )
    ) +
    ggplot2::theme_minimal(base_size = 11)

print(p_cal)

save_figure(
    p_cal,
    "09_calibration.png",
    width = 12,
    height = 7.5
)


# ============================================================
# 11 — LOAD FINAL MODELS FOR PREDICTOR INTERPRETATION
# ============================================================

logistic_fit <- readRDS(
    file.path(MODEL_DIR, "logistic.rds")
)

lasso_fit <- readRDS(
    file.path(MODEL_DIR, "lasso.rds")
)

forest_fit <- readRDS(
    file.path(MODEL_DIR, "forest.rds")
)


# ============================================================
# 12 — EXTRACT AND SAVE PREDICTOR SIGNALS
# ============================================================

# These measures have different scales and are not
# directly comparable across model families.
log_coef <- stats::coef(
    logistic_fit$model
)

# Use generic dispatch for glmnet's selected lambda.
lasso_coef <- as.matrix(
    stats::coef(
        lasso_fit$model,
        s = as.numeric(lasso_fit$config)
    )
)

forest_imp <- forest_fit$model$variable.importance

signals <- rbind(
    data.frame(
        family = "logistic",
        term = names(log_coef),
        value = as.numeric(log_coef),
        type = "log-odds coefficient"
    ),
    data.frame(
        family = "lasso",
        term = rownames(lasso_coef),
        value = as.numeric(lasso_coef[, 1]),
        type = "penalized log-odds coefficient"
    ),
    data.frame(
        family = "forest",
        term = names(forest_imp),
        value = as.numeric(forest_imp),
        type = "impurity importance (biased by variable scale)"
    )
)

save_csv(
    signals,
    "09_predictor_signals.csv"
)


# ============================================================
# 13 — RELEASE MEMORY AND REPORT COMPLETION
# ============================================================

rm(test)

gc(verbose = FALSE)

message(
    "[COMPLETE] Final held-out evaluation and figures saved."
)
