# ============================================================
# ASSESSING THE GAP — WEIGHTED MODEL EVALUATION
# ============================================================
# Purpose:
#   Provide reusable functions for fitting, predicting,
#   and evaluating machine-learning classification models.
#
# Models:
#   Logistic Regression
#   LASSO Logistic Regression
#   Random Forest
#
# Evaluation:
#   Compare model performance using weighted discrimination,
#   classification, and calibration metrics.
#
# Weighting:
#   Uses survey weights adjusted for computational
#   sub-sampling to support population comparisons.
# ============================================================


# ============================================================
# 01 — CALCULATE WEIGHTED AUROC
# ============================================================

# Calculate the weighted Area Under the Receiver Operating
# Characteristic Curve (AUROC).
#
# AUROC measures how well a model distinguishes between
# respondents with and without frequent mental distress.
#
# Interpretation:
#   0.50 = No better discrimination than random ranking
#   1.00 = Perfect discrimination
#
# Higher AUROC indicates better discrimination.
#
# This calculation accounts for observation weights
# and gives tied prediction scores half credit.

weighted_auc <- function(y, p, w) {
    
    # --------------------------------------------------------
    # 01A — VALIDATE INPUTS
    # --------------------------------------------------------
    
    # Include observations with:
    #   Binary outcomes (0 or 1)
    #   Finite predicted probabilities
    #   Finite, positive weights
    
    valid <- (
        y %in% c(0, 1) &
            is.finite(p) &
            is.finite(w) &
            w > 0
    )
    
    # Retain eligible observations.
    y <- y[valid]
    p <- p[valid]
    w <- w[valid]
    
    # AUROC requires both outcome classes.
    if (!any(y == 0) || !any(y == 1)) {
        return(NA_real_)
    }
    
    
    # --------------------------------------------------------
    # 01B — SORT PREDICTED PROBABILITIES
    # --------------------------------------------------------
    
    # Order predictions from lowest to highest.
    ix <- order(p)
    
    y <- y[ix]
    p <- p[ix]
    w <- w[ix]
    
    
    # --------------------------------------------------------
    # 01C — GROUP IDENTICAL PREDICTIONS
    # --------------------------------------------------------
    
    # Assign the same group to observations with
    # identical predicted probabilities.
    group <- cumsum(
        c(
            TRUE,
            diff(p) != 0
        )
    )
    
    # Calculate total positive weight per score group.
    pos <- as.numeric(
        rowsum(
            w * (y == 1),
            group,
            reorder = FALSE
        )[, 1]
    )
    
    # Calculate total negative weight per score group.
    neg <- as.numeric(
        rowsum(
            w * (y == 0),
            group,
            reorder = FALSE
        )[, 1]
    )
    
    
    # --------------------------------------------------------
    # 01D — CALCULATE WEIGHTED AUROC
    # --------------------------------------------------------
    
    # Calculate cumulative negative weight below
    # each predicted-probability group.
    lower <- c(
        0,
        head(
            cumsum(neg),
            -1
        )
    )
    
    # Calculate the weighted probability that a
    # positive observation ranks above a negative one.
    #
    # Tied prediction scores receive half credit.
    return(
        sum(
            pos * (lower + 0.5 * neg)
        ) / (
            sum(pos) * sum(neg)
        )
    )
}


# ============================================================
# 02 — CALCULATE WEIGHTED AVERAGE PRECISION
# ============================================================

# Calculate weighted Average Precision (AP).
#
# Average Precision summarizes the precision-recall curve.
# It is useful when the positive outcome is less common than 
# the negative outcome.
#
# Interpretation:
#   Higher values indicate better precision-recall
#   performance.
#
# The calculation groups identical prediction scores
# before calculating cumulative weighted precision.

weighted_average_precision <- function(y, p, w) {
    
    # --------------------------------------------------------
    # 02A — VALIDATE INPUTS
    # --------------------------------------------------------
    
    valid <- (
        y %in% c(0, 1) &
            is.finite(p) &
            is.finite(w) &
            w > 0
    )
    
    # Retain eligible observations.
    y <- y[valid]
    p <- p[valid]
    w <- w[valid]
    
    # Precision-recall evaluation requires both classes.
    if (!any(y == 0) || !any(y == 1)) {
        return(NA_real_)
    }
    
    
    # --------------------------------------------------------
    # 02B — SORT PREDICTIONS
    # --------------------------------------------------------
    
    # Order predictions from highest to lowest.
    ix <- order(-p)
    
    y <- y[ix]
    p <- p[ix]
    w <- w[ix]
    
    
    # --------------------------------------------------------
    # 02C — GROUP IDENTICAL PREDICTIONS
    # --------------------------------------------------------
    
    group <- cumsum(
        c(
            TRUE,
            diff(p) != 0
        )
    )
    
    # Calculate positive outcome weight per score group.
    pos <- as.numeric(
        rowsum(
            w * (y == 1),
            group,
            reorder = FALSE
        )[, 1]
    )
    
    # Calculate total observation weight per score group.
    total <- as.numeric(
        rowsum(
            w,
            group,
            reorder = FALSE
        )[, 1]
    )
    
    
    # --------------------------------------------------------
    # 02D — CALCULATE AVERAGE PRECISION
    # --------------------------------------------------------
    
    # Calculate cumulative weighted precision and
    # average it across increases in recall.
    return(
        sum(
            (cumsum(pos) / cumsum(total)) * pos
        ) / sum(pos)
    )
}


# ============================================================
# 03 — SAFELY CALCULATE PROPORTIONS
# ============================================================

# Divide a numerator by a denominator while preventing
# division by zero or invalid denominators.
#
# Returns:
#   A numeric proportion when the denominator is valid.
#   NA when the denominator is missing or nonpositive.
#
# Used for classification metrics such as sensitivity,
# specificity, accuracy, and positive predictive value.

safe_fraction <- function(a, b) {
    
    # Return NA when the denominator is invalid.
    if (is.na(b) || b <= 0) {
        return(NA_real_)
    }
    
    return(
        a / b
    )
}


# ============================================================
# 04 — CALCULATE CLASSIFICATION PERFORMANCE METRICS
# ============================================================

# Evaluate a classification model using observed outcomes,
# predicted probabilities, and observation weights.
#
# Inputs:
#   y         = Observed FMD outcome (0 or 1)
#   p         = Predicted probability of FMD
#   w         = Evaluation weights
#   threshold = Probability cutoff for classification
#
# Metrics:
#   AUROC              = Discrimination
#   Average Precision  = Precision-recall performance
#   Accuracy           = Overall classification correctness
#   Brier Score        = Probability prediction error
#   Sensitivity        = Detection of positive FMD cases
#   Specificity        = Detection of negative FMD cases
#   PPV                = Precision of positive classifications
#   False Negative Rate = Proportion of FMD cases missed
#   Calibration Gap    = Mean predicted minus observed FMD
#
# Note:
#   n and events are unweighted respondent counts.
#   Other performance estimates use observation weights.

score_metrics <- function(
        y,
        p,
        w,
        threshold = CLASSIFICATION_THRESHOLD
) {
    
    # --------------------------------------------------------
    # 04A — VALIDATE PREDICTIONS
    # --------------------------------------------------------
    
    # Include valid binary outcomes and probabilities
    # between 0 and 1 with positive, finite weights.
    ok <- (
        y %in% c(0, 1) &
            is.finite(p) &
            valid_weight(w) &
            p >= 0 &
            p <= 1
    )
    
    # Retain eligible observations.
    y <- as.integer(y[ok])
    p <- as.numeric(p[ok])
    w <- as.numeric(w[ok])
    
    # Stop if no valid observations are available.
    if (!length(y)) {
        
        stop(
            "Evaluation group contains no valid predictions."
        )
    }
    
    
    # --------------------------------------------------------
    # 04B — CLASSIFY PREDICTED OUTCOMES
    # --------------------------------------------------------
    
    # Classify respondents as FMD positive when their
    # predicted probability meets or exceeds the threshold.
    positive <- p >= threshold
    
    
    # --------------------------------------------------------
    # 04C — CALCULATE WEIGHTED CONFUSION MATRIX
    # --------------------------------------------------------
    
    # True positives:
    #   Predicted FMD and observed FMD.
    tp <- sum(
        w[positive & y == 1]
    )
    
    # True negatives:
    #   Predicted no FMD and observed no FMD.
    tn <- sum(
        w[!positive & y == 0]
    )
    
    # False positives:
    #   Predicted FMD but observed no FMD.
    fp <- sum(
        w[positive & y == 0]
    )
    
    # False negatives:
    #   Predicted no FMD but observed FMD.
    fn <- sum(
        w[!positive & y == 1]
    )
    
    
    # --------------------------------------------------------
    # 04D — SUMMARIZE MODEL PERFORMANCE
    # --------------------------------------------------------
    
    return(
        data.frame(
            n = length(y),
            
            events = sum(y == 1),
            
            auroc = weighted_auc(
                y,
                p,
                w
            ),
            
            average_precision = weighted_average_precision(
                y,
                p,
                w
            ),
            
            accuracy = safe_fraction(
                tp + tn,
                tp + tn + fp + fn
            ),
            
            brier = stats::weighted.mean(
                (y - p)^2,
                w
            ),
            
            sensitivity = safe_fraction(
                tp,
                tp + fn
            ),
            
            specificity = safe_fraction(
                tn,
                tn + fp
            ),
            
            ppv = safe_fraction(
                tp,
                tp + fp
            ),
            
            false_negative_rate = safe_fraction(
                fn,
                fn + tp
            ),
            
            observed_fmd = stats::weighted.mean(
                y,
                w
            ),
            
            mean_prediction = stats::weighted.mean(
                p,
                w
            ),
            
            calibration_gap = stats::weighted.mean(
                p - y,
                w
            ),
            
            threshold = threshold
        )
    )
}


# ============================================================
# 05 — GENERATE PREDICTIONS FROM TRAINED MODELS
# ============================================================

# Generate predicted probabilities of FMD using one
# of the three trained classification models.
#
# Supported model families:
#   logistic = Logistic regression
#   lasso    = LASSO logistic regression
#   forest   = Random forest
#
# Returns:
#   Numeric predicted probabilities for each observation.
#
# The same predictor preparation must be used during
# training and prediction.

predict_model <- function(
        fit,
        family,
        dat,
        config = NA_real_
) {
    
    # --------------------------------------------------------
    # 05A — LOGISTIC REGRESSION
    # --------------------------------------------------------
    
    if (family == "logistic") {
        
        return(
            as.numeric(
                stats::predict(
                    fit,
                    newdata = dat,
                    type = "response"
                )
            )
        )
    }
    
    
    # --------------------------------------------------------
    # 05B — PREPARE NUMERIC PREDICTORS
    # --------------------------------------------------------
    
    # LASSO and random forest use the numeric design
    # matrix generated by the shared utility function.
    X <- make_matrix(dat)
    
    
    # ============================================================
    # LASSO — GENERATE PREDICTED FMD PROBABILITIES
    # ============================================================
    
    if (family == "lasso") {
        
        # Use method dispatch to return probabilities
        # rather than linear predictors (log-odds).
        result <- stats::predict(
            fit,
            newx = X,
            s = as.numeric(config),
            type = "response"
        )
        
        # Convert predictions into a numeric vector.
        p <- as.numeric(result)
        
        # Verify one prediction per respondent.
        if (length(p) != nrow(dat)) {
            
            stop(
                "LASSO prediction count does not match ",
                "the assessment sample."
            )
        }
        
        # Ensure all predictions are valid probabilities.
        if (
            any(!is.finite(p)) ||
            any(p < 0 | p > 1)
        ) {
            
            stop(
                "LASSO returned invalid predicted probabilities."
            )
        }
        
        return(p)
    }
    
    
    # --------------------------------------------------------
    # 05D — RANDOM FOREST
    # --------------------------------------------------------
    
    if (family == "forest") {
        
        # Generate the estimated probability of the
        # positive outcome class ("yes").
        return(
            as.numeric(
                predict(
                    fit,
                    data = data.frame(
                        X,
                        check.names = FALSE
                    ),
                    num.threads = RF_THREADS
                )$predictions[, "yes"]
            )
        )
    }
    
    
    # --------------------------------------------------------
    # 05E — HANDLE UNSUPPORTED MODEL FAMILIES
    # --------------------------------------------------------
    
    stop(
        "Unknown model family ",
        family
    )
}


# ============================================================
# 06 — FIT CLASSIFICATION MODELS
# ============================================================

# Train one of the three supported machine-learning models.
#
# Inputs:
#   family = Classification algorithm
#   dat    = Prepared training dataset
#   config = Model-specific tuning parameter
#
# The training weights are normalized before fitting.
#
# Model-specific parameters:
#   Logistic:
#     Binary outcome with quasibinomial regression.
#
#   LASSO:
#     L1-penalized logistic regression using lambda values.
#
#   Random Forest:
#     Probability forest with configurable mtry.
#
# Returns:
#   A fitted model object.

fit_model <- function(
        family,
        dat,
        config = NA_real_
) {
    
    # --------------------------------------------------------
    # 06A — PREPARE MODEL WEIGHTS
    # --------------------------------------------------------
    
    # Normalize previously calculated training weights.
    w <- model_weight(dat)
    
    
    # --------------------------------------------------------
    # 06B — LOGISTIC REGRESSION
    # --------------------------------------------------------
    
    if (family == "logistic") {
        
        # Store weights directly in the model data
        # to avoid formula-environment lookup issues.
        dat$MODEL_WEIGHT <- w
        
        # Fit weighted logistic regression.
        fit <- stats::glm(
            formula = MODEL_FORMULA,
            data = dat,
            weights = MODEL_WEIGHT,
            family = stats::quasibinomial(),
            control = stats::glm.control(
                maxit = 35
            )
        )
        
        # Verify that model optimization converged.
        if (!isTRUE(fit$converged)) {
            
            stop(
                "Logistic fit did not converge."
            )
        }
        
        return(fit)
    }
    
    
    # --------------------------------------------------------
    # 06C — PREPARE NUMERIC DESIGN MATRIX
    # --------------------------------------------------------
    
    # Convert predictors to numeric columns for
    # LASSO and random forest fitting.
    X <- make_matrix(dat)
    
    
    # --------------------------------------------------------
    # 06D — LASSO LOGISTIC REGRESSION
    # --------------------------------------------------------
    
    if (family == "lasso") {
        
        # Fit L1-penalized logistic regression using
        # the predefined regularization parameter grid.
        return(
            glmnet::glmnet(
                x = X,
                y = dat$FMD,
                weights = w,
                family = "binomial",
                alpha = 1,
                lambda = LAMBDA_GRID,
                standardize = TRUE
            )
        )
    }
    
    
    # --------------------------------------------------------
    # 06E — RANDOM FOREST
    # --------------------------------------------------------
    
    if (family == "forest") {
        
        # Convert the binary outcome to a labeled factor.
        # This allows ranger to build a probability forest.
        forest_data <- data.frame(
            target = factor(
                dat$FMD,
                c(0, 1),
                c("no", "yes")
            ),
            X,
            check.names = FALSE
        )
        
        # Fit the weighted random forest classifier.
        return(
            ranger::ranger(
                dependent.variable.name = "target",
                data = forest_data,
                probability = TRUE,
                num.trees = RF_TREES,
                mtry = min(
                    as.integer(config),
                    ncol(X)
                ),
                min.node.size = 20,
                importance = "impurity",
                case.weights = w,
                num.threads = RF_THREADS,
                seed = SEED,
                write.forest = TRUE
            )
        )
    }
    
    
    # --------------------------------------------------------
    # 06F — EDGE CASE: HANDLE UNSUPPORTED MODEL FAMILIES
    # --------------------------------------------------------
    
    stop(
        "Unknown model family ",
        family
    )
}
