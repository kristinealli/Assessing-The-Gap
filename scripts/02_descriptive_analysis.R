
# ============================================================
# ASSESSING THE GAP — DESCRIPTIVE ANALYSIS
# ============================================================
# Purpose:
#   Examine frequent mental distress (FMD) prevalence,
#   population representation, and temporal trends among
#   AI/AN-alone and other identified single-race adults.
#
# Research Aim 1:
#   Compare the prevalence of frequent mental distress
#   between population groups and survey periods.
#
# Dataset:
#   BRFSS 2014–2024
#
# Study Periods:
#   Pre-2020  = 2014–2019
#   Post-2020 = 2020–2024
#
# Weighting:
#   Primary estimates use the historical POOL_WEIGHT.
#
#   Sensitivity analyses compare:
#     1. Historical pooled survey weights.
#     2. Original annual survey weights.
#     3. Equal-weight averages of annual prevalence.
#
# Important:
#   These are descriptive estimates from repeated
#   cross-sectional surveys.
#
#   The script does not calculate complex-survey
#   confidence intervals or establish causal changes.
# 
# This script produces: 
# Respondent counts and complete-case availability:
#   outputs/tables/02_group_retention.csv
# FMD prevalence and weighting sensitivity:
#   outputs/tables/02_prevalence_by_group_period.csv
# Annual group-specific FMD prevalence:
#   outputs/tables/02_annual_prevalence.csv
# Changes in prevalence between pre- and post-2020 periods:
#   outputs/tables/02_pre_post_change.csv
# Supplementary age-standardized estimate:
#   outputs/tables/02_age_standardized_sensitivity.csv
# Line chart of annual FMD prevalence trends, 2014 - 2024:
#   outputs/figures/02_annual_fmd_trend.png
# ============================================================


# ============================================================
# 01 — LOAD PROJECT UTILITIES AND DATA
# ============================================================

# Import shared configuration, validation functions,
# and output utilities.

source(
    file.path(
        "R",
        "01_utilities.R"
    )
)

# Verify that the required packages are available.
check_packages(
    c(
        "data.table",
        "ggplot2"
    )
)

# Load data.table without displaying startup messages.
suppressPackageStartupMessages(
    library(data.table)
)

# Load the existing pooled BRFSS dataset.
d <- load_data()


# ============================================================
# 02 — DEFINE STUDY POPULATIONS AND PERIODS
# ============================================================

# Convert the existing AI/AN-alone indicator into
# readable population group labels.
#
# TRUE:
#   AI/AN alone
#
# FALSE:
#   Other identified single-race
#
# NA:
#   Multiracial, unknown, or excluded classification
#
# Important:
#   The comparison group is not an exhaustive definition
#   of people without Indigenous ancestry.

d[
    ,
    cohort := two_group(AIAN_ALONE)
]


# ------------------------------------------------------------
# 02A — ASSIGN SURVEY PERIODS
# ------------------------------------------------------------

# Divide survey years into pre-2020 and post-2020 periods.
#
# Pre-2020:
#   2014–2019
#
# Post-2020:
#   2020–2024

d[
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


# ------------------------------------------------------------
# 02B — IDENTIFY ELIGIBLE OUTCOMES
# ------------------------------------------------------------

# Identify observations eligible for weighted prevalence.
#
# Requirements:
#   Valid binary FMD outcome (0 or 1).
#   Positive, finite pooled survey weight.
#   Identified primary comparison group.

d[
    ,
    valid_outcome := (
        FMD %in% c(0, 1) &
            valid_weight(POOL_WEIGHT) &
            !is.na(cohort)
    )
]


# ============================================================
# 03 — ASSESS SAMPLE RETENTION
# ============================================================

# Summarize sample availability by study period and
# population classification.
#
# Measures:
#   n:
#     Total respondents.
#
#   outcome_valid:
#     Respondents with valid FMD outcomes.
#
#   complete_predictors:
#     Respondents with valid FMD outcomes and complete
#     values for all model predictors.
#
# Missing or excluded race classifications are retained
# in this audit as an NA cohort category.

retention <- d[
    ,
    .(
        n = .N,
        
        outcome_valid = sum(
            FMD %in% c(0, 1)
        ),
        
        complete_predictors = sum(
            FMD %in% c(0, 1) &
                stats::complete.cases(.SD)
        )
    ),
    by = .(
        study_period,
        cohort
    ),
    .SDcols = FEATURES
]

# Save the sample retention summary.
save_csv(
    retention,
    "02_group_retention.csv"
)


# ============================================================
# 04 — CALCULATE WEIGHTED FMD PREVALENCE
# ============================================================

# Retain respondents who meet all eligibility criteria:
#   Valid FMD outcome (0 or 1).
#   Positive, finite survey weight.
#   Eligible population classification.

eligible <- d[
    valid_outcome == TRUE
]


# ------------------------------------------------------------
# 04A — PREVALENCE BY GROUP AND PERIOD
# ------------------------------------------------------------

# Calculate primary and alternative weighted prevalence.
#
# Primary:
#   POOL_WEIGHT
#
# Alternative:
#   RAW_WEIGHT
#
# Prevalence estimates are multiplied by 100
# to express the results as percentages.

rates <- eligible[
    ,
    .(
        respondents = .N,
        
        cases = sum(
            FMD == 1
        ),
        
        prevalence_pct = 100 * weighted_rate(
            FMD,
            POOL_WEIGHT
        ),
        
        raw_weight_pct = 100 * weighted_rate(
            FMD,
            RAW_WEIGHT
        )
    ),
    by = .(
        study_period,
        cohort
    )
][
    order(study_period, cohort)
]


# ============================================================
# 05 — CALCULATE ANNUAL FMD PREVALENCE
# ============================================================

# Estimate annual FMD prevalence separately for
# each population group.
#
# Annual estimates use RAW_WEIGHT because each year
# represents a separate BRFSS survey.
#
# Multiplying all weights within a year by the same
# positive constant does not change that year's
# weighted prevalence estimate.

annual <- eligible[
    ,
    .(
        respondents = .N,
        
        cases = sum(
            FMD == 1
        ),
        
        prevalence_pct = 100 * weighted_rate(
            FMD,
            RAW_WEIGHT
        )
    ),
    by = .(
        YEAR,
        cohort
    )
][
    order(YEAR, cohort)
]


# ============================================================
# 06 — COMPARE WEIGHTING APPROACHES
# ============================================================

# Evaluate whether estimated FMD prevalence changes
# when different multi-year weighting methods are used.
#
# Three estimates are compared:
#
#   prevalence_pct:
#     Historical pooled-weight estimate.
#
#   raw_weight_pct:
#     Estimate using the original annual survey weights
#     pooled across respondents.
#
#   equal_year_mean_pct:
#     Arithmetic mean of the annual weighted estimates,
#     giving each survey year equal importance.
#
# These methods represent different weighting choices
# and should not be treated as identical estimands.


# ------------------------------------------------------------
# 06A — CALCULATE EQUAL-YEAR PREVALENCE
# ------------------------------------------------------------

# Assign each annual estimate to its corresponding
# study period and calculate the mean prevalence.

equal_year <- annual[
    ,
    .(
        equal_year_mean_pct = mean(
            prevalence_pct
        )
    ),
    by = .(
        study_period = ifelse(
            YEAR < 2020,
            "Pre-2020",
            "Post-2020"
        ),
        cohort
    )
]


# ------------------------------------------------------------
# 06B — COMBINE WEIGHTING ESTIMATES
# ------------------------------------------------------------

# Merge the alternative estimates with the primary
# prevalence table.

rates <- merge(
    rates,
    equal_year,
    by = c(
        "study_period",
        "cohort"
    )
)


# ------------------------------------------------------------
# 06C — MEASURE WEIGHTING SENSITIVITY
# ------------------------------------------------------------

# Calculate the largest absolute difference between
# the primary pooled-weight estimate and either
# alternative estimate.
#
# Units:
#   Percentage points (pp).
#
# Smaller differences indicate that the descriptive
# prevalence estimate is less sensitive to these
# particular weighting choices.

rates[
    ,
    largest_sensitivity_difference_pp := pmax(
        abs(
            prevalence_pct - raw_weight_pct
        ),
        abs(
            prevalence_pct - equal_year_mean_pct
        )
    )
]


# ============================================================
# 07 — CALCULATE PRE/POST PREVALENCE CHANGES
# ============================================================

# Reshape the prevalence results so the two study
# periods appear as separate columns.
#
# Each row represents one population group.

wide <- data.table::dcast(
    rates,
    cohort ~ study_period,
    value.var = "prevalence_pct"
)

# Calculate the percentage-point change from
# pre-2020 to post-2020.
#
# Positive value:
#   FMD prevalence increased.
#
# Negative value:
#   FMD prevalence decreased.
#
# This describes differences between survey periods,
# not changes observed within the same individuals.

wide[
    ,
    change_pp := `Post-2020` - `Pre-2020`
]


# ============================================================
# 08 — DISPLAY DESCRIPTIVE RESULTS
# ============================================================

# ------------------------------------------------------------
# 08A — WEIGHTED PREVALENCE BY PERIOD
# ------------------------------------------------------------

cat(
    "\n[WEIGHTED PREVALENCE BY PERIOD]\n"
)

print(
    rates[
        ,
        .(
            study_period,
            cohort,
            respondents,
            cases,
            prevalence_pct,
            raw_weight_pct,
            equal_year_mean_pct,
            largest_sensitivity_difference_pp
        )
    ]
)


# ------------------------------------------------------------
# 08B — PRE/POST COMPARISON
# ------------------------------------------------------------

cat(
    "\n[PRE/POST COMPARISON]\n"
)

print(wide)


# ------------------------------------------------------------
# 08C — INTERPRETATION AND LIMITATIONS
# ------------------------------------------------------------

# Explain the repeated cross-sectional study design.
cat(
    "\n[READING THE TABLE] Percentage-point changes describe ",
    "differences between repeated cross-sectional samples, not ",
    "changes within individuals.\n",
    sep = ""
)

# Clarify the absence of design-based uncertainty estimates.
cat(
    "[CAUTION] No complex-survey confidence intervals are ",
    "calculated by this script.\n",
    sep = ""
)


# ============================================================
# 09 — SAVE DESCRIPTIVE ANALYSIS TABLES
# ============================================================

# Save weighted FMD prevalence by group and period.
save_csv(
    rates,
    "02_prevalence_by_group_period.csv"
)

# Save annual FMD prevalence for temporal analysis.
save_csv(
    annual,
    "02_annual_prevalence.csv"
)

# Save percentage-point changes between periods.
save_csv(
    wide,
    "02_pre_post_change.csv"
)


# ============================================================
# 10 — VISUALIZE ANNUAL FMD TRENDS
# ============================================================

# Create a line chart showing annual weighted FMD
# prevalence for the two study populations.
#
# Plot:
#   X-axis = Survey year
#   Y-axis = Weighted FMD prevalence (%)
#   Color  = Population group
#
# A dashed vertical line separates the pre-2020
# and post-2020 study periods.


# ------------------------------------------------------------
# 10A — CREATE THE PLOT
# ------------------------------------------------------------

p <- ggplot2::ggplot(
    annual,
    ggplot2::aes(
        x = YEAR,
        y = prevalence_pct,
        color = cohort
    )
) +
    
    # Connect annual estimates with lines.
    ggplot2::geom_line(
        linewidth = 1
    ) +
    
    # Mark individual annual estimates.
    ggplot2::geom_point(
        size = 2
    ) +
    
    # Mark the boundary between study periods.
    ggplot2::geom_vline(
        xintercept = 2019.5,
        linetype = "dashed",
        color = "gray55"
    ) +
    
    # Display every study year on the X-axis.
    ggplot2::scale_x_continuous(
        breaks = STUDY_YEARS
    ) +
    
    # Define chart titles, labels, and interpretation notes.
    ggplot2::labs(
        title = "Frequent mental distress, 2014–2024",
        
        subtitle = paste(
            "Annual BRFSS weighted prevalence;",
            "descriptive estimates"
        ),
        
        x = "Survey year",
        
        y = "Respondents reporting 14+ days (%)",
        
        color = "Population",
        
        caption = paste(
            "Each point is one annual estimate.",
            "Dashed line marks the start of 2020.",
            "No complex-survey confidence intervals are shown."
        )
    ) +
    
    # Apply consistent visual styling.
    ggplot2::theme_minimal(
        base_size = 12
    )


# ------------------------------------------------------------
# 10B — DISPLAY AND SAVE THE FIGURE
# ------------------------------------------------------------

# Display the figure in RStudio.
print(p)

# Export the figure as a PNG file.
save_figure(
    p,
    "02_annual_fmd_trend.png",
    width = 10
)


# ============================================================
# 11 — AGE-STANDARDIZED SENSITIVITY ANALYSIS
# ============================================================

# Perform a supplementary age-standardized comparison.
#
# Purpose:
#   Examine whether differences in the age composition
#   of study populations affect FMD prevalence estimates.
#
# Method:
#   Direct standardization to a common age distribution
#   calculated from the pooled eligible sample.
#
# Important:
#   This is a descriptive sensitivity analysis.
#   It is not an official age-adjusted surveillance
#   statistic based on an external standard population.


# ------------------------------------------------------------
# 11A — RETAIN VALID AGE CATEGORIES
# ------------------------------------------------------------

# Exclude observations with missing age categories.
age_data <- eligible[
    !is.na(AGE_BAND)
]


# ------------------------------------------------------------
# 11B — CALCULATE REFERENCE AGE DISTRIBUTION
# ------------------------------------------------------------

# Sum pooled survey weights within each age category.
age_weights <- age_data[
    ,
    .(
        weight_sum = sum(
            POOL_WEIGHT
        )
    ),
    by = AGE_BAND
]

# Convert age-category weight totals into proportions.
#
# The proportions form a common reference distribution
# shared across population groups and study periods.
age_weights[
    ,
    reference := weight_sum / sum(weight_sum)
]


# ------------------------------------------------------------
# 11C — ESTIMATE AGE-SPECIFIC FMD PREVALENCE
# ------------------------------------------------------------

# Calculate weighted FMD prevalence separately by:
#   Study period
#   Population group
#   Age category

age_cells <- age_data[
    ,
    .(
        age_prevalence = weighted_rate(
            FMD,
            POOL_WEIGHT
        )
    ),
    by = .(
        study_period,
        cohort,
        AGE_BAND
    )
]


# ------------------------------------------------------------
# 11D — ATTACH REFERENCE AGE WEIGHTS
# ------------------------------------------------------------

# Merge age-specific prevalence estimates with
# the common reference age distribution.

age_cells <- merge(
    age_cells,
    age_weights[
        ,
        .(
            AGE_BAND,
            reference
        )
    ],
    by = "AGE_BAND",
    all.x = TRUE
)


# ------------------------------------------------------------
# 11E — CALCULATE AGE-STANDARDIZED PREVALENCE
# ------------------------------------------------------------

# Multiply each age-specific prevalence by its
# reference population proportion.
#
# Sum the weighted estimates and multiply by 100
# to express age-standardized prevalence as a percentage.

age_standardized <- age_cells[
    ,
    .(
        age_standardized_pct = 100 * sum(
            reference * age_prevalence
        )
    ),
    by = .(
        study_period,
        cohort
    )
]


# ------------------------------------------------------------
# 11F — SAVE AGE-STANDARDIZED RESULTS
# ------------------------------------------------------------

save_csv(
    age_standardized,
    "02_age_standardized_sensitivity.csv"
)


# ============================================================
# 12 — RELEASE MEMORY AND REPORT COMPLETION
# ============================================================

# Remove large intermediate datasets from memory.
rm(
    d,
    eligible,
    age_data
)

# Request garbage collection to release unused memory.
gc(
    verbose = FALSE
)

# Confirm successful completion of the descriptive analyses.
message(
    "[COMPLETE] Descriptive analyses saved."
)
