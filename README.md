# Assessing the Gap
### Frequent Mental Distress Trends and Predictive Model Performance Among American Indian/Alaska Native Adults, BRFSS 2014–2024

An **RStudio-native, reproducible research project** based on an 11-year BRFSS pooled analytical dataset. 

## Start here (5 steps)

1. Unzip the project and open **`Assessing_the_Gap.Rproj`** in RStudio.
2. Confirm this file exists: **`data/processed/analysis_pooled_2014_2024.rds`**. 
3. Run `source("scripts/00_install_packages.R")` **once**. RStudio's R installation needs internet access to install any missing packages.
4. Optionally run `source("tests/smoke_test.R")` to check metric and encoding helpers, then run `source("run_all.R")` for the required research pipeline, or run the scripts individually in numerical order. The model-training steps are computationally intensive, so completed checkpoints are reused where possible.
5. Read `reports/Assessing_the_Gap_Report.html` after it renders. 

## What is in this folder?

```text
Assessing_the_Gap_RStudio_Project/
├── Assessing_the_Gap.Rproj
├── README.md
├── run_all.R
├── R/
│   ├── 00_config.R                  # Study settings and output directories
│   ├── 01_utilities.R               # RDS loading, common data transformations
│   └── 02_metrics.R                 # Weighted AUROC and other evaluation metrics
├── tests/smoke_test.R              # Small unit tests for metrics and encoding
├── scripts/
│   ├── 00_install_packages.R
│   ├── 01_audit_pooled_data.R
│   ├── 02_descriptive_analysis.R
│   ├── 03_prepare_model_samples.R
│   ├── 04_pca.R
│   ├── 05_associations.R
│   ├── 06_cross_validation.R
│   ├── 07_fit_final_models.R
│   ├── 08_score_heldout_years.R
│   ├── 09_evaluate_and_visualize.R
│   ├── 10_conditional_bootstrap.R   
│   ├── 11_render_report.R
│   └── 12_session_information.R
├── data/processed/
│   ├── analysis_pooled_2014_2024.rds
│   └── analysis_pooled_2014_2024.meta.rds
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── models/                      # Trained models, fit locally
│   ├── checkpoints/                 # CV jobs and annual predictions
│   └── logs/
└── reports/
    ├── Final_Research_Report.Rmd    # Renderable stakeholder research report
    └── Presentation_Outline.md
```

## Research aims and assignment mapping

| Project component | R scripts | Output |
|---|---|---|
| Research question and scientific background | `reports/Final_Research_Report.Rmd` | Study report |
| Descriptive tables, trends, missingness | `01`, `02` | Prevalence, sample retention, annual trend |
| PCA using dummy-coded predictors | `04` | PCA chart, loadings, explained variance |
| Three supervised classifiers | `06`, `07` | Logistic, LASSO logistic, random forest |
| Shared cross-validation and tuning | `03`, `06` | CV tables and winning configurations |
| Held-out subgroup comparisons | `08`, `09` | AUROC, PR precision, Brier, calibration, class rates |
| Associations and interpretations | `05`, `09` | Exploratory coefficients, predicted probabilities |
| Final stakeholder communication | `11` + R Markdown | Rendered HTML report and 15-minute outline |

## Population and variable definitions

- **Outcome:** FMD = 14+ mentally unhealthy days in the last 30 days, from `MENTHLTH`. The source code `88` means *zero*, while `77` and `99` are missing.
- **AI/AN group:** `AIAN_ALONE == TRUE` (identified AI/AN alone).
- **Comparison group:** `AIAN_ALONE == FALSE` (other identified single-race categories).
- **Excluded:** Multiracial/unknown calculated race-alone categories (`AIAN_ALONE == NA`). The comparison is operational due to multiracial coding not dictating the AI/AN status. The AI/AN group is the primary focus of this research.
- **Predictors:** physically unhealthy days, general health, age band, recorded sex, education, harmonized nominal income band, period.
- **Periods:** 2014–2019 vs 2020–2024. This is a repeated cross-sectional comparison, NOT within-person change or a causal evaluation of the pandemic.

## Original dataset and weighting choices

The provided analytical RDS contains **4,790,742 records and 25 variables**. Script `01` checks those counts and the internal FMD/race coding again on the transferred copy.

The existing `POOL_WEIGHT` is constructed using a historical rule from compiling 11 years of data:

`RAW_WEIGHT × (number of eligible respondents in that survey year / pooled eligible respondents)`

It is retained here as the **primary analytical weight**. 

`02_descriptive_analysis.R` compares:
- historical `POOL_WEIGHT`,
- raw annual survey weights,
- an equal-weight mean of annual group-specific weighted prevalence estimates.

These are intended to provide a benchmark for the primary analytical weight. 

## Model-training and leakage safeguards

- All 11 years contribute to training and held-out evaluation.
- An independent test set is selected **within year × race-group × FMD strata**.
- CV uses **the same three folds** for logistic, LASSO and random forest.
- Hyperparameters are selected using cross-validation only.
- No held-out outcomes enter fitting or tuning.
- The much larger comparison group is sampled for computational efficiency; probabilities are accounted for in `ML_WEIGHT`.
- The classification threshold is fixed at **0.20** before test evaluation. Sensitivity/specificity depend on this choice.
- FMD is never included in the PCA input features.


## Run only selected sections

For a quick look at the supplied dataset:

```r
source("scripts/01_audit_pooled_data.R")
source("scripts/02_descriptive_analysis.R")
```

To complete machine learning (in order):

```r
source("scripts/03_prepare_model_samples.R")
source("scripts/04_pca.R")
source("scripts/05_associations.R")
source("scripts/06_cross_validation.R")
source("scripts/07_fit_final_models.R")
source("scripts/08_score_heldout_years.R")
source("scripts/09_evaluate_and_visualize.R")
source("scripts/11_render_report.R")
```

Optional fixed-prediction bootstrap (computationally expensive; **not** a BRFSS design-based CI):

```r
source("scripts/10_conditional_bootstrap.R")
```

## Checkpoint and resource notes

- The full original `.rds` file is roughly **108 MB compressed** but can expand substantially in RAM.
- RStudio should run with sufficient free memory; a modern 16 GB machine is preferable for simultaneous data and model objects.
- During long scripts, large objects are released as possible. Run each script separately if memory pressure appears.
- Completed CV folds, final models, and year-specific predictions are saved to `outputs/`.
- **Do not change sample caps, seed, weights, factor encoding, or model definitions while reusing old checkpoints.** If settings change, archive/reset `outputs/checkpoints` and `outputs/models` before rerunning.
- The project uses local, relative file paths. Confirm proper for your environment.
