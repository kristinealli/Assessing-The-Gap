# Assessing the Gap — 15-minute presentation outline

This plan maps directly to the four components of the assignment.
Use the generated tables and figures, not notebook development logs.

| Slide | Time | Content | Suggested evidence |
|---|---:|---|---|
| 1. The problem and primary question | 1:30 | Does national model performance generalize equally to AI/AN-alone respondents? | One-sentence primary research question |
| 2. Motivation and prior research | 1:30 | FMD, AI/AN surveillance, and algorithmic evaluation | CDC literature and study gap |
| 3. Data and operational definitions | 1:30 | 2014–2024 BRFSS; 14+ days; race-group exclusions | Annual sample audit |
| 4. Pre/post FMD pattern | 2:00 | Compare prevalence and changes, descriptive only | Annual FMD trend figure |
| 5. Missingness and covariates | 1:30 | Exclusions and predictors | Retention and prevalence tables |
| 6. PCA | 2:00 | Explain dummy-coded predictor structure, not outcome prediction | PCA chart and variance table |
| 7. Models and common CV | 2:00 | Logistic, LASSO, RF; common folds and performance measures | CV winning-configurations table |
| 8. Who does each model predict well? | 2:00 | AI/AN vs comparison in each period | AUROC-by-group-period graphic |
| 9. Interpretation and limitations | 1:00 | Calibration, race-category limitations, weights, uncertainty | Calibration plot or key findings |

## Speaker-note guardrails

- Call AI/AN *alone* by that name; excluded multiracial respondents may include AI/AN identities.
- Do not imply 2020 caused the changes: BRFSS is a repeated cross-sectional survey.
- Explain AUROC on a 0–1 scale, with 0.5 representing chance ranking.
- The sample's large size does not make every difference statistically significant.
- The class imbalance and 0.20 decision threshold affect sensitivity and specificity.
- Present sensitivity analyses as robustness checks, not proof of representativeness.
- Keep source-level FMD-category discrepancies and limits on pooling-weight inference visible.
