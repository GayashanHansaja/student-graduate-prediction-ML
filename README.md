# Flood Prediction ML Assignment Repository

This repository is prepared for the **Machine Learning group assignment** and is structured to satisfy all required submission components.

## Assignment Scope

- Problem domain: **Flood prediction using machine learning**
- Learning type: **Supervised / Unsupervised (No deep learning)**
- Team size: **4 members**
- Required algorithm count for supervised/unsupervised: **4 distinct algorithms**

## Repository Structure

```text
Flood-predection-ML/
├── members.txt
├── submission.txt
├── README.md
├── notebooks/
│   ├── 01_member1_linear_regression.ipynb
│   ├── 02_member2_random_forest.ipynb
│   ├── 03_member3_svr.ipynb
│   └── 04_member4_knn.ipynb
├── report/
│   └── report_template.md
├── data/
│   └── README.md
└── references/
    └── data_source_citation.md
```

## Deliverables Checklist

Use this checklist before final submission:

- [ ] `members.txt` contains all member IDs and emails.
- [ ] `submission.txt` contains:
  - [ ] dataset link
  - [ ] public GitHub repository link
  - [ ] YouTube presentation/demo link (max 20 min)
- [ ] Four distinct ML algorithms implemented in notebooks (no deep learning).
- [ ] PDF report completed with methodology, results, comparison, limitations/future work.
- [ ] Appendix in report includes **all source code as text** (not screenshots).
- [ ] Final zip named **ML-assignment.zip** contains all required files.

## Suggested Team Allocation (4 Members)

- Member 1: Linear Regression baseline + preprocessing
- Member 2: Random Forest
- Member 3: Support Vector Regression (SVR)
- Member 4: K-Nearest Neighbors (KNN)

## How to Work

1. Add dataset files to `data/` (or document external source if not committed).
2. Record exact dataset source in `references/data_source_citation.md`.
3. Each member develops and commits their notebook.
4. Compare model performance using appropriate metrics (e.g., MAE, RMSE, R² for regression).
5. Fill `submission.txt` and `members.txt` before submission.
6. Generate final report PDF from `report/report_template.md` content.

## Notes

- Keep commit messages detailed so GitHub history clearly shows each member's contribution.
- Do not use deep learning models for this assignment.
- Ensure all links in `submission.txt` are publicly accessible.
