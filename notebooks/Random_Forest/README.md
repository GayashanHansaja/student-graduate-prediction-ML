# Random Forest Classifier - Student Graduate vs Dropout Prediction

## Student Details
- Name: Sanjeewa P.D.L.B
- Student ID: IT22629708

## Overview
This directory contains the Random Forest implementation for predicting student outcomes (`Graduate` vs. `Dropout`). The objective is to classify students based on their academic and demographic background. The dataset is preprocessed to remove students with the `Enrolled` target so the problem remains a strict binary classification task.

The folder includes:
- a compact notebook with the full pipeline in one place,
- a presentation-style notebook with explanations and saved outputs,
- and a serialized trained model for reuse in future deployment work.

## Folder Structure
```text
Random_Forest/
|-- README.md
|-- random_forest.ipynb
|-- Random_Tree_Forest.ipynb
\-- Modal/
    \-- Random_Forest_Model.pkl
```

## File Descriptions
### `Random_Tree_Forest.ipynb` (Presentation Version)
This is the detailed notebook version of the project. It is organized as a step-by-step walkthrough with markdown explanations and code cells.

It contains:
- 14 total cells
- 7 markdown cells
- 7 code cells
- executed outputs, including printed evaluation results and visualization previews

This notebook covers the complete workflow:
- importing libraries,
- loading and preprocessing the dataset,
- plotting the target distribution,
- splitting the data into training and testing sets,
- training the Random Forest model,
- evaluating predictions,
- visualizing feature importance,
- and exporting the trained model.

### `random_forest.ipynb` (Compact Version)
This is the streamlined notebook version. It contains the entire Random Forest workflow in a single code cell.

It is useful for:
- quick review of the final code,
- compact submission format,
- and simple reruns without explanatory markdown sections.

### `Modal/Random_Forest_Model.pkl`
This file is the serialized trained Random Forest model saved using `joblib`. Its approximate size is 4.6 MB.

This model file can be reused for:
- loading the trained classifier without retraining,
- future web interface integration,
- and downstream prediction workflows.

## Workflow Summary
The implementation follows this pipeline:

1. Load the dataset using semicolon-separated formatting.
2. Check for null values in the dataset.
3. Remove rows where the target is `Enrolled`.
4. Separate input features (`X`) and target labels (`y`).
5. Visualize the target class distribution.
6. Split the data using an 80-20 train-test split with `stratify=y`.
7. Train a `RandomForestClassifier` with 100 estimators and `random_state=42`.
8. Evaluate the trained model using multiple classification metrics.
9. Plot the confusion matrix.
10. Plot the top 10 most important features.
11. Export the trained model as `Random_Forest_Model.pkl`.

## Model Performance & Results
The Random Forest model was trained with 100 estimators and evaluated using an 80-20 stratified train-test split.

### Key Metrics
- Test Accuracy: 91.46% (`0.9146`)
- MCC Score: `0.8215`

### Classification Report Snapshot
- Dropout:
  precision `0.95`, recall `0.82`, F1-score `0.88`, support `284`
- Graduate:
  precision `0.90`, recall `0.97`, F1-score `0.93`, support `442`
- Overall accuracy:
  `0.91` across `726` samples

### Visualizations Included
- Target Variable Distribution
- Confusion Matrix
- Top 10 Feature Importances

## Requirements
To run the notebooks in this directory, install the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`

## Execution Notes
The notebooks expect the dataset to be available as:

```python
df = pd.read_csv('data.csv', delimiter=';')
```

Before running the notebooks, make sure `data.csv` is available in the correct working location for your notebook session.

The project also includes:
- target distribution visualization for class balance inspection,
- confusion matrix visualization for classification analysis,
- and feature importance visualization for identifying the strongest predictors of student dropout.

## Output Summary
When executed successfully, the notebook produces:
- printed accuracy and MCC results,
- a full classification report,
- a confusion matrix plot,
- a top-10 feature importance chart,
- and a saved model file named `Random_Forest_Model.pkl`.

## Conclusion
This folder represents the full Random Forest workflow for the student graduate vs. dropout prediction task. It includes both a compact implementation and a presentation-style version, along with a saved trained model for reuse.
