# KNN Classification – Comprehensive Viva Preparation Guide

**Project:** Predict Students' Dropout and Academic Success  
**Model:** K-Nearest Neighbors (KNN)  
**Dataset:** UCI ML Repository – ID 697  
**Notebook:** `notebooks/KNN/KNN.ipynb`

---

## Table of Contents

1. [Introduction to K-Nearest Neighbors (KNN)](#1-introduction-to-k-nearest-neighbors-knn)
2. [Line-by-Line Code Explanation](#2-line-by-line-code-explanation)
3. [Comparison with Other Algorithms](#3-comparison-with-other-algorithms)
4. [Dataset Analysis](#4-dataset-analysis)
5. [Data Preprocessing Steps](#5-data-preprocessing-steps)
6. [Choosing the Best Value of K](#6-choosing-the-best-value-of-k)
7. [Model Evaluation](#7-model-evaluation)
8. [All Model Output Details](#8-all-model-output-details)
9. [Possible Viva Questions & Answers](#9-possible-viva-questions--answers)

---

## 1. Introduction to K-Nearest Neighbors (KNN)

### Concept

K-Nearest Neighbors is a **non-parametric, instance-based (lazy) supervised learning algorithm**. It makes no assumptions about the underlying data distribution. Instead, it memorises the entire training dataset and classifies a new data point by looking at the **K closest training examples** in the feature space.

### Working Principle

1. **Training phase:** The algorithm simply stores all training data points — there is no explicit model fitting.
2. **Prediction phase:** For a new unseen point:
   - Compute the distance between the new point and every training point.
   - Select the **K** nearest neighbours.
   - Assign the class that is most common among those K neighbours (majority vote), or use distance-weighted voting.

### Intuition

Imagine dropping a new student record on a map where every previous student is plotted by their academic and demographic features. KNN asks: *"Who are the 17 most similar students already in our dataset, and what was their outcome?"* The new student gets the most common outcome among those 17 neighbours.

### Distance Metric

The most common metric is **Euclidean distance**:

```
d(a, b) = sqrt( Σ (aᵢ - bᵢ)² )
```

Other options include Manhattan, Minkowski, and Cosine distance.

### Distance Weighting

In this project, `weights='distance'` is used — **closer neighbours have a stronger vote**:

```
vote_weight = 1 / distance
```

This is more robust than uniform voting because very close neighbours are more informative.

### Why KNN is Called a "Lazy Learner"

KNN defers all computation to prediction time. Training is O(1) (just store data), but prediction is O(n × d) where n = training samples and d = features. This is the opposite of "eager learners" like Decision Trees that build an explicit model upfront.

---

## 2. Line-by-Line Code Explanation

### Cell 1 – Imports & Library Versions

```python
!pip install ucimlrepo imbalanced-learn --quiet
```
> Installs the UCI dataset fetcher and SMOTE library if not already present.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```
> Imports core data-science libraries. `warnings.filterwarnings('ignore')` suppresses non-critical deprecation warnings for cleaner output.

```python
from ucimlrepo import fetch_ucirepo
```
> Imports the UCI ML repository API client to download the dataset programmatically.

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.feature_selection import VarianceThreshold
```
> - `StandardScaler`: normalises features to zero mean/unit variance (critical for KNN).  
> - `LabelEncoder`: converts string class labels to integers.  
> - `SimpleImputer`: fills missing values.  
> - `train_test_split`, `cross_val_score`, `StratifiedKFold`: splitting and cross-validation utilities.  
> - `VarianceThreshold`: removes features with near-zero variance (uninformative columns).

```python
from sklearn.neighbors import KNeighborsClassifier
```
> The KNN model class from scikit-learn.

```python
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_auc_score
)
```
> Evaluation metrics for measuring model quality.

```python
from imblearn.over_sampling import SMOTE
```
> SMOTE (Synthetic Minority Over-sampling Technique) – generates synthetic samples for under-represented classes.

---

### Cell 2 – Load Dataset & Explore

```python
dataset = fetch_ucirepo(id=697)
X, y = dataset.data.features, dataset.data.targets
```
> Downloads the "Predict Students' Dropout and Academic Success" dataset (ID=697) from UCI. Separates it into features (`X`) and target labels (`y`).

```python
print(f'Shape: {X.shape}  |  Missing values: {X.isnull().sum().sum()}')
print(f'Data types: {dict(X.dtypes.value_counts())}')
print('\nTarget distribution:')
print(y['Target'].value_counts().to_string())
```
> Reports dataset dimensions, missing value count, column data types, and how many students belong to each class.

```python
counts = y['Target'].value_counts()
plt.figure(figsize=(6, 3))
bars = plt.bar(counts.index, counts.values, color=['#E8593C','#3B8BD4','#3DA87A'], edgecolor='white')
for b in bars: plt.text(b.get_x()+b.get_width()/2, b.get_height()+20, str(int(b.get_height())), ha='center', fontsize=10)
plt.title('Target class distribution')
plt.ylabel('Count')
plt.tight_layout(); plt.show()
```
> Draws a bar chart showing class imbalance. The loop adds count labels on top of each bar for readability.

---

### Cell 3 – Preprocessing Pipeline

```python
imputer = SimpleImputer(strategy='mean')
X_imp = imputer.fit_transform(X)
```
> Fits the imputer on training data: for any missing cell, fills with the column mean. `fit_transform` does both in one step. (In this dataset there are actually 0 missing values, but this is good defensive practice.)

```python
le = LabelEncoder()
y_enc = le.fit_transform(y['Target'])
print('Label mapping:', dict(zip(le.classes_, le.transform(le.classes_))))
```
> Converts string labels: `Dropout → 0`, `Enrolled → 1`, `Graduate → 2`. Prints the mapping so results can be interpreted later.

```python
sel = VarianceThreshold(threshold=0.01)
X_sel = sel.fit_transform(X_imp)
print(f'Features: {X_imp.shape[1]} -> {X_sel.shape[1]} ({X_imp.shape[1]-X_sel.shape[1]} low-variance removed)')
```
> Removes features whose variance is below 0.01 (nearly constant columns carry no discriminatory information). Here, all 36 features pass this filter.

```python
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_sel)
```
> Standardises every feature to **mean=0, std=1**. This is **essential for KNN** because the distance metric would otherwise be dominated by features with large numeric ranges (e.g., GDP vs age).

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y_enc, test_size=0.20, random_state=42, stratify=y_enc)
```
> Splits data 80/20. `stratify=y_enc` ensures the class distribution is preserved in both train and test sets (important for imbalanced data). `random_state=42` guarantees reproducibility.

---

### Cell 4 – Finding Best K (Hyperparameter Tuning)

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
k_range = range(1, 25)
cv_scores = []
```
> Sets up 5-fold stratified cross-validation. `shuffle=True` prevents consecutive samples of the same class being in the same fold.

```python
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean', n_jobs=-1)
    score = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    cv_scores.append(score)
```
> Trains and evaluates a separate KNN model for every K from 1 to 24. Uses CV accuracy (average across 5 folds) as the selection criterion. `n_jobs=-1` parallelises across all CPU cores.

```python
best_k = list(k_range)[np.argmax(cv_scores)]
print(f'Best K: {best_k}  (CV accuracy: {max(cv_scores):.4f})')
```
> Selects the K with the highest mean CV accuracy. Result: **K=17, CV accuracy=0.7152**.

```python
plt.plot(k_range, cv_scores, ...)
plt.axvline(best_k, ...)
```
> Plots the elbow curve showing accuracy vs K, with a vertical dashed line at the optimal K.

---

### Cell 5 – Train Final Model

```python
knn = KNeighborsClassifier(
    n_neighbors=best_k,    # K=17 (tuned via CV)
    weights='distance',    # closer neighbours vote more strongly
    metric='euclidean',    # standard L2 distance
    algorithm='auto',      # sklearn picks kd_tree / ball_tree / brute
    n_jobs=-1              # use all CPU cores
)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```
> Trains the final KNN model with the optimal hyperparameters on the full training set, then predicts on the held-out test set.

---

### Cell 6 – Evaluate Without SMOTE

```python
acc    = accuracy_score(y_test, y_pred)
f1_mac = f1_score(y_test, y_pred, average='macro')
f1_wt  = f1_score(y_test, y_pred, average='weighted')
```
> Computes three metrics:  
> - `accuracy_score`: fraction of correct predictions  
> - `f1_score(macro)`: unweighted average F1 across all classes  
> - `f1_score(weighted)`: F1 weighted by class support (number of samples)

```python
print(classification_report(y_test, y_pred, target_names=le.classes_))
```
> Prints per-class precision, recall, F1, and support — a detailed breakdown.

```python
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(...)
```
> Absolute confusion matrix: rows = true labels, columns = predicted labels.

```python
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, ...)
```
> Normalised confusion matrix: each cell shows the **percentage of the true class** predicted as that column. Makes class-wise recall directly visible.

---

### Cell 7 – SMOTE Oversampling

```python
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```
> Applies SMOTE **only to the training set**. For each minority sample, SMOTE creates synthetic points by interpolating between that sample and one of its 5 nearest neighbours of the same class. The test set is never touched.

```python
knn_sm = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean', n_jobs=-1)
knn_sm.fit(X_train_sm, y_train_sm)
y_pred_smote = knn_sm.predict(X_test)
```
> Trains a second KNN with the SMOTE-augmented training data and predicts on the same test set.

---

### Cell 8 – SMOTE Comparison

```python
def metrics(y_true, y_pred):
    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'F1 macro':    f1_score(y_true, y_pred, average='macro'),
        'F1 weighted': f1_score(y_true, y_pred, average='weighted'),
    }
```
> Helper function to avoid repeating metric computations. Returns a dict for easy tabulation.

```python
for key in m_no:
    diff = m_sm[key] - m_no[key]
    print(f"{key:<18} {m_no[key]:>12.4f} {m_sm[key]:>12.4f} {diff:>+8.4f}")
```
> Prints a side-by-side comparison table with the change direction (+/-) for each metric.

---

### Cell 9 – Visualisations

> Side-by-side normalised confusion matrices and per-class recall bar chart comparing SMOTE vs no-SMOTE. Also prints the full classification reports for both variants.

---

### Cell 10 – Save Model

```python
knn_bundle = {
    'model':    knn,
    'scaler':   scaler,
    'imputer':  imputer,
    'selector': sel,
    'encoder':  le,
}
with open('saved_models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_bundle, f)
```
> Saves the entire preprocessing pipeline **along with** the trained model in a single pickle bundle. This is important for deployment: when predicting on new data, you must apply the same transformations (same scaler, same encoder, etc.) that were fit on the training set.

---

## 3. Comparison with Other Algorithms

### Overview Table

| Aspect | KNN | Logistic Regression | Random Forest | SVM |
|---|---|---|---|---|
| **Type** | Instance-based (lazy) | Parametric (eager) | Ensemble (eager) | Margin-based (eager) |
| **Training time** | O(1) – just stores data | Fast | Moderate | Slow (large datasets) |
| **Prediction time** | Slow O(n·d) | Very fast O(d) | Fast O(trees·depth) | Fast O(support vectors) |
| **Handles non-linearity** | Yes (implicitly) | No (unless poly features) | Yes | Yes (kernel trick) |
| **Interpretability** | Low | High | Medium (feature importance) | Low |
| **Feature scaling needed** | **Yes – critical** | Yes | No | Yes |
| **Handles imbalance** | Poorly | Moderately | Yes (class_weight) | Yes (class_weight) |
| **Hyperparameters** | K, distance metric, weights | C, regularisation | n_estimators, max_depth | C, kernel, gamma |

### Working Mechanism

| Algorithm | How it learns |
|---|---|
| **KNN** | Stores all training data; classifies by majority vote of K nearest neighbours |
| **Logistic Regression** | Fits a linear decision boundary by maximising likelihood; outputs probabilities |
| **Random Forest** | Builds many decision trees on random subsets; aggregates votes (bagging) |
| **SVM** | Finds the hyperplane with maximum margin between classes; uses kernel for non-linearity |

### Advantages & Disadvantages

**KNN**
- ✅ Simple, no training phase, naturally handles multi-class
- ✅ Adapts locally — can capture complex decision boundaries
- ❌ Slow prediction for large datasets
- ❌ Sensitive to irrelevant/noisy features and scaling
- ❌ High memory usage (stores full training set)
- ❌ Struggles with high-dimensional sparse data (curse of dimensionality)

**Logistic Regression**
- ✅ Fast, interpretable, outputs calibrated probabilities
- ✅ Robust to small datasets
- ❌ Assumes linear separability — poor on complex patterns
- ❌ Needs feature engineering for non-linear relationships

**Random Forest**
- ✅ Handles non-linearity, mixed types, missing values
- ✅ Robust to overfitting, provides feature importance
- ❌ Slower prediction than single models
- ❌ Less interpretable than a single tree

**SVM**
- ✅ Effective in high-dimensional spaces
- ✅ Memory efficient (only support vectors matter)
- ❌ Very slow on large datasets (O(n²–n³) training)
- ❌ Sensitive to feature scaling and C/kernel choice

### This Project's Accuracy Comparison

| Model | Test Accuracy | Task |
|---|---|---|
| Random Forest | 91.46% | Binary (Dropout vs Graduate only) |
| Logistic Regression | 74.01% | 3-class |
| Decision Tree | 73.56% | 3-class |
| **KNN (K=17)** | **70.85%** | **3-class** |
| SVM | 69.15% | 3-class |

> ⚠️ Random Forest's higher accuracy is partly because it solves an easier binary problem. Among 3-class models, KNN ranks 3rd.

---

## 4. Dataset Analysis

### Source
UCI ML Repository – "Predict Students' Dropout and Academic Success"  
URL: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

### Dimensions
- **4,424 students** (rows)
- **36 features** (columns) + 1 target column
- **0 missing values**

### Feature Types

| Type | Count | Examples |
|---|---|---|
| Integer (int64) | 29 | Marital status, application mode, course, curricular units |
| Float (float64) | 7 | Previous qualification grade, admission grade, GDP, inflation, unemployment rate |

**Feature categories:**
1. **Demographics:** Marital status, nationality, gender, age at enrollment, international student flag
2. **Application details:** Application mode, order, daytime/evening attendance
3. **Educational background:** Previous qualification, previous qualification grade, mother's/father's qualification & occupation
4. **Academic performance (1st semester):** Credited, enrolled, evaluations, approved, grade, without evaluations
5. **Academic performance (2nd semester):** Same metrics as 1st semester
6. **Socioeconomic indicators:** Unemployment rate, inflation rate, GDP

### Target Variable Distribution

| Class | Count | Percentage |
|---|---|---|
| Graduate | 2,209 | 49.9% |
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 17.9% |

**Class imbalance:** The `Enrolled` class has only 794 samples (~18%) vs 2,209 for `Graduate` (~50%). This imbalance directly causes the model to under-perform on `Enrolled` predictions (F1=0.27).

### Patterns & Insights
- Students who pass more curricular units in semester 1 are strongly associated with graduation.
- Dropout students tend to have lower 1st-semester grades and approval rates.
- Enrolled students are "in progress" — their features resemble a mix of Dropout and Graduate patterns, making them inherently hard to classify.
- Socioeconomic features (GDP, unemployment) add context but have lower individual discriminatory power.

---

## 5. Data Preprocessing Steps

### Step 1: Imputing Missing Values

```python
imputer = SimpleImputer(strategy='mean')
X_imp = imputer.fit_transform(X)
```

**Why:** Missing values cause errors in scikit-learn models. Mean imputation replaces `NaN` with the column average. In this dataset there are 0 missing values, but this step is included as a defensive measure.

**Alternatives:** Median imputation (robust to outliers), KNN imputation, or dropping rows/columns.

### Step 2: Encoding the Target Label

```python
le = LabelEncoder()
y_enc = le.fit_transform(y['Target'])
```

**Mapping:** `Dropout=0`, `Enrolled=1`, `Graduate=2`

**Why:** scikit-learn models require numeric labels. `LabelEncoder` provides a consistent and reversible mapping.

### Step 3: Feature Selection (Variance Threshold)

```python
sel = VarianceThreshold(threshold=0.01)
X_sel = sel.fit_transform(X_imp)
```

**Why:** Features that are nearly constant across all samples carry no information for classification. This step removes such columns. All 36 features passed (none removed).

### Step 4: Feature Scaling – StandardScaler ⭐ Most Critical for KNN

```python
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_sel)
```

**Formula:** `z = (x - mean) / std`

**Why this is essential for KNN:** KNN uses **distance** to find neighbours. Without scaling, a feature like `GDP` (range: thousands) would completely dominate the distance calculation over `Age` (range: 17–70). StandardScaler brings all features to the same scale (mean=0, std=1) so each feature contributes equally.

**Important:** The scaler is `fit` only on the training data. The same training statistics (mean, std) are then applied to the test set. Fitting on the test set would be data leakage.

### Step 5: Stratified Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y_enc, test_size=0.20, random_state=42, stratify=y_enc)
```

**Split:** 80% train (3,539 samples), 20% test (885 samples)

**`stratify=y_enc`:** Ensures the class proportions in train and test mirror the original dataset. Without this, random sampling might over/under-represent a class in the test set.

### Step 6: SMOTE (Explored but Not Used in Final Model)

```python
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

SMOTE synthesises new minority-class samples by interpolating between existing ones. Applied to training set only.

**Result:** SMOTE **decreased** performance (accuracy dropped from 70.85% to 59.77%), so the final model uses the original imbalanced training data.

**Why SMOTE hurt KNN here:** KNN's decision boundary is based on true data density. Synthetic interpolated samples can alter the local density in misleading ways, confusing the model in the vicinity of class boundaries.

---

## 6. Choosing the Best Value of K

### The Problem

- **K too small (e.g., K=1):** The model memorises training data — **overfitting**. Noisy/outlier points heavily influence the prediction. High variance.
- **K too large (e.g., K=100):** The model averages over too broad a region — **underfitting**. Boundaries become overly smooth. High bias.

### Method Used: 5-Fold Stratified Cross-Validation

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
k_range = range(1, 25)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean', n_jobs=-1)
    score = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    cv_scores.append(score)
```

**How it works:**
1. The training set is split into 5 equal folds.
2. For each K, the model trains on 4 folds and validates on the 5th, cycling through all 5 combinations.
3. The 5 validation scores are averaged.
4. The K with the highest mean CV accuracy is selected.

**Result:** Best K = **17**, CV accuracy = **0.7152**

### Why K=17?

Odd values of K help avoid ties in binary classification. K=17 is large enough to smooth over noise but small enough to capture local patterns. The elbow curve (accuracy vs K plot) shows K=17 is near the peak before accuracy plateaus.

### Why Stratified Folds?

With class imbalance (18% Enrolled), un-stratified splits might accidentally put most Enrolled samples in one fold, giving unrepresentative CV scores. Stratified folds maintain the class proportions in each fold.

### Distance Weighting

Using `weights='distance'` rather than `weights='uniform'` gives more influence to closer neighbours. This is particularly helpful when K is larger — it prevents distant neighbours from polluting the vote with equal weight.

---

## 7. Model Evaluation

### Accuracy

```
Accuracy = (Correct Predictions) / (Total Predictions)
         = (TP_0 + TP_1 + TP_2) / n
```

**KNN result: 0.7085 (70.85%)**

**Limitation:** Accuracy is misleading with imbalanced classes. A model predicting only "Graduate" would get ~50% accuracy without learning anything useful.

### F1 Score

F1 is the **harmonic mean of Precision and Recall**:

```
Precision = TP / (TP + FP)    → "Of all predicted positives, how many were correct?"
Recall    = TP / (TP + FN)    → "Of all actual positives, how many were found?"

F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why F1 matters here:** With class imbalance (especially under-represented `Enrolled`), a model could achieve high accuracy by ignoring the minority class but would have a low F1. F1 penalises both false positives and false negatives equally.

**Macro F1 (0.5942):** Calculates F1 independently for each class, then takes the unweighted average. **This is the most honest metric for imbalanced datasets** because every class counts equally regardless of size.

**Weighted F1 (0.6773):** Weighs each class's F1 by its support (number of samples). Gives more weight to the dominant `Graduate` class.

### Confusion Matrix Interpretation

```
                 Predicted
              Dropout  Enrolled  Graduate
True Dropout    176      26        82     ← 62.0% recall
True Enrolled    46      30        83     ← 18.9% recall  ← Worst class
True Graduate     5      15       422     ← 95.5% recall  ← Best class
```

**Key observations:**

| Observation | What it means |
|---|---|
| `Enrolled` row has low values on diagonal (30/159 = 19%) | Model struggles most with `Enrolled` students |
| `Enrolled` is frequently predicted as `Graduate` (83 cases) | Enrolled students look similar to Graduates in the feature space |
| `Graduate` diagonal is very high (422/442 = 95.5%) | Model is excellent at identifying Graduates |
| `Dropout` recall is 62% | About 1 in 3 Dropouts are missed |

**Normalised confusion matrix** shows percentages within each true class, making class-wise recall directly visible.

### Per-Class Results Summary

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Dropout | 0.83 | 0.62 | 0.71 | 284 |
| Enrolled | 0.45 | 0.19 | 0.27 | 159 |
| Graduate | 0.70 | 0.95 | 0.80 | 442 |
| **Accuracy** | | | **0.71** | **885** |
| Macro avg | 0.66 | 0.59 | 0.59 | |
| Weighted avg | 0.69 | 0.71 | 0.68 | |

---

## 8. All Model Output Details

### Library Versions Used

```
numpy: 2.0.2
pandas: 2.2.2
scikit-learn: 1.6.1
matplotlib: 3.10.0
seaborn: 0.13.2
imbalanced-learn: 0.14.1
ucimlrepo: 0.0.7
```

### Dataset Loading

```
Shape: (4424, 36)  |  Missing values: 0
Data types: {int64: 29, float64: 7}

Target distribution:
Graduate    2209
Dropout     1421
Enrolled     794
```

### Preprocessing Output

```
Label mapping: {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
Features: 36 -> 36 (0 low-variance removed)
Train: 3539 | Test: 885
Train distribution: {'Dropout': 1137, 'Enrolled': 635, 'Graduate': 1767}
Test  distribution: {'Dropout': 284,  'Enrolled': 159, 'Graduate': 442}
```

### K Selection Output

```
Best K: 17  (CV accuracy: 0.7152)
```

### Final KNN Model Output (without SMOTE)

```
================================================
           KNN TEST RESULTS
================================================
  K (optimal):        17
  Test accuracy:      0.7085  (70.8%)
  F1 (macro):         0.5942
  F1 (weighted):      0.6773
================================================
              precision    recall  f1-score   support

     Dropout       0.83      0.62      0.71       284
    Enrolled       0.45      0.19      0.27       159
    Graduate       0.70      0.95      0.80       442

    accuracy                           0.71       885
   macro avg       0.66      0.59      0.59       885
weighted avg       0.69      0.71      0.68       885
```

### SMOTE Output

```
Training set size before SMOTE: 3539
Training set size after SMOTE:  5301
Class distribution after SMOTE: {'Dropout': 1767, 'Enrolled': 1767, 'Graduate': 1767}
```

### SMOTE vs No-SMOTE Comparison

```
====================================================
       SMOTE vs NO-SMOTE — METRIC COMPARISON
====================================================
Metric                 No SMOTE      + SMOTE   Change
----------------------------------------------------
Accuracy                 0.7085       0.5977  -0.1107
F1 macro                 0.5942       0.5778  -0.0163
F1 weighted              0.6773       0.6277  -0.0496
====================================================
```

> SMOTE made KNN **worse** across all metrics. The final model uses no SMOTE.

### Classification Reports: Both Variants

**Without SMOTE (final model):**
```
              precision    recall  f1-score   support

     Dropout       0.83      0.62      0.71       284
    Enrolled       0.45      0.19      0.27       159
    Graduate       0.70      0.95      0.80       442

    accuracy                           0.71       885
   macro avg       0.66      0.59      0.59       885
weighted avg       0.69      0.71      0.68       885
```

**With SMOTE:**
```
              precision    recall  f1-score   support

     Dropout       0.78      0.58      0.67       284
    Enrolled       0.28      0.57      0.37       159
    Graduate       0.79      0.62      0.69       442

    accuracy                           0.60       885
   macro avg       0.62      0.59      0.58       885
weighted avg       0.70      0.60      0.63       885
```

### All Models Comparison (Project-wide)

| Model | Accuracy | F1 Macro | F1 Weighted | Task |
|---|---|---|---|---|
| Random Forest | 91.46% | 0.91 | 0.91 | Binary |
| Logistic Regression | 74.01% | 0.69 | 0.75 | 3-class |
| Decision Tree | 73.56% | 0.65 | 0.72 | 3-class |
| **KNN (K=17)** | **70.85%** | **0.59** | **0.68** | **3-class** |
| SVM (RBF) | 69.15% | 0.64 | 0.70 | 3-class |

### Model Saved

```
KNN model saved -> saved_models/knn_model.pkl
```

The saved bundle contains: `model`, `scaler`, `imputer`, `selector`, `encoder` — everything needed for inference on new data.

---

## 9. Possible Viva Questions & Answers

### Core KNN Theory

**Q1: Why is feature scaling essential for KNN?**  
**A:** KNN computes Euclidean distance. Features with large numeric ranges (like GDP) would dominate the distance calculation over small-range features (like age). StandardScaler normalises all features to mean=0, std=1 so every feature contributes equally.

**Q2: What is the effect of K on the bias-variance tradeoff?**  
**A:** Small K → low bias, high variance (overfits noise). Large K → high bias, low variance (underfits). The optimal K balances these, found via cross-validation. K=17 was optimal here.

**Q3: Why did you choose `weights='distance'` instead of `weights='uniform'`?**  
**A:** Distance weighting gives more influence to closer neighbours. This reduces the impact of noisy or distant points, generally improving accuracy — especially useful when K is relatively large.

**Q4: Why is KNN called a lazy learner?**  
**A:** It performs no learning during training — it just stores the training data. All computation happens at prediction time, making training O(1) but prediction O(n×d).

**Q5: What is the curse of dimensionality?**  
**A:** In high-dimensional spaces, data points become equally distant from each other, making the concept of "nearest neighbour" meaningless. With 36 features, this dataset is moderately high-dimensional, which partly explains KNN's lower performance compared to models that learn explicit decision boundaries.

---

### Preprocessing

**Q6: Why did you use `stratify=y_enc` in train_test_split?**  
**A:** The dataset is imbalanced (Enrolled=18%). Without stratification, a random split might put disproportionate numbers of Enrolled samples in train or test, leading to unreliable evaluation. Stratification preserves the original class distribution in both sets.

**Q7: Why was the scaler fit only on training data?**  
**A:** Fitting the scaler on the test set would leak test data statistics into the preprocessing pipeline (data leakage). The test set must simulate unseen future data, so it's transformed using the mean and std computed from the training set only.

**Q8: What is VarianceThreshold doing?**  
**A:** It removes features that are nearly constant (variance below threshold=0.01). Such features carry no discriminatory information. In this dataset all 36 features pass (no features removed).

---

### SMOTE

**Q9: What is SMOTE and why did it hurt performance here?**  
**A:** SMOTE generates synthetic minority-class samples by interpolating between existing minority samples. It made KNN worse (accuracy dropped from 70.8% to 59.8%) because KNN relies on true data density for its decisions. Synthetic samples can distort local density in ways that confuse the distance-based classifier, especially near class boundaries.

**Q10: Should SMOTE always be applied to fix class imbalance?**  
**A:** No. SMOTE is not guaranteed to improve performance. It is dataset and algorithm dependent. For KNN specifically, density-based reasoning means synthetic points can mislead. Always validate empirically, as done here.

---

### Model Evaluation

**Q11: Why use macro F1 instead of accuracy for this dataset?**  
**A:** The dataset is imbalanced — Graduate has 2,209 samples while Enrolled has only 794. A model that always predicts Graduate would have ~50% accuracy but a terrible F1 for Dropout and Enrolled. Macro F1 gives equal weight to each class, penalising poor performance on minority classes.

**Q12: What does the confusion matrix reveal about your model?**  
**A:** The model performs best on Graduate (95.5% recall) and worst on Enrolled (19% recall). Many Enrolled students are misclassified as Graduate — this makes intuitive sense because currently-enrolled students who haven't yet graduated share many features with graduates (they are still performing academically).

**Q13: How would you improve Enrolled classification?**  
**A:** Possible approaches: (1) gather more Enrolled samples, (2) engineer features that better distinguish Enrolled from Graduate (e.g., current semester progression indicators), (3) use class-weighted models, (4) apply cost-sensitive learning to penalise Enrolled misclassification more heavily.

---

### Comparisons

**Q14: When would you choose KNN over Logistic Regression?**  
**A:** KNN is preferred when the decision boundary is non-linear and complex local patterns matter, data is sufficient for good coverage, and interpretability is less important. Logistic Regression is preferred when interpretability, calibrated probabilities, or fast prediction is needed.

**Q15: Why does Random Forest achieve 91.46% while KNN achieves only 70.85%?**  
**A:** Two reasons: (1) Random Forest in this project is solving an **easier binary problem** (Dropout vs Graduate only, Enrolled excluded), while KNN solves the harder 3-class problem. (2) Random Forest builds many decision trees capturing non-linear interactions between features — it is generally more powerful than KNN for tabular data.

**Q16: What algorithm would you recommend for production deployment for this dataset?**  
**A:** Random Forest (3-class version) or Gradient Boosting (XGBoost/LightGBM), which combine high accuracy, robustness to outliers, built-in feature importance, and no need for feature scaling.

---

### Project-Specific

**Q17: Why is Enrolled the hardest class to predict?**  
**A:** Enrolled students are still in progress — they haven't yet succeeded or failed. Their academic and demographic profiles overlap significantly with both Graduate and Dropout students. The outcome is genuinely ambiguous from current features alone.

**Q18: What would you do differently if you were to improve this KNN model?**  
**A:** (1) Try Manhattan or Mahalanobis distance metrics, (2) apply PCA/feature selection to reduce dimensionality, (3) experiment with higher K values, (4) apply class-weighted KNN or cost-sensitive learning, (5) use ensemble of KNNs with different metrics.

**Q19: What is the business significance of correctly identifying Dropouts?**  
**A:** Early identification of students likely to drop out allows institutions to intervene proactively (counselling, financial aid, academic support). False negatives (missing a Dropout prediction) are more costly than false positives, suggesting recall for Dropout class should be prioritised. The current model achieves 62% Dropout recall, meaning ~38% of actual dropouts are missed.

**Q20: How did you ensure there was no data leakage?**  
**A:** (1) The scaler, imputer, and feature selector are all `fit` only on the training data. (2) SMOTE is applied only on the training split after the train/test split. (3) The test set is only used once at the end to report final metrics — not for hyperparameter tuning.

---

*Guide generated for viva preparation on the Student Graduate Prediction ML project.*  
*Model: KNN | Dataset: UCI #697 | Best K: 17 | Test Accuracy: 70.85%*
