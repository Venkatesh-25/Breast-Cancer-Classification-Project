
# Breast Cancer Diagnosis Using Supervised Machine Learning

A comparative study of supervised machine learning algorithms for classifying breast cancer tumors as **malignant or benign** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

This project was developed as part of **Continuous Assessment 4 (CA-4)** for the Supervised Machine Learning course at Symbiosis Institute of Technology, Hyderabad.

## 1. Project Overview

The objective of this project is to evaluate different supervised classification algorithms and identify a suitable model for distinguishing malignant from benign breast tumor samples.

The study implements five machine learning algorithms, compares their predictive performance, performs hyperparameter optimization, and analyzes the results using standard classification metrics.

**Key features:**
- Exploratory Data Analysis (EDA)
- Missing value and outlier detection
- Feature distribution and correlation analysis
- Data preprocessing and standardization
- Stratified 70:30 train-test split
- Five supervised classification algorithms
- Five-fold stratified cross-validation
- Hyperparameter optimization using GridSearchCV
- ROC curve and confusion matrix analysis
- Feature importance visualization
- Model comparison and final selection

## 2. Dataset

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset contains measurements computed from digitized images of fine-needle aspirates of breast masses.

| Attribute | Description |
|---|---|
| Total samples | 569 |
| Input features | 30 |
| Classification type | Binary |
| Benign samples | 357 |
| Malignant samples | 212 |
| Missing values | 0 |
| Training samples | 398 |
| Testing samples | 171 |

The target variable is encoded as:

- `0` — Benign
- `1` — Malignant

The dataset includes nuclear measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## 3. Methodology

The project follows a structured supervised learning workflow.

```text
Dataset Loading
       |
       v
Exploratory Data Analysis
       |
       v
Data Cleaning and Preprocessing
       |
       v
Stratified Train-Test Split (70:30)
       |
       v
Baseline Model Training
       |
       v
Advanced Ensemble Models
       |
       v
Cross-Validation and GridSearchCV
       |
       v
Model Evaluation and Comparison
       |
       v
Final Model Selection
```

### Preprocessing

- Missing values were assessed, with median imputation included in the modeling pipelines.
- Outliers were examined using the Interquartile Range (IQR) method.
- StandardScaler was applied to Logistic Regression and KNN.
- Stratified sampling preserved class proportions.
- Balanced class weights were used for Logistic Regression, Decision Tree, and Random Forest.
- Preprocessing was fitted using training data to prevent data leakage.

## 4. Machine Learning Algorithms

Five supervised learning algorithms were implemented.

### Baseline Models
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)

### Advanced Models
4. Random Forest Classifier
5. Gradient Boosting Classifier

Random Forest was further optimized using GridSearchCV with five-fold stratified cross-validation.

## 5. Experimental Results

The models were evaluated using accuracy, precision, recall, F1 score, ROC AUC, and mean cross-validation F1.

Malignant tumors were treated as the positive class.

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | CV F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.977** | 0.984 | **0.953** | **0.968** | 0.997 | **0.957** |
| Decision Tree | 0.912 | 0.902 | 0.859 | 0.880 | 0.898 | 0.900 |
| KNN | 0.965 | 1.000 | 0.906 | 0.951 | 0.988 | 0.943 |
| Random Forest | 0.971 | 1.000 | 0.922 | 0.959 | 0.997 | 0.922 |
| Gradient Boosting | 0.965 | 1.000 | 0.906 | 0.951 | 0.997 | 0.912 |

*Results are based on the 171-sample held-out test set. CV F1 represents five-fold cross-validation performance on the training set.*

### Hyperparameter Optimization

Random Forest was optimized using GridSearchCV.

The selected hyperparameters were:

```python
{
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 3,
    "class_weight": "balanced"
}
```

**Optimized Random Forest results:**

| Metric | Value |
|---|---:|
| Accuracy | 97.66% |
| Precision | 100.00% |
| Recall | 93.75% |
| F1 Score | 0.9677 |
| ROC AUC | 0.9980 |
| Cross-validation F1 | 0.9267 |

### Final Selected Model: Logistic Regression

Logistic Regression was selected as the preferred model based on its strong performance, higher malignant recall, better mean cross-validation F1, and relative interpretability.

**Final performance:**

| Metric | Value |
|---|---:|
| Accuracy | 97.66% |
| Precision | 98.39% |
| Malignant Recall | 95.31% |
| F1 Score | 0.9683 |
| ROC AUC | 0.9969 |
| Cross-validation F1 | 0.9572 |

**Confusion Matrix:**

| Actual / Predicted | Benign | Malignant |
|---|---:|---:|
| Benign | 106 | 1 |
| Malignant | 3 | 61 |

The model correctly classified 167 out of 171 test observations.

Although the optimized Random Forest achieved a slightly higher ROC AUC, Logistic Regression identified one additional malignant case at the default classification threshold.

These results apply to the specific dataset and experimental configuration.

## 6. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 7. Repository Structure

```text
breast-cancer-classification/
│
├── README.md
│
├── Breast_Cancer_CA4_Final_Split_Notebook.ipynb
│
└── 24070721014 ML REPORT.pdf
```

## 8. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-classification.git
cd breast-cancer-classification
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Install dependencies

Ensure Python 3.10 or later is installed.

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

### Step 3: Launch Jupyter Notebook

```bash
jupyter notebook
```

### Step 4: Execute the project

Open:

`Breast_Cancer_CA4_Final_Split_Notebook.ipynb`

Select **Kernel → Restart & Run All**.

The notebook loads the dataset directly through scikit-learn, so no separate dataset download is required.

## 9. Reproducibility

The experiment uses:

- Random state: `42`
- Train-test ratio: `70:30`
- Stratified sampling
- Five-fold cross-validation
- Training-only preprocessing pipelines
- A consistent evaluation procedure for all classifiers

The notebook contains the complete implementation, including data analysis, model training, hyperparameter tuning, evaluation metrics, and visualizations.

## 10. Limitations

Despite achieving high classification performance, the project has several limitations:

- The dataset contains only 569 observations.
- The experiment uses a single train-test partition.
- No independent external clinical dataset was evaluated.
- The results may vary across different data distributions.
- Model performance does not establish clinical safety or reliability.
- Further validation and threshold analysis would be necessary before real-world application.

**Disclaimer:** This project is intended for educational and research demonstration purposes only. It is not a medical diagnostic tool.

## 11. References

1. Wolberg, W. H., Street, W. N., and Mangasarian, O. L. *Breast Cancer Wisconsin (Diagnostic).* UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

2. Scikit-learn Developers. *Scikit-learn Documentation.* https://scikit-learn.org/stable/

3. Breiman, L. (2001). *Random Forests.* Machine Learning, 45, 5–32. https://doi.org/10.1023/A:1010933404324

4. Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* The Annals of Statistics, 29(5), 1189–1232.

---

## Author

**K Venkatesh**  
B.Tech — Computer Science and Engineering  
Symbiosis Institute of Technology, Hyderabad

**Course:** Supervised Machine Learning  
**Assignment:** Continuous Assessment 4 (CA-4)  
**Academic Year:** 2026–2027
