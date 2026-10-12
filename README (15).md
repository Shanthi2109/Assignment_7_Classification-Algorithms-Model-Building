# Crowdfunding Campaign Success Prediction

## Project Overview

As an aspiring Data Scientist, this project explores how machine learning classification algorithms can be used to predict whether a crowdfunding campaign will be **successful or unsuccessful**. The project compares multiple models, evaluates their performance, and applies cross-validation and hyperparameter tuning to identify a suitable model.

## Problem Statement

Crowdfunding platforms host campaigns across categories such as technology, education, healthcare, and creative arts. Campaign outcomes may depend on factors such as funding goals, campaign duration, backer engagement, and pledged amounts. Predicting campaign success can help stakeholders better understand campaign outcomes and make more informed decisions.

## Project Objectives

- Explore and understand the dataset.
- Clean the data by handling missing values and duplicate records.
- Encode categorical variables and prepare features for machine learning.
- Split the data into training and testing sets and scale numerical features where required.
- Build and compare classification algorithms.
- Evaluate model performance using standard classification metrics.
- Use cross-validation and hyperparameter tuning to assess and improve model performance.
- Select a model based on measured results and explain the comparison.

## Dataset

- **Dataset:** Crowdfunding Campaign Dataset
- **Target column used in the notebook:** `IsSuccessful`
- **Prediction task:** Binary classification (successful vs. unsuccessful campaign)

The notebook expects a CSV file to be uploaded when prompted in Google Colab. Make sure the dataset contains the `IsSuccessful` target column. The campaign features available for training depend on the CSV file used.

## Tools and Technologies

- **Python**
- **Pandas** — data loading, exploration, and cleaning
- **NumPy** — numerical operations
- **Matplotlib** — visualizing model performance and confusion matrices
- **Scikit-learn** — preprocessing, model building, evaluation, cross-validation, and hyperparameter tuning
- **Google Colab / Jupyter Notebook** — development environment

## Workflow

### 1. Data Exploration and Preprocessing
- Load the CSV dataset and inspect its shape, columns, data types, and sample records.
- Check missing values and duplicate records.
- Remove duplicate records and fill missing numerical values with the median and categorical values with the mode where possible.
- Separate the target (`IsSuccessful`) from the input features.
- Remove identifier columns such as `id`, `project_id`, or `campaign_id` when present.
- Encode categorical features, split the dataset into training and testing sets (80% training and 20% testing), and scale numerical features where required.

### 2. Classification Model Building
The notebook implements these five classifiers:
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Gaussian Naive Bayes
- Linear Support Vector Machine (SVM)

The notebook also experiments with selected model parameters.

### 3. Model Evaluation
Models are compared using:
- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Confusion matrices
- Training versus testing accuracy

The F1-score is used as a key comparison metric in the notebook.

### 4. Cross-Validation and Hyperparameter Tuning
`RandomizedSearchCV` is used with three-fold cross-validation to explore parameter combinations for each classifier. The notebook records the best parameters, cross-validation F1-score, and cross-validation score variation, then evaluates the tuned models on the test set.

### 5. Model Comparison and Selection
The notebook creates comparison tables and charts to compare classifier performance. The model with the highest F1-score in the final comparison is selected as the best-performing model for this experiment. The final choice should be based on the outputs generated when the notebook is run on the selected dataset.


## Trained Classification Models

The notebook trains and tunes the following five classifiers:

1. Random Forest Classifier
2. K-Nearest Neighbors (KNN)
3. Decision Tree Classifier
4. Gaussian Naive Bayes
5. Linear Support Vector Machine (SVM)

The tuned estimators are stored in the notebook's `best_models` dictionary and evaluated on the test set. The notebook does not currently export the trained models as separate `.pkl` or `.joblib` files.

## Evaluation Metrics and Model Comparison

The following results are from the notebook's **final comparison of tuned models** on the 20% test split:

| Model | Training Accuracy | Testing Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) |
|---|---:|---:|---:|---:|---:|
| Random Forest | 1.0000 | 0.9635 | 0.9645 | 0.9635 | 0.9634 |
| KNN | 1.0000 | 0.5114 | 0.5114 | 0.5114 | 0.5114 |
| Decision Tree | 0.9987 | 0.9925 | 0.9925 | 0.9925 | 0.9925 |
| Naive Bayes | 0.6672 | 0.6726 | 0.6972 | 0.6726 | 0.6623 |
| SVM | 0.9990 | 0.9989 | 0.9989 | 0.9989 | 0.9989 |

*Precision, recall, and F1-score are weighted averages. Values are copied from the notebook output and may vary if the dataset or training setup changes.*

### Cross-validation and hyperparameter tuning

The notebook uses `RandomizedSearchCV` with three-fold cross-validation. The best cross-validation F1-scores reported during tuning were:

| Model | Best cross-validation F1-score | Best parameters reported |
|---|---:|---|
| Random Forest | 0.9295 | `n_estimators=150`, `min_samples_split=2`, `max_depth=None` |
| KNN | 0.5074 | `weights='distance'`, `n_neighbors=7` |
| Decision Tree | 0.9802 | `min_samples_split=10`, `max_depth=None`, `criterion='gini'` |
| Naive Bayes | 0.6571 | `var_smoothing=1e-9` |
| SVM | 0.9942 | `C=1.0` |

## Final Conclusion

Based on the notebook's final test-set comparison, **Linear Support Vector Machine (SVM)** is the best-performing model in this experiment. It achieved **99.89% testing accuracy, 99.89% weighted precision, 99.89% weighted recall, and a 99.89% weighted F1-score**. Its training accuracy was 99.90%, giving a very small training–testing accuracy gap of 0.01 percentage points.

The Decision Tree was the next strongest model, with a 99.25% test F1-score. KNN and Naive Bayes performed substantially worse on this dataset.

**Important validation note:** The dataset includes `RaisedAmount` and `NumBackers`, which may contain information that is only known after a campaign has run. If the goal is to predict success before or at campaign launch, these features can cause target leakage—especially if `IsSuccessful` is defined using the amount raised relative to the funding goal. Before presenting this as a real-world predictive result, verify how the target was generated and rerun the experiment using only features available at the intended prediction time. The very high SVM score should therefore be interpreted cautiously.

## How to Run the Project

1. Open the notebook `Assinment_7_Classification_Algorithms_Model_Building.ipynb` in Google Colab or Jupyter Notebook.
2. Run the cells in order.
3. When prompted, upload the crowdfunding campaign CSV file.
4. Confirm that the CSV includes the `IsSuccessful` target column.
5. Review the data-cleaning, model evaluation, confusion matrix, cross-validation, and tuning outputs.
6. Use the final comparison table and chart to report which model performed best on the test data.

## Key Learning Outcomes

Through this project, I practised:
- Data inspection and preprocessing
- Handling missing values and duplicates
- Encoding categorical variables and scaling numerical features
- Training multiple classification algorithms
- Evaluating classifiers with accuracy, precision, recall, F1-score, and confusion matrices
- Applying cross-validation and randomized hyperparameter search
- Comparing model performance and selecting a model using evaluation evidence

## Conclusion

This project demonstrates an end-to-end classification workflow, from preparing campaign data to evaluating and tuning machine learning models. The final model selection should be supported by the notebook's actual test-set metrics and cross-validation results; no performance values are assumed in this README.


