# Heart Disease Prediction

[![Made with PySpark](https://img.shields.io/badge/Made%20with-PySpark-orange.svg)](https://spark.apache.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive machine learning project to predict the severity of heart disease in patients using the UCI Heart Disease dataset. This project leverages the power of **Apache Spark (PySpark)** to build a robust, end-to-end data processing and machine learning pipeline.

The project explores and compares three different classification models to tackle this **multi-class classification problem**, with the best model achieving an accuracy of approximately **63%** on the test set.

---

### Project Overview

This project's primary goal is to predict the presence and severity of heart disease, which is categorized into five levels (0: no disease, 1-4: increasing severity). It demonstrates a complete machine learning workflow, including:
1.  Setting up a Spark environment.
2.  Performing extensive data cleaning and pre-processing.
3.  Building a modular feature engineering pipeline using `pyspark.ml`.
4.  Training and tuning multiple classification models.
5.  Evaluating and interpreting model performance.

The entire solution is built on PySpark, making it scalable to handle datasets that are too large for single-machine tools like Pandas.

---

### Key Features

- **Extensive Data Pre-processing:** Includes handling missing values (imputation with median/mode), encoding categorical features, and removing statistical outliers to create a high-quality dataset.
- **PySpark ML Pipeline:** Utilizes a robust `pyspark.ml.Pipeline` to streamline feature engineering (`StringIndexer`, `VectorAssembler`) and ensure consistency between training and testing.
- **Multi-Model Comparison:** Implements, tunes, and compares three powerful classification models:
    - **Random Forest Classifier**
    - **Decision Tree Classifier**
    - **Logistic Regression** (for multi-class classification)
- **Hyperparameter Tuning:** Employs `CrossValidator` and `ParamGridBuilder` to find the optimal hyperparameters for each model, maximizing performance.
- **In-depth Model Evaluation:** Goes beyond simple accuracy to analyze model performance using:
    - **Confusion Matrices** to visualize per-class performance.
    - **Feature Importance** analysis to identify the key medical predictors of heart disease.

---

### Dataset

This project uses the **Heart Disease UCI** dataset, a well-known benchmark for classification tasks in healthcare. The dataset combines data from four different hospitals and contains 14 attributes per patient. The target variable, `num`, has been used as a multi-class label representing the severity of heart disease (0, 1, 2, 3, 4).

---

###  Technology Stack

- **Language:** Python
- **Core Framework:** Apache Spark (PySpark)
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab (with automated JDK installation)

---

### Workflow

The project follows a structured machine learning workflow:

`Data Loading & Cleaning` -> `Feature Engineering Pipeline (Indexing & Assembling)` -> `Train/Test Split` -> `Model Training & Cross-Validation (RF, DT, LR)` -> `Prediction on Test Data` -> `Performance Evaluation (Accuracy, Confusion Matrix, Feature Importance)`

---

### Results and Analysis

The models were trained on 80% of the data and evaluated on the remaining 20%.

#### Model Performance

The Random Forest model delivered the best performance on this challenging multi-class classification task.

| Model                 | Test Accuracy |
| --------------------- | :-----------: |
| **Random Forest**     |   **~62.9%**   |
| Logistic Regression   |    ~62.1%     |
| Decision Tree         |    ~61.4%     |

#### Analysis of Low Accuracy

An accuracy of ~63% may seem low, but it is explainable due to the following challenges:
1.  **Difficult Multi-Class Problem:** Distinguishing between 5 fine-grained levels of disease severity is inherently more complex than a simple binary (yes/no) prediction. The subtle differences between classes (e.g., severity 1 vs. 2) make the task difficult for any model.
2.  **Severe Class Imbalance:** The dataset is heavily skewed, with a large majority of patients belonging to Class 0 (no disease). This imbalance causes the model to become biased towards predicting the majority class, leading to poor performance on the minority classes (1-4), which are often of greater clinical interest.
3.  **Limited Data for Minority Classes:** After cleaning and outlier removal, the number of samples for the higher-severity classes becomes very small, making it difficult for the model to learn their underlying patterns effectively.

#### Feature Importance

Feature importance analysis from the best model (Random Forest) revealed the most significant predictors for heart disease severity. This provides valuable insight into which clinical factors are most influential.

| Rank | Feature         | Importance Score |
| :--: | --------------- | :--------------: |
| 1    | `cp` (Chest Pain Type) |      ~0.203      |
| 2    | `exang` (Exercise Induced Angina) |      ~0.144      |
| 3    | `oldpeak` (ST Depression) |      ~0.131      |
| 4    | `age`           |      ~0.130      |
| 5    | `thalch` (Max Heart Rate) |      ~0.088      |

---

### Conclusion

This project successfully demonstrates the use of PySpark to build an end-to-end multi-class classification system for predicting heart disease severity. The final Random Forest model achieved an accuracy of **~62.9%**. Despite the moderate accuracy, the project highlights key challenges like class imbalance and the difficulty of multi-class problems. The feature importance results align with clinical knowledge, confirming that factors like chest pain type and exercise-induced indicators are critical for diagnosis.

### Future Work

To improve model performance, the following strategies could be implemented:

- **Handle Class Imbalance:**
  - **Oversampling:** Use techniques like **SMOTE (Synthetic Minority Over-sampling TEchnique)** to generate synthetic data for the minority classes, creating a more balanced dataset.
  - **Class Weights:** Assign higher weights to minority classes during model training using the `weightCol` parameter available in PySpark classifiers. This forces the model to pay more attention to misclassifying rare samples.

- **Simplify the Problem (Binary Classification):**
  - Convert the problem into a binary classification task (disease vs. no disease) by mapping `num > 0` to `1` and `num = 0` to `0`. This simpler, high-impact problem is likely to yield significantly higher accuracy and more actionable results.

- **Explore Advanced Models:**
  - Experiment with other powerful algorithms like **Gradient-Boosted Trees (`GBTClassifier`)** or third-party libraries like **XGBoost on Spark**, which often outperform Random Forests.

- **Advanced Feature Engineering:**
  - Create new features (e.g., interaction terms between age and cholesterol, or polynomial features) to help the model capture more complex, non-linear relationships in the data.