# Heart_Disease_Prediction

A comprehensive machine learning project to predict the presence of heart disease in patients using the UCI Heart Disease dataset. This project leverages the power of **Apache Spark (PySpark)** for data processing and model training, demonstrating an end-to-end workflow from data cleaning to model evaluation and analysis.

The final `RandomForest` model achieves an accuracy of approximately **86%** on the test set.

---

###  Key Features

- **Extensive Data Pre-processing:** Includes handling missing values, encoding categorical features, and removing outliers to create a clean, high-quality dataset.
- **PySpark ML Pipeline:** Utilizes a robust `pyspark.ml.Pipeline` to streamline feature engineering and model training.
- **Advanced Model Training:** Implements and compares two powerful classification models:
    - **RandomForest Classifier** (with hyperparameter tuning using `CrossValidator`).
    - **Gradient-Boosted Trees (GBT) Classifier**.
- **In-depth Model Evaluation:** Goes beyond accuracy to analyze model performance using:
    - **Confusion Matrices** to visualize true/false positives and negatives.
    - **Feature Importance** analysis to understand the key predictors of heart disease.
- **Scalable Solution:** Built on PySpark, making the solution scalable to much larger datasets.

---

###  Dataset

This project uses the **Heart Disease UCI** dataset, a well-known benchmark dataset for classification tasks in healthcare. It contains 14 attributes collected from four different hospitals. The goal is to predict the presence of heart disease (the `num` attribute) based on the other features.

---

###  Technology Stack

- **Language:** Python
- **Core Library:** PySpark
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab

---

###  How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mohammadkahr/Heart_Disease_Prediction.git
    cd Heart_Disease_Prediction
    ```
2.  **Upload the Dataset:**
    Ensure the `heart_disease_uci.csv` file is in the same directory as the notebook.
3.  **Run the Notebook:**
    Open the `.ipynb` file in Jupyter Notebook or Google Colab and run the cells in order. The notebook is self-contained and handles all environment setup required for Colab.

---

###  Results

- The optimized **RandomForest model** emerged as the top performer with an **accuracy of 85.6%**.
- The **GBT model** also showed strong performance with an accuracy of **81.6%**.
- **Feature importance analysis** revealed that the most significant predictors for heart disease in the model were:
    1.  `cp` (Chest Pain Type)
    2.  `oldpeak` (ST depression induced by exercise)
    3.  `exang` (Exercise Induced Angina)
    4.  `thalch` (Maximum Heart Rate Achieved)

This indicates that the model successfully learned medically relevant patterns from the data.