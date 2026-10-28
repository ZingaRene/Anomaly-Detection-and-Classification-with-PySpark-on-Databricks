# Anomaly Detection and Classification with PySpark on Databricks

## Project Overview

This project was developed using **Databricks and PySpark** to demonstrate a data processing and machine learning workflow, from data preparation and transformation to model training and evaluation.

The project explores two machine learning algorithms — **Isolation Forest** and **Logistic Regression** — to investigate anomaly detection and classification approaches in a data-driven environment.

The main objective is to prepare and analyse data, develop machine learning models, and evaluate their performance to identify the most suitable approach for the problem.

## Objectives

- Load and explore the dataset using PySpark.
- Clean and transform raw data into a suitable format for analysis.
- Handle data quality issues, including timestamp conversion and invalid values.
- Prepare the data for machine learning.
- Train and evaluate Isolation Forest and Logistic Regression models.
- Compare model performance and understand the strengths and limitations of each approach.

## Technologies and Tools

- **Databricks** — cloud-based data engineering and machine learning environment.
- **Apache Spark / PySpark** — distributed data processing and transformation.
- **Python** — data manipulation and machine learning workflows.
- **Machine Learning** — anomaly detection and classification.

## Machine Learning Models

### 1. Isolation Forest

Isolation Forest is an **unsupervised anomaly detection algorithm** designed to identify observations that differ significantly from the general data distribution.

In this project, it is used to explore the detection of potentially anomalous records without requiring labelled examples during training.

Key concepts:
- Anomaly detection without labelled training data.
- Isolation of unusual observations through random partitioning.
- Identification of records that may require further investigation.

### 2. Logistic Regression

Logistic Regression is a **supervised classification algorithm** used to estimate the probability that an observation belongs to a particular class.

In this project, it is used as a classification approach, provided that labelled data is available.

Key concepts:
- Training with labelled observations.
- Predicting class membership.
- Evaluating classification performance using appropriate metrics.

### Model Evaluation

Both approaches are considered within the project workflow, with evaluation methods depending on the model's purpose and the availability of ground-truth labels.

For Logistic Regression, relevant evaluation metrics may include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For Isolation Forest, relevant evaluation methods may include:
- Number and proportion of detected anomalies.
- Comparison between anomaly predictions and known labels, when available.
- Precision, recall and F1-score, when reliable ground-truth labels exist.

**Note:** The final model comparison should be based on the actual evaluation results obtained in the Databricks notebook.

## Project Workflow

1. **Data ingestion** — load the source dataset into Databricks.
2. **Data exploration** — inspect the schema, columns and data quality.
3. **Data cleaning** — handle missing values, invalid records and data type conversions.
4. **Feature preparation** — select and transform the variables required for modelling.
5. **Model training** — train the Isolation Forest and Logistic Regression models.
6. **Model evaluation** — assess model predictions using appropriate evaluation techniques.
7. **Results analysis** — interpret the results and consider the strengths and limitations of both approaches.

## Repository Structure

A possible repository structure is:

```text
databricks-anomaly-detection/
├── notebooks/
│   └── anomaly_detection_and_classification.py
├── data/
│   └── README.md
├── results/
│   └── README.md
└── README.md
```

*Adapt the structure to match the files actually included in the repository. Do not upload confidential or sensitive datasets.*

## Key Learning Outcomes

Through this project, I developed practical experience with:

- Distributed data processing using PySpark.
- Data cleaning and transformation in Databricks.
- Preparing datasets for machine learning.
- Training and evaluating unsupervised and supervised learning models.
- Understanding the differences between anomaly detection and binary or multiclass classification.
- Structuring a reproducible data analysis and machine learning workflow.

## Future Improvements

Potential improvements include:

- Comparing model performance using a consistent evaluation strategy.
- Optimising model parameters.
- Improving feature engineering and data quality checks.
- Visualising model predictions and evaluation metrics.
- Documenting experiment results and adding reproducibility instructions.

## Conclusion

This project demonstrates a practical application of data processing and machine learning in Databricks, combining PySpark transformations with two complementary algorithms: **Isolation Forest for anomaly detection** and **Logistic Regression for supervised classification**.

It reflects my ongoing development of practical skills in data engineering, data analysis and machine learning.

---

**Tools:** Python · PySpark · Apache Spark · Databricks · Machine Learning
