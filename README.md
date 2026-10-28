# SWaT Industrial Anomaly Detection with Databricks

## Project Overview

This project explores the **SWaT (Secure Water Treatment) dataset** using Databricks, PySpark, Delta Lake and machine learning. The goal is to build a structured data pipeline, investigate sensor behaviour, prepare data for modelling, and evaluate models that distinguish **normal operation** from **attack events**.

The notebook covers data exploration, data-quality checks, Medallion Architecture layers (Bronze, Silver and Gold), Delta tables, sensor analysis, temporal train/validation/test splits, and model evaluation.

> **Model scope:** The uploaded notebook implements and evaluates **Logistic Regression** and **Random Forest**. It does not contain an Isolation Forest implementation. If Isolation Forest is part of another notebook or project stage, add it to this README only after documenting that implementation and its results.

## Objectives

- Explore the SWaT dataset and inspect its schema and data quality.
- Build Bronze, Silver and Gold data layers.
- Store processed datasets as Delta tables in Databricks.
- Parse timestamps and convert sensor measurements to numeric values.
- Investigate missing sensor values, duplicated records and class distribution.
- Analyse the relationship between sensor readings and attack labels.
- Train and evaluate supervised classification models.
- Use time-based data splits to reduce temporal leakage and evaluate performance on later periods.

## Technology Stack

- **Databricks**
- **Apache Spark / PySpark**
- **Delta Lake**
- **Python**
- **Spark MLlib**
- **scikit-learn** for Average Precision calculation in the notebook

## Dataset

The notebook reads the source CSV from a Databricks Unity Catalog Volume:

`/Volumes/first_catalog/default/first_volume/swat_dataset_merged.csv`

The initial exploration reports:

- **1,441,719 records**
- **53 columns**
- A timestamp field
- The target field `Normal/Attack`
- Sensor measurements used to investigate normal and attack behaviour

The source data covers December 2015 and January 2016 in the time-based modelling workflow. The notebook examines the target classes `NORMAL` and `ATTACK`.

The dataset is expected to be made available separately in the Databricks environment. Do not commit the dataset to GitHub unless its licence and distribution terms explicitly permit it.

## Data Engineering: Medallion Architecture

The notebook organises data processing into Bronze, Silver and Gold layers. Each layer is stored as a Delta table.

### Bronze — Raw Data

**Table:** `first_catalog.default.bronze_swat`

The Bronze layer preserves the source records before the main data type conversions.

Activities include:
- Reading the CSV with PySpark.
- Inspecting the schema, columns and record count.
- Preserving the original data for subsequent processing.
- Investigating timestamp formats and duplicated records.

### Silver — Cleaned and Typed Data

**Table:** `first_catalog.default.silver_swat`

The Silver layer prepares the data for reliable analysis.

Activities include:
- Cleaning whitespace around column names and values.
- Parsing `Timestamp` into a timestamp data type.
- Renaming `Normal/Attack` to `normal_attack`.
- Converting sensor columns to numeric (`DOUBLE`) types.
- Normalising the target labels.
- Checking timestamp conversion and data quality.

The notebook reports **zero null timestamps** after the validated timestamp conversion.

### Gold — Analytical and Machine Learning Data

The notebook creates Gold tables for class summaries and machine learning.

| Delta table | Purpose |
|---|---|
| `first_catalog.default.gold_swat_class_summary` | Aggregated counts and percentages by target class |
| `first_catalog.default.gold_swat_ml` | Analytical dataset with selected sensor columns removed |
| `first_catalog.default.gold_swat_ml_dedup` | Machine-learning dataset after exact duplicate rows are removed |
| `first_catalog.default.gold_swat_sensor_auc` | Per-sensor ROC-AUC analysis results |

For the machine-learning dataset, the notebook removes seven sensor columns with missing-data concerns: `AIT201`, `MV101`, `MV201`, `MV303`, `P201`, `P202` and `P204`.

The resulting `gold_swat_ml` table contains **1,441,719 records and 46 columns**. The deduplicated table `gold_swat_ml_dedup` is then used for further analysis and modelling. The modelling workflow identifies **44 sensor features**, excluding the timestamp and target columns.

## Delta Lake and Data Versioning

The pipeline writes tables using the Delta format, for example with Spark's `.write.format("delta").saveAsTable(...)` API.

Delta Lake provides a transactional table format and supports features such as schema enforcement and table history. Delta time travel can make previous table versions queryable while the relevant history and data files remain available under the table's retention and maintenance settings.

Delta table versioning is different from machine-learning model versioning: this notebook's Delta tables track data-table changes; the notebook does not establish a separate model registry or model-versioning workflow.

## Exploratory Data Analysis and Data Quality

The notebook investigates:

- Dataset dimensions, columns and data types.
- Timestamp parsing and invalid timestamp values.
- Missing values in sensor columns.
- Exact duplicate records and repeated timestamps.
- The distribution of `NORMAL` and `ATTACK` observations.
- Sensor statistics and differences between the two classes.
- Per-sensor ROC-AUC as an exploratory measure of how well individual sensor values separate the classes.

The notebook also saves the per-sensor ROC-AUC analysis in the Gold table `gold_swat_sensor_auc`.

## Machine Learning

The target is converted into a binary label:

- `NORMAL` → `0`
- `ATTACK` → `1`

The notebook uses sensor readings as input features and excludes `Timestamp`, `normal_attack` and `label` from the feature vector.

### 1. Logistic Regression

The notebook trains a binary Logistic Regression model with Spark MLlib.

Pipeline steps:
1. Assemble the sensor columns into a feature vector using `VectorAssembler`.
2. Scale the features with `StandardScaler`.
3. Train `LogisticRegression` with `maxIter=50` and `regParam=0.01`.
4. Generate predictions and attack probabilities.
5. Evaluate predictions using precision, recall, F1-score and PR-AUC.
6. Explore different probability thresholds to investigate the trade-off between missed attacks and false alarms.

The notebook reports a **PR-AUC of 0.2527** for the initial Logistic Regression evaluation. Its default decision threshold has a low recall, which motivates the subsequent threshold analysis.

### 2. Random Forest

The notebook also trains Random Forest classifiers using Spark MLlib.

The initial configuration uses:
- `numTrees=50`
- `maxDepth=10`
- `seed=42`

The notebook then introduces a more careful temporal evaluation setup:
- **Training:** through 28 December 2015
- **Validation:** 29–30 December 2015
- **Final test:** 31 December 2015–2 January 2016

The validation set is used to compare probability thresholds. A threshold of **0.30** is selected based on the F1-score in validation, then fixed for the final test evaluation.

## Results

### Random Forest — final temporal test

With the threshold fixed at `0.30`, the notebook reports:

| Metric | Result |
|---|---:|
| Precision | 0.5763 |
| Recall | 0.7923 |
| F1-score | 0.6672 |
| False-positive rate | 0.1358 |
| True positives (TP) | 33,962 |
| False positives (FP) | 24,973 |
| False negatives (FN) | 8,902 |
| True negatives (TN) | 158,882 |

These results show that the selected threshold detects a substantial proportion of attack records, while still producing false alarms. The threshold is an experimental choice and should be selected according to the operational cost of missed attacks versus false positives.

The notebook also reports **Average Precision = 0.3431** on the validation period, where attack prevalence is 4.22%. An earlier Random Forest experiment reports Average Precision = 0.8697 on a different test split. These results are **not directly comparable**, because the temporal splits differ.

The attack prevalence also changes across the temporal splits: approximately 0.82% in training, 4.22% in validation and 18.91% in the final test. This distribution shift is an important limitation when interpreting model performance and transferring a threshold between time periods.

## Workflow Summary

1. Read the SWaT CSV from a Databricks Volume.
2. Inspect the source dataset and create the Bronze Delta table.
3. Clean and type the data in Silver.
4. Create Gold analytical and machine-learning tables.
5. Investigate sensor quality, duplicate records, class balance and sensor-level ROC-AUC.
6. Create time-based training, validation and test datasets.
7. Train Logistic Regression and Random Forest models.
8. Evaluate classification performance and compare probability thresholds.
9. Evaluate the final Random Forest model using the threshold selected on validation data.

## Repository Structure

A suggested GitHub structure is:

```text
Anomaly-Detection-and-Classification-with-PySpark-on-Databricks/
├── 00_SWAT_Data_Exploration.ipynb
├── README.md
└── .gitignore
```

Adapt the structure to match the files you publish. The source dataset and Databricks-specific credentials or configuration should not be committed to the repository.

## How to Run

1. Create or open a Databricks workspace.
2. Make the SWaT CSV available at the Volume path configured in the notebook, or update the path to your own approved location.
3. Ensure that the catalog `first_catalog`, schema `default` and required permissions are available, or update the table names.
4. Import `00_SWAT_Data_Exploration.ipynb` into Databricks.
5. Run the notebook cells in order to create the Delta tables, explore the data and execute the modelling workflow.

The notebook depends on Databricks/Spark features and a source dataset that are not included in this repository by default.

## Limitations and Future Improvements

- Add an Isolation Forest implementation and document its anomaly scores and evaluation results if it is part of the wider project. It is not implemented in this notebook.
- Compare models using the same temporal splits and evaluation protocol.
- Investigate class imbalance and the effect of changing attack prevalence over time.
- Tune model parameters using validation data only.
- Assess the operational trade-off between missed attacks and false alarms.
- Add model persistence, experiment tracking and model versioning.
- Document the exact source dataset version and data access requirements.

## Skills Demonstrated

`Databricks` · `PySpark` · `Delta Lake` · `Medallion Architecture` · `Data Quality` · `Feature Engineering` · `Time-Based Validation` · `Spark MLlib` · `Logistic Regression` · `Random Forest` · `Anomaly Detection`

---

This project demonstrates an end-to-end data exploration and machine-learning workflow on industrial sensor data, combining layered data engineering with temporal evaluation of attack classification models.
