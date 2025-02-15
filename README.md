# Statistical Methods in Data Mining

## Final Project

---

## Contents

1. [Introduction](#introduction)
2. [Task 1: Data Preparation and Statistical Analysis](#task-1-data-preparation-and-statistical-analysis)
3. [Task 2: Feature Selection and Outlier Detection](#task-2-feature-selection-and-outlier-detection)
4. [Task 3: Clustering](#task-3-clustering)
5. [Task 4: Classification – Implementation with scikit-learn](#task-4-classification--implementation-with-scikit-learn)
6. [Task 5: Classification – Implementation with Keras/TensorFlow](#task-5-classification--implementation-with-kerastensorflow)
7. [References](#references)

---

## Introduction

Data mining and machine learning are essential components of modern data analysis, allowing for the extraction of valuable insights from large datasets. This project aims to familiarize us with data mining techniques and machine learning algorithms using popular tools such as **scikit-learn** and **TensorFlow** in Python.

For this analysis, we utilized the **Adult Income Dataset** from the **UCI Machine Learning Repository**, which contains demographic and occupational attributes of 48,842 individuals. The objective is to classify individuals based on their income level as either **above or below $50,000 per year**.

---

## Task 1: Data Preparation and Statistical Analysis

The first phase involved preprocessing and analyzing the dataset:

- **Data cleaning** (handling missing values, removing outliers)
- **Data normalization** (MinMaxScaler & Log Transformation)
- **Encoding categorical variables** using **Label Encoding**
- **Dimensionality reduction** using **Principal Component Analysis (PCA)**

After preprocessing, our dataset was clean, balanced, and ready for analysis.

---

## Task 2: Feature Selection and Outlier Detection

We applied **SelectKBest** with **ANOVA F-statistic** to select the most relevant features. Additionally, outliers were detected using **Z-score threshold = 3** and visualized through **box plots**.

We removed variables that had little significance or added noise, such as **fnlwgt**.

---

## Task 3: Clustering

We implemented two clustering techniques:

- **K-Means**
- **Agglomerative Clustering**

The clustering performance was evaluated using the **Silhouette Score**:

| Algorithm       | Clusters | Silhouette Score |
|---------------|---------|----------------|
| K-Means      | 2       | 0.6166         |
| Agglomerative | 2       | 0.6936         |

Agglomerative Clustering achieved a better score. However, clustering did not effectively separate the income classes.

---

## Task 4: Classification – Implementation with scikit-learn

We tested four classification algorithms:

| Model                   | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|----------|--------|----------|
| Logistic Regression | 80.96%   | 79.53%   | 80.96% | 79.26%   |
| Random Forest       | 84.13%   | 83.79%   | 84.13% | 83.87%   |
| Support Vector Machine | 77.86% | 75.28% | 77.86% | 74.90% |
| Gradient Boosting   | **86.18%** | **85.62%** | **86.18%** | **85.52%** |

The **Gradient Boosting** model outperformed all others, achieving the highest accuracy and performance.

---

## Task 5: Classification – Implementation with Keras/TensorFlow

We trained a **Multi-Layer Perceptron (MLP)** with three different neuron configurations:

| Model       | Accuracy |
|------------|----------|
| MLP_16-8   | 83.90%   |
| MLP_32-16  | 84.01%   |
| **MLP_64-32** | **84.87%** |

The model with **64-32 neurons** performed the best. We observed that increasing the number of neurons improved accuracy, but beyond a certain point, it had diminishing returns.

The **accuracy/loss curves** indicate that all models converged correctly without significant overfitting.

---

## References

For the completion of this project, we relied on the following sources:

- **Lecture notes and slides** from the course
- **GitHub** (repositories related to similar projects)
- **Stack Overflow** (troubleshooting and code optimization)

This project provided hands-on experience in applying data mining and machine learning techniques on real-world datasets, enhancing our skills in statistical analysis and programming.

---

**End of Report**
