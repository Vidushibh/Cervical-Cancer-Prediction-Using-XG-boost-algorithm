## Cervical Cancer Prediction Using XG-boost algorithm

## Overview:
Every year Cervical cancer kills about 4,000 women in the U.S. and about 300,000 women worlwide. 
The death rate can be reduced if we can increase the medical screening. So, the goal of this project is to predict 
cervical cancer in 858 patients based on the input features like Age, Number of Pregnancies,their smoking habits, 
IUD, STDs etc. 


This repository contains Python code for performing exploratory data analysis (EDA), data visualization, and machine learning model training for predicting cervical cancer.

## Objective
The objective of this project is to analyze the cervical cancer dataset, perform data preprocessing, visualize the data distribution, and train a machine learning model to predict the occurrence of cervical cancer based on various features.

## Dataset
This dataset was collected at 'Hospital Universitario de Caracas' in Caracas, Venezuela and 
contains demographic information, habits and historic medical records of 858 patients.

## Libraries Used
- Pandas: Data manipulation and analysis library.
- NumPy: Numerical computing library for handling arrays and matrices.
- Seaborn: Data visualization library based on Matplotlib, used for creating attractive and informative statistical graphics.
- Matplotlib: Plotting library for creating visualizations in Python.
- Plotly: Interactive visualization library for creating plots and dashboards.
- Scikit-learn: Machine learning library for data preprocessing, modeling, and evaluation.
- XGBoost: Gradient boosting library for building and optimizing machine learning models.

## What's Included
- `Cervical_Cancer_Prediction.ipynb`: Jupyter Notebook containing the code for EDA, data visualization, and machine learning model training.
- `cervical_cancer.csv`: Dataset containing information related to cervical cancer.
- `README.md`: Readme file providing an overview of the project and usage instructions.

## Overview of tasks performed
- The dataset contains various features related to cervical cancer, such as age, number of sexual partners, STD history, and biopsy results.
- Exploratory data analysis is performed to understand the data distribution, identify missing values, and visualize correlations between features.
- Data preprocessing techniques such as handling missing values, scaling features, and splitting the dataset into training and testing sets are applied.
- An XGBoost classifier model is trained on the preprocessed data to predict the occurrence of cervical cancer.
- Model evaluation metrics such as accuracy, precision, recall, and confusion matrix are used to assess the performance of the trained model.
