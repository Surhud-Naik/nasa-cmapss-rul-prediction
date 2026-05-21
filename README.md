# NASA CMAPSS RUL Prediction

Machine Learning project for predicting the **Remaining Useful Life (RUL)** of aircraft engines using the NASA CMAPSS turbofan engine degradation dataset.

## Overview

Predictive maintenance is a critical application of machine learning in aerospace and industrial systems.  
This project focuses on estimating the **Remaining Useful Life (RUL)** of aircraft engines from multivariate sensor data.

Using the NASA CMAPSS dataset, different machine learning techniques were explored to model engine degradation patterns and predict failure timelines.

The project was completed as part of the **ME228 Course Project**.

---

## Problem Statement

Aircraft engines generate large amounts of operational sensor data during runtime.  
The objective of this project is to:

- Analyze engine sensor behavior over time
- Identify degradation trends
- Predict the remaining operational cycles before engine failure
- Evaluate machine learning models for RUL estimation

---

## Dataset

Dataset used: **NASA CMAPSS Turbofan Engine Degradation Simulation Dataset**

The dataset contains:
- Multiple engine units
- Operational settings
- Time-series sensor measurements
- Simulated degradation trajectories

Source:
- NASA Prognostics Data Repository

---

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Sensor trend visualization
- Feature engineering
- Remaining Useful Life (RUL) labeling
- Machine learning model training
- Performance evaluation and comparison

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Project Workflow

1. Data Loading
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. RUL Calculation
6. Model Training
7. Performance Evaluation
8. Visualization of Results

---

## Model Performance

The trained models were evaluated using regression metrics such as:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

---

## Repository Structure

```bash
├── data/
├── notebooks/
├── src/
├── results/
├── README.md
└── requirements.txt