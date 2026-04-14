# netflix-movie-rating-prediction
Regression models predicting user movie ratings using large-scale Netflix-style rating datasets.

# Netflix Movie Rating Prediction

This project builds regression models to predict user movie ratings using a large-scale movie rating dataset.

## Overview

The goal of this project is to predict how a user would rate a movie based on historical rating patterns.

The dataset contains hundreds of thousands of user-movie interactions including:

- user ID
- movie ID
- rating
- timestamp

## Methods

The project includes:

- parsing and structuring raw rating data
- feature engineering
- regression modeling
- model evaluation

Models explored include:

- Linear Regression
- Regularized Regression (Ridge / Lasso)
- Tree-based models

## Evaluation

Model performance is evaluated using:

- Root Mean Squared Error (RMSE)

## Tools

- Python
- Pandas
- Scikit-learn
- NumPy
- Matplotlib

## Project Structure

netflix-movie-rating-prediction/
├── data/
├── notebooks/
├── model/
├── figures/
├── requirements.txt
└── README.md
