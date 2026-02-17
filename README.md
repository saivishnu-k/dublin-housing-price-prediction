# Dublin Housing Price and Purchase Prediction

Predictive analytics project analysing housing listings in Dublin to understand price drivers and buyer behaviour.  
The project applies data cleaning, feature engineering, regression modelling, and classification techniques to support informed decision making for buyers and developers.

## Overview
The Dublin property market is influenced by multiple economic, regulatory, and structural factors.  
This project explores how location, size, building energy rating, and property features affect housing prices and purchase decisions.

Using a dataset of over 13,000 listings, the analysis combines exploratory data analysis, predictive modelling, and model optimisation to generate practical insights.

## Dataset
- 13,320 records
- 11 features
- Mix of numerical and categorical variables

Key attributes:
- Location
- Bedrooms and bathrooms
- Total square footage
- Balcony
- Building Energy Rating (BER)
- Renovation status
- Property scope
- Price per square foot
- Buying status

The original dataset is not included in this repository.

## Data Preparation
- Removed duplicate records (98 entries)
- Dropped missing values
- Normalised text fields using regular expressions
- Cleaned inconsistent entries in size and square footage
- Converted BER ratings from A–G to ordinal numeric scale
- Processed value ranges using mean imputation
- Removed extreme outliers using IQR method
- Standardised categorical variables

## Feature Engineering
- Created `total_price` as:
  
  total_sqft × price_per_sqft

- Generated polynomial features (degree 2) for numerical variables
- Applied one hot encoding to categorical features
- Scaled numerical features using StandardScaler

## Exploratory Data Analysis
Key insights:
- Strong correlation between total price and square footage (0.99)
- High correlation between bathrooms and bedrooms (0.89)
- Fingal and South Dublin show lower median prices
- BER and renovation status show limited price impact
- Total square footage is the strongest price predictor

## Regression Modelling

### Linear Regression (Polynomial Features)
- MAE: 177,430
- RMSE: 244,678
- R²: 0.563

Key drivers:
- Total square footage
- Bathroom count
- Interaction terms

### Random Forest Regressor (Baseline)
- MAE: 175,381
- RMSE: 245,593
- R²: 0.560

### Tuned Random Forest Regressor
Optimised using RandomizedSearchCV.

Best parameters:
- n_estimators: 300
- max_depth: 10
- min_samples_split: 10
- min_samples_leaf: 2
- max_features: sqrt

Performance:
- MAE: 168,507
- RMSE: 234,600
- R²: 0.599

This model provided the best overall regression performance.

## Classification Modelling (Buying Prediction)

### Logistic Regression
- Accuracy: 67.2%
- Minority class recall: 22%
- Strong bias toward "Not Buying"

Balanced version:
- Accuracy: 64.1%
- Minority class recall: 32%

### Random Forest Classifier
- Accuracy: 63.6%
- Minority class recall: 25%

### XGBoost Classifier
- Accuracy: 61.8%
- Minority class recall: 45
- F1 score (Buying): 43

XGBoost achieved the best performance for identifying buyers, at the cost of reduced overall accuracy.

## Key Findings
- Total square footage is the dominant price driver
- Bedroom and bathroom counts strongly influence value
- Location remains a major determinant
- BER and renovation status have limited impact
- Tuned Random Forest performs best for price prediction
- XGBoost performs best for identifying buyers
- Class imbalance significantly affects classification models

## Files
- `analysis.py`  
  End to end preprocessing and modelling pipeline

- `report.pdf`  
  Academic project report

## Skills Demonstrated
- Exploratory data analysis
- Data cleaning and preprocessing
- Feature engineering
- Regression modelling
- Classification modelling
- Hyperparameter tuning
- Imbalanced data handling
- Model interpretation
- Python and scikit learn

## Author
Sai Vishnu Kandagattla  
MSc Business Analytics  
University College Cork  
Cork, Ireland  

LinkedIn: https://www.linkedin.com/in/sai-kandagattla/
