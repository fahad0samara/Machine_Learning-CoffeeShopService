


# CoffeeShopServiceRatingPredictor

## Overview

The **CoffeeShopServiceRatingPredictor** is a comprehensive machine learning project aimed at predicting service ratings for coffee shops. The project encompasses data preprocessing, model training using Random Forest, performance evaluation, and a user-friendly interface built with Streamlit.

## Features

- **Data Preprocessing**: Clean and prepare the dataset for modeling.
- **Model Training**: Use Random Forest Regressor to predict service ratings.
- **Model Evaluation**: Assess model performance with metrics such as Mean Squared Error (MSE) and R-squared (R²).
- **Hyperparameter Tuning**: Optimize model performance using Grid Search.
- **Visualization**: Generate scatter plots comparing actual vs. predicted ratings.
- **User Interface**: Interactive web application built with Streamlit for model predictions and visualizations.



## Data

The dataset used in this project is the "Coffee Shop Service Ratings" dataset. It contains various features related to coffee shop operations and customer feedback. The target variable is `Service Rating`.

### Data Preprocessing

1. **Load Data**:
   ```python
   import pandas as pd
   data = pd.read_csv('data/coffee_shop.csv')
   ```

2. **Preprocess Data**:
   - Drop unnecessary columns.
   - One-hot encode categorical variables.
   - Split data into training and testing sets.

### Model Training

1. **Initialize and Train Model**:
   ```python
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.model_selection import train_test_split

   features = data.drop(columns=['Service Rating'])
   target = data['Service Rating']
   features_encoded = pd.get_dummies(features, drop_first=True)
   X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

   model = RandomForestRegressor(random_state=42)
   model.fit(X_train, y_train)
   ```

2. **Save Model**:
   ```python
   import joblib
   joblib.dump(model, 'random_forest_model.joblib')
   ```

### Model Evaluation

1. **Make Predictions**:
   ```python
   predictions = model.predict(X_test)
   ```

2. **Calculate Metrics**:
   ```python
   from sklearn.metrics import mean_squared_error, r2_score

   mse = mean_squared_error(y_test, predictions)
   r2 = r2_score(y_test, predictions)
   ```

3. **Cross-Validation and Hyperparameter Tuning**:
   - Perform cross-validation to assess model performance.
   - Use Grid Search for hyperparameter tuning to find the best model configuration.

### User Interface with Streamlit

1. **Create Streamlit App**:
   - Build a web application to interact with the model, input new data, and view predictions.
   - Visualize the results using Streamlit’s interactive components.

2. **Streamlit Code Example**:
   ```python
   import streamlit as st
   import joblib
   import pandas as pd

   # Load model
   model = joblib.load('random_forest_model.joblib')

   st.title('Coffee Shop Service Rating Predictor')

   # User inputs
   feature1 = st.number_input('Feature 1')
   feature2 = st.number_input('Feature 2')
   # Add more inputs as needed

   # Predict button
   if st.button('Predict'):
       input_data = pd.DataFrame([[feature1, feature2]], columns=['Feature 1', 'Feature 2'])
       prediction = model.predict(input_data)
       st.write(f'Predicted Service Rating: {prediction[0]}')
   ```


   ```

2. **Interacting with the App**:
   - Open the Streamlit app in your browser.
   - Enter values for the features and click the "Predict" button to get the service rating prediction.

## Results

The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R²). The cross-validation results and best parameters from Grid Search are also documented to ensure the model’s reliability.


## Contributing

Contributions are welcome! Please submit pull requests or open issues if you have suggestions or encounter problems.

