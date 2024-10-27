## Project Documentation: Car Price Prediction with Random Forest and Flask API

### Overview

This project aims to predict car prices based on various features using a Random Forest regression model. The project is divided into three main parts:
1. **Data Processing and Model Training**: This Python code processes car data, performs feature engineering, and trains a Random Forest model.
2. **Flask API**: A REST API using Flask to serve predictions from the trained model.
3. **HTML Frontend**: A simple HTML form to collect input from users and interact with the API for predictions.

---

### 1. Data Processing and Model Training (Python Code)

#### Dependencies

Make sure to install the required packages using:

```bash
pip install joblib numpy pandas seaborn matplotlib scikit-learn
```

#### Code Walkthrough

1. **Imports and Warnings Suppression**:
   - Imports necessary libraries for data manipulation, model training, and evaluation.
   - `warnings.filterwarnings("ignore")` is used to suppress any warning messages that might clutter the output.

2. **Data Loading and Basic Inspection**:
   ```python
   file_path = r'path\\CarStory (2).csv'
   df = pd.read_csv(file_path)
   print(df.head())
   print(df.info())
   print(df.describe())
   ```
   - Loads the data from a CSV file and performs basic inspection to understand the structure and check for missing values.

3. **Data Cleaning**:
   ```python
   df['tax'] = df['tax'].fillna(df['tax(£)'])
   df.drop(columns=['tax(£)', 'Unnamed: 0', 'model'], inplace=True)
   ```
   - Fills missing values in `tax` using the `tax(£)` column.
   - Drops unnecessary columns (`tax(£)`, `Unnamed: 0`, `model`).

4. **Missing Value Imputation for `tax`**:
   - Fills missing values in `tax` based on the median tax grouped by `engineSize` and `year`.

5. **Removing Future Data**:
   - Filters out rows where the `year` is greater than the current year to maintain realistic data.

6. **Additional Cleaning Steps**:
   - Fills any remaining missing values with the median.
   - Converts categorical columns to lowercase.

7. **Save Processed Data**:
   - Saves the cleaned data to a new CSV file named `Processed_CarData.csv`.

8. **Data Normalization**:
   - Normalizes numeric columns (`mileage`, `tax`, `mpg`, `engineSize`) to a range of 1-10 for easier processing.
   - Saves the normalized data to `Car_Normalized.csv`.

9. **Feature Engineering for Clustering and Regression**:
   - Creates `price_per_mileage` and `tax_to_price_ratio` as derived features.
   - These features are useful for clustering and serve as inputs for regression.

10. **KMeans Clustering**:
This is not the correct approach however its a demo of how clustering work
    - Performs clustering on `tax_to_price_ratio` and `price_per_mileage`.
    - Uses `StandardScaler` to scale the features before clustering.
    - Calculates the silhouette score to evaluate clustering quality and displays the results.


11. **Train-Test Split**:
    - Splits the data into training and testing sets with a 70-30 ratio.

12. **Grid Search for Hyperparameter Tuning**:
    ```python
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5],
        'min_samples_split': [3],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
    grid_search.fit(X_train, y_train)
    ```
    - Conducts hyperparameter tuning for `RandomForestRegressor` using Grid Search with a limited parameter grid for efficiency.

13. **Train Final Model with Best Parameters**:
    - Trains a Random Forest model using the best parameters from Grid Search.
    - Evaluates the model using R-squared, Mean Absolute Error (MAE), and Mean Squared Error (MSE).

14. **Save the Model**:
    - Saves the trained model as `random_forest_price_model.joblib`.

---

### 2. Flask API (app.py)

To deploy this model, a Flask app is used to create an API endpoint that accepts input features and returns the predicted car price.

#### Flask Setup

1. **Flask App Initialization**:
   ```python
   app = Flask(__name__)
   ```

2. **Rate Limiter Configuration**:
   ```python
   limiter = Limiter(
       get_remote_address,
       app=app,
       default_limits=["10 per second"]
   )
   ```
   - Restricts each user to a maximum of 10 requests per second.

3. **Prediction Endpoint**:
   - The `/predict` endpoint is configured to accept POST requests.
   - Extracts input features from the form data, converts them to floats, and reshapes them for the model.
   - Returns the predicted price as a JSON response.

4. **Running the App**:
   - The Flask app runs locally on `http://127.0.0.1:3000`.

#### Running the Flask App

Run the Flask app using:
```bash
python app.py
```

You can access the API at:
```
http://127.0.0.1:3000/predict
```

---

### 3. HTML and CSS (Frontend)

The HTML form allows users to input values for prediction and submit them to the `/predict` endpoint.

#### Directory Structure

```
CARTEST/
├── templates/
│   └── index.html     # HTML file for the form
├── app.py             # Flask API
├── random_forest_price_model.joblib # Saved model
```

#### `index.html`

This file is placed in the `templates` folder. It includes a form for inputting data and JavaScript to submit the form data as an AJAX request to the Flask API.


---

### Conclusion

This project covers end-to-end data processing, model training, and deployment using Flask and an HTML interface. The key components are:

1. **Data Preparation**: Cleaning, feature engineering, normalization.
2. **Model Training**: Random Forest with hyperparameter tuning.
3. **API Deployment**: Flask API with rate limiting.
4. **User Interface**: HTML form for user input and JavaScript for async requests.

contact zaianabdelrahman@gmail.com