import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'random_forest_price_model.joblib')
model = joblib.load(model_path)

# Initialize Flask app
app = Flask(__name__)

# Configure rate limiter to restrict to 1 request per second per user
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["2 per second"]
)

# Home route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API endpoint at http://127.0.0.1:3000/predict
@app.route('/predict', methods=['POST'])
@limiter.limit("2 per second")
def predict():
    """
    Endpoint for predicting car price based on user input.
    Accessible via POST request at http://127.0.0.1:3000/predict
    """
    # Retrieve user input
    mileage = request.form.get('mileage')
    tax = request.form.get('tax')
    mpg = request.form.get('mpg')
    engineSize = request.form.get('engineSize')
    price_per_mileage = request.form.get('price_per_mileage')
    tax_to_price_ratio = request.form.get('tax_to_price_ratio')

    # Check if all fields are provided
    if not all([mileage, tax, mpg, engineSize, price_per_mileage, tax_to_price_ratio]):
        return jsonify({"error": "Please enter all data"}), 400

    try:
        # Convert input data to float
        input_data = np.array([[float(mileage), float(tax), float(mpg), float(engineSize),
                                float(price_per_mileage), float(tax_to_price_ratio)]])

        # Predict using the loaded model
        prediction = model.predict(input_data)

        # Return the prediction result as JSON
        return jsonify({"predicted_price": round(prediction[0], 2)})

    except ValueError:
        return jsonify({"error": "Invalid input. Please ensure all values are numbers."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app on port 3000
if __name__ == '__main__':
    app.run(port=3000, debug=True)
