from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load("path/to/trained_model.pkl")

# Initialize Flask application
app = Flask(__name__)

# Define endpoint for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the request
    data = request.json
    
    # Preprocess input data (if necessary)
    # Note: Make sure the preprocessing steps match those used during model training
    
    # Make predictions using the model
    predictions = model.predict(data)
    
    # Return predictions as JSON response
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    # Run the Flask application
    app.run(host="0.0.0.0", port=5000)