import pickle
import pandas as pd
import numpy
from flask import Flask, request, jsonify

# --- Your Classifier Class (with a small path modification) ---
class Soil_quality_Classifier:
    def __init__(self, model_path='random_forest_pkl.pkl'):
        """
        Initializes the classifier by loading the pickled model.
        The model_path is now a parameter for flexibility.
        """
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The model file was not found at {model_path}. Make sure it's in the same directory as app.py.")
        except Exception as e:
            raise RuntimeError(f"Error loading the model: {e}")

    def preprocessing(self, input_data):
        """Converts JSON/dict input to a pandas DataFrame."""
        # The model expects data in a specific format, ensure the input matches.
        input_data = pd.DataFrame(input_data, index=[0])
        return input_data

    def predict(self, input_data):
        """Makes a prediction using the loaded model."""
        return self.model.predict(input_data)

    def postprocessing(self, prediction):
        """Converts the numeric prediction into a human-readable category."""
        # Ensure the prediction is an integer index
        if isinstance(prediction, numpy.ndarray):
            prediction_index = int(prediction[0])
        else:
            prediction_index = int(prediction)
            
        categories = ["Less Fertile", "Fertile", "Highly Fertile"]
        
        if 0 <= prediction_index < len(categories):
            return categories[prediction_index]
        else:
            return "Unknown Category"

    def compute_prediction(self, input_data):
        """
        Runs the full prediction pipeline: preprocessing, prediction, and postprocessing.
        """
        try:
            processed_data = self.preprocessing(input_data)
            raw_prediction = self.predict(processed_data)
            final_prediction = self.postprocessing(raw_prediction)
            return {"status": "Success", "prediction": final_prediction}
        except Exception as e:
            # Return a structured error message
            return {"status": "Error", "message": str(e)}

# --- Flask Application ---

# 1. Initialize the Flask App
app = Flask(__name__)

# 2. Load the Model
# This creates a single, global instance of our classifier
try:
    classifier = Soil_quality_Classifier()
    print("Model loaded successfully.")
except Exception as e:
    classifier = None
    print(f"FATAL: Could not load model. Error: {e}")


# 3. Define the API Endpoint
@app.route('/')
def home():
    """A simple home route to check if the API is running."""
    return "<h1>Soil Quality Classifier API</h1><p>Send a POST request to /predict.</p>"

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    The main prediction route. It expects JSON data in the request body.
    """
    if not classifier:
         return jsonify({"status": "Error", "message": "Model is not loaded. Check server logs."}), 500

    if not request.is_json:
        return jsonify({"status": "Error", "message": "Request must be JSON."}), 400
        
    # Get data from the POST request
    input_data = request.get_json()
    
    # Use the classifier to compute the prediction
    result = classifier.compute_prediction(input_data)
    
    # Return the result as JSON
    if result['status'] == 'Error':
        return jsonify(result), 400 # Bad request if processing fails
    
    return jsonify(result)

# 4. Run the App
# This block is not used by production servers like Gunicorn, but it's
# good for local testing (python app.py)
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your local network
    # Render will use its own server (Gunicorn) to run the app.
    app.run(host='0.0.0.0', port=8080, debug=True)
