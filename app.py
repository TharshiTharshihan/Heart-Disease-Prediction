from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('heart_disease_model.pkl', 'rb'))

@app.route('/')
def first():
    
    return render_template('first.html')


@app.route('/calculate', methods=['Get'])
def home():
     return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data 
        input_data = [float(x) for x in request.form.values()]
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(input_data_as_numpy_array)[0]
        result = "The person has heart disease" if prediction == 1 else "The person does not have heart disease"
        
        return render_template('result.html', prediction_text=result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
