import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from flask import Flask, render_template, request

# Load the pre-trained model, preprocessors, and metrics
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')

# Load pre-calculated metrics
metrics = joblib.load('metrics.joblib')
mse = metrics['mse']
score = metrics['score']

# Flask web app
app = Flask(__name__)

def preprocess_input_data(form_data):
    temperature = float(form_data['temperature'])
    humidity = float(form_data['humidity'])
    square_footage = float(form_data['square_footage'])
    occupancy = int(form_data['occupancy'])
    hvac_usage = form_data['hvac_usage']
    lighting_usage = form_data['lighting_usage']

    # Preprocess numerical data
    numerical_data = [[temperature, humidity, square_footage, occupancy]]
    scaled_numerical_data = scaler.transform(numerical_data)
    
    # Preprocess categorical data
    categorical_data = [[hvac_usage, lighting_usage]]
    encoded_data = encoder.transform(categorical_data).toarray()

    # Combine features into a single DataFrame
    numerical_features = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy']
    combined_data = pd.concat([pd.DataFrame(scaled_numerical_data, columns=numerical_features),
                               pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['HVACUsage', 'LightingUsage']))], axis=1)

    return combined_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            new_data_df = preprocess_input_data(request.form)
            energy_usage = model.predict(new_data_df)[0]
            return render_template('index.html', energy_usage=energy_usage, mse=mse, score=score)
        except Exception as e:
            return render_template('index.html', error=str(e), mse=mse, score=score)
    
    return render_template('index.html', mse=mse, score=score)

if __name__ == '__main__':
    app.run(port=5001)