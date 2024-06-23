import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from flask import Flask, render_template, request

# Load the data
data = pd.read_csv('Energy_consumption.csv')

# Define features
numerical_features = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy']
categorical_features = ['HVACUsage', 'LightingUsage']

# Preprocess categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(data[categorical_features]).toarray()
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Preprocess numerical features
scaler = StandardScaler()
numerical_data = data[numerical_features]
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Combine features
X = pd.concat([pd.DataFrame(scaled_numerical_data, columns=numerical_features), encoded_df], axis=1)
y = data['EnergyConsumption']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

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
    combined_data = pd.concat([pd.DataFrame(scaled_numerical_data, columns=numerical_features),
                               pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))], axis=1)

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
    app.run(debug=True, port=5001)
