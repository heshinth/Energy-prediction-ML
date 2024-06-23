import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

# Load and preprocess data (same as in your original script)
data = pd.read_csv('data.csv')

numerical_features = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy']
categorical_features = ['HVACUsage', 'LightingUsage']

encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(data[categorical_features]).toarray()
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

scaler = StandardScaler()
numerical_data = data[numerical_features]
scaled_numerical_data = scaler.fit_transform(numerical_data)

X = pd.concat([pd.DataFrame(scaled_numerical_data, columns=numerical_features), encoded_df], axis=1)
y = data['EnergyConsumption']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

# Save the model, scaler, encoder, and metrics
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(encoder, 'encoder.joblib')
joblib.dump({'mse': mse, 'score': score}, 'metrics.joblib')