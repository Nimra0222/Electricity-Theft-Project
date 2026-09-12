from utils.preprocess import preprocess_data
from utils.model_utils import train_model, evaluate_model
from data.weather_api import fetch_weather_data

API_KEY = "92b8d643e46c01a5fa98580ba0324b94"
LAT, LON = 24.8607, 67.0011  # Karachi

# Step 1: Fetch weather
weather = fetch_weather_data(LAT, LON, API_KEY)

# Step 2: Preprocess data
X, y, scaler = preprocess_data('data/Electricity_Theft_Data.csv', weather)

# Step 3: Train and evaluate
model = train_model(X, y, 'model/model.pkl')
evaluate_model(model, X, y)
