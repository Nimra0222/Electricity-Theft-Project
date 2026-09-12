# Imports
import dash
from dash import dcc, html
import pandas as pd
import joblib
from dash.dependencies import Input, Output
import numpy as np
import sys
import os

# Add the project root to sys.path so Python can find the 'utils' and 'data' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import preprocess_data
from utils.model_utils import train_model, evaluate_model
from data.weather_api import fetch_weather_data

# Constants for weather API
API_KEY = "92b8d643e46c01a5fa98580ba0324b94"
LAT, LON = 24.8607, 67.0011  # Karachi

# Initialize Dash app
app = dash.Dash(__name__)

# Load raw consumption data (long format)
# Load raw data with second row skipped
raw_df = pd.read_csv('data/Electricity_Theft_Data.csv', skiprows=[1])

# Normalize column names
raw_df.columns = raw_df.columns.str.strip().str.upper()

# Print to confirm correct structure
# print("Columns loaded:", raw_df.columns.tolist())
 # debug print
# Drop rows with no consumption data
raw_df.dropna(how='all', subset=raw_df.columns[1:], inplace=True)

# Melt the wide-format data to long format
data = raw_df.melt(id_vars=['CONS_NO'], var_name='date', value_name='consumption')

# Convert 'date' column to datetime
# Convert 'date' column to datetime with specified format
data['date'] = pd.to_datetime(data['date'], errors='coerce')



# Drop rows with invalid dates or missing consumption values
data.dropna(subset=['date', 'consumption'], inplace=True)

# Ensure numeric values
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', errors='coerce')
data.dropna(subset=['consumption'], inplace=True)


# Get unique consumer IDs
unique_users = data['CONS_NO'].dropna().unique()
user_options = [{'label': str(u), 'value': u} for u in unique_users]

# Dash layout with dropdown, graph, and output text
app.layout = html.Div([
    html.H2("Electricity Consumption & Theft Detection Dashboard"),
    dcc.Dropdown(id='user-select', options=user_options, value=unique_users[0]),
    dcc.Graph(id='consumption-graph'),
    html.Div(id='anomaly-output')
])
@app.callback(
    Output('consumption-graph', 'figure'),
    Output('anomaly-output', 'children'),
    Input('user-select', 'value')
)
def update_graph(user):
    # Filter data for selected user
    df = data[data['CONS_NO'] == user].copy()

    # Fetch current weather data
    weather = fetch_weather_data(LAT, LON, API_KEY)

    # Prepare a dataframe for training (wide format)
    pivot_df = df.pivot(index='date', columns='CONS_NO', values='consumption')
    pivot_df.fillna(0, inplace=True)

    # Save to temp and preprocess
    temp_path = 'data/temp_user_pivot.csv'
    pivot_df.to_csv(temp_path)
    X, y, _ = preprocess_data(temp_path, weather)
    if isinstance(X, pd.DataFrame):
        index_used = X.index
    else:
        index_used = pivot_df.index[:len(X)]

    # Train model and get predictions
    model = train_model(X, y, 'model/model.pkl')
    predictions = model.predict(X)

    # Ensure predictions are 1D
    if len(predictions.shape) > 1:
        predictions = predictions.ravel()

    # Use index_used to get anomaly dates
    anomaly_dates = index_used[predictions == 1]

    df['label'] = df['date'].isin(anomaly_dates).astype(int)



    # Plot the consumption with anomaly markers
    fig = {
        'data': [
            {'x': df['date'], 'y': df['consumption'], 'type': 'line', 'name': 'Consumption'},
            {'x': df[df['label'] == 1]['date'], 'y': df[df['label'] == 1]['consumption'],
             'type': 'scatter', 'mode': 'markers', 'name': 'Anomaly',
             'marker': {'color': 'red', 'size': 10}}
        ],
        'layout': {'title': f'Consumption Pattern for User {user}'}
    }
    print(f"Type of X: {type(X)}")
    if isinstance(X, pd.DataFrame):
        print(f"Columns in X: {X.columns.tolist()}")
    else:
        print(f"X is not a DataFrame, shape: {X.shape}")

    # Construct anomaly explanation table
    anomalies = df[df['label'] == 1].set_index('date')
    if isinstance(X, pd.DataFrame):
        anomalies = anomalies.join(X[['temp', 'humidity']], how='left')
    else:
        # Assign default or fetch from weather dict directly
        anomalies['temp'] = weather.get('temp', np.nan)
        anomalies['humidity'] = weather.get('humidity', np.nan)
  # Ensure weather is in X

    def explain_anomaly(row):
        reasons = []
        if row['consumption'] > df['consumption'].mean() * 1.5:
            reasons.append("Unusually high consumption")
        if row['temp'] < 25 and row['consumption'] > df['consumption'].mean():
            reasons.append("High usage despite cool weather")
        return ", ".join(reasons) if reasons else "Pattern deviation"

    anomalies['Reason'] = anomalies.apply(explain_anomaly, axis=1)

    if anomalies.empty:
        summary = html.P("No anomalies detected.")
    else:
        summary = html.Div([
            html.H4("Anomaly Details:"),
            html.Table([
                html.Tr([html.Th("Date"), html.Th("Consumption (kWh)"), html.Th("Temp (°C)"),
                         html.Th("Humidity (%)"), html.Th("Reason")])
            ] + [
                html.Tr([
                    html.Td(date.strftime("%Y-%m-%d")),
                    html.Td(f"{row['consumption']:.2f}"),
                    html.Td(f"{row['temp']:.1f}"),
                    html.Td(f"{row['humidity']:.1f}"),
                    html.Td(row['Reason'])
                ])
                for date, row in anomalies.iterrows()
            ])
        ])

    return fig, summary

# @app.callback(
#     Output('consumption-graph', 'figure'),
#     Output('anomaly-output', 'children'),
#     Input('user-select', 'value')
# )
# def update_graph(user):
#     # Filter data for selected user
#     df = data[data['CONS_NO'] == user].copy()

#     # Fetch current weather data
#     weather = fetch_weather_data(LAT, LON, API_KEY)

#     # Prepare a dataframe for training (wide format)
#     pivot_df = df.pivot(index='date', columns='CONS_NO', values='consumption')
#     pivot_df.fillna(0, inplace=True)
#     # pivot_df.to_csv('data/Electricity_Theft_Data.csv')  # Save for preprocessing

#     # # Preprocess the data using helper function (adds weather features)
#     # X, y, _ = preprocess_data('data/Electricity_Theft_Data.csv', weather)
#     temp_path = 'data/temp_user_pivot.csv'
#     pivot_df.to_csv(temp_path)

#     X, y, _ = preprocess_data(temp_path, weather)
#     # Train model and predict
#     model = train_model(X, y, 'model/model.pkl')
#     predictions = model.predict(X)

#     # Flatten if needed
#     if len(predictions.shape) > 1:
#         predictions = predictions.ravel()
#     print("Pivot DataFrame shape:", pivot_df.shape)
#     print("Preprocessed X shape:", X.shape)
#     print("Predictions shape:", predictions.shape)
#     # print("Index sample of X:", X.index[:5])
#     print("Index sample of pivot_df:", pivot_df.index[:5])

#     # Align predictions with index of X (which should be same as index returned from preprocess)
#     if isinstance(X, pd.DataFrame):
#         index_used_for_prediction = X.index
#         print("Index sample of X (DataFrame):", X.index[:5])
#     else:
#         index_used_for_prediction = pivot_df.index[:len(predictions)]
#         print("Index sample of X (array fallback):", index_used_for_prediction[:5])
# # Set label = 1 for dates where predictions == 1
#     anomaly_dates = index_used_for_prediction[predictions == 1]
#     df['label'] = df['date'].isin(anomaly_dates).astype(int)


#     # Assign predicted labels to original dataframe for highlighting
#     # df['label'] = 0
#     # df.loc[df['date'].isin(pivot_df.index[predictions == 1]), 'label'] = 1

#     # Plot consumption
#     fig = {
#         'data': [
#             {'x': df['date'], 'y': df['consumption'], 'type': 'line', 'name': 'Consumption'},
#             {'x': df[df['label'] == 1]['date'], 'y': df[df['label'] == 1]['consumption'], 'type': 'scatter', 'mode': 'markers', 'name': 'Anomaly', 'marker': {'color': 'red', 'size': 10}}
#         ],
#         'layout': {'title': f'Consumption Pattern for User {user}'}
#     }

#     # Output anomaly summary
#     suspicious_count = df['label'].sum()
#     alert_msg = f"{suspicious_count} potential theft instances detected based on consumption and weather." if suspicious_count else "No anomalies detected."

#     return fig, alert_msg

if __name__ == '__main__':
    app.run(debug=True)


