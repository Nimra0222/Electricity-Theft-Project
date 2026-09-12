import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(consumption_path, weather_data):
    df = pd.read_csv(consumption_path, index_col=0)
    df.replace('', np.nan, inplace=True)
    df.dropna(how='all', inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)

    X = df.T  # transpose to make rows = dates
    y = np.where(X.mean(axis=1) > 5, 1, 0)  # simple theft label

    # Add weather features
    weather_df = pd.DataFrame([weather_data] * len(X), index=X.index)
    X['temp'] = weather_df['temp']
    X['humidity'] = weather_df['humidity']
    X['pressure'] = weather_df['pressure']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
