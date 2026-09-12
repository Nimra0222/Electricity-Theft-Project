# ⚡ Electricity Theft Detection System
This is an academic machine learning project  focused on forecasting energy consumption using historical usage data. The goal is to help with better energy planning, grid stability, and eco-friendly energy use by predicting future consumption patterns from past data.
An end-to-end machine learning system that detects potential electricity theft by analyzing consumer usage patterns alongside real-time weather data, with results surfaced through an interactive Dash dashboard.

## 📋 Description

Electricity theft — through meter tampering, illegal connections, or bypassing — causes significant revenue loss for utility providers. This project builds a data-driven anomaly detection pipeline that flags suspicious consumption behavior by combining historical electricity usage records with contextual weather data (temperature, humidity, pressure), on the theory that genuine consumption should correlate with weather patterns while theft/tampering often doesn't.

The system:
- Ingests daily electricity consumption data for thousands of consumers
- Fetches live weather data for the region via the OpenWeatherMap API
- Engineers a feature set combining usage statistics with weather variables
- Trains a **Random Forest Classifier** to label consumption records as normal or anomalous
- Serves predictions through an interactive **Dash** web dashboard with per-consumer drill-down, anomaly visualization, and human-readable explanations for each flagged anomaly

## 🎯 Key Features

- **Per-consumer anomaly detection** — select any consumer ID and view their full consumption timeline with anomalies highlighted
- **Weather-aware modeling** — incorporates temperature, humidity, and pressure as features, since legitimate usage spikes/drops often track weather
- **Explainable flags** — each anomaly is paired with a plain-language reason (e.g. "Unusually high consumption", "High usage despite cool weather")
- **Interactive visualization** — line chart of usage over time with anomaly markers, plus a details table
- **Simple auth layer** — basic login gate for dashboard access

## 🗂️ Project Structure

```
electricity-theft-project/
├── auth/
│   └── users.txt              # Login credentials (⚠️ move to a secrets manager / hash before deploying)
├── dashboard/
│   └── app.py                 # Dash web app — main interactive dashboard
├── data/
│   ├── Electricity_Theft_Data.csv   # Raw consumer consumption data (consumer x daily readings)
│   ├── temp_user_pivot.csv          # Intermediate per-user pivot used at inference time
│   └── weather_api.py                # OpenWeatherMap API wrapper
├── model/
│   └── model.pkl               # Trained Random Forest model (serialized)
├── utils/
│   ├── preprocess.py           # Data cleaning, feature engineering, scaling
│   └── model_utils.py          # Model training & evaluation helpers
├── main.py                     # CLI entry point: fetch weather → preprocess → train/evaluate
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- **Python 3.13**
- **scikit-learn** — Random Forest classifier, preprocessing (`StandardScaler`)
- **pandas / NumPy** — data wrangling
- **Dash (Plotly)** — interactive web dashboard
- **OpenWeatherMap API** — live weather data
- **joblib / pickle** — model persistence

## ⚙️ How It Works

1. **Data ingestion** — wide-format CSV (`CONS_NO` + one column per date) is loaded and reshaped into a long format per consumer.
2. **Weather enrichment** — current temperature, humidity, and pressure for the target location are pulled from OpenWeatherMap and joined onto the usage data.
3. **Feature engineering & scaling** — usage values are cleaned, transposed so each row is a date/reading, combined with weather features, and standardized with `StandardScaler`.
4. **Labeling & training** — a heuristic threshold on average consumption produces initial labels, and a `RandomForestClassifier` is trained on the resulting feature set.
5. **Dashboard inference** — for a selected consumer, the pipeline re-runs preprocessing + prediction on demand and renders the consumption curve with anomalies marked in red, plus a table explaining each flag.

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Configuration
Create a `.env` file (not included) with your OpenWeatherMap API key rather than hardcoding it:
```
OPENWEATHER_API_KEY=your_key_here
```
> **Note:** The current code has the API key and login credentials hardcoded/in plain text for prototyping. Before deploying or open-sourcing, move the API key to an environment variable (e.g. via `python-dotenv`) and hash/secure the credentials in `auth/users.txt`.

### Run the training pipeline
```bash
python main.py
```

### Launch the dashboard
```bash
python dashboard/app.py
```
Then open the local URL Dash prints (default `http://127.0.0.1:8050`) in your browser.

## 📊 Model

- **Algorithm:** Random Forest Classifier (scikit-learn defaults)
- **Features:** consumption values + temperature, humidity, pressure
- **Output:** binary anomaly label per date/consumer, evaluated via `classification_report`

## 🔮 Future Improvements

- Replace the heuristic mean-threshold labeling with verified theft/fraud labels if available
- Add model comparison (Gradient Boosting, Isolation Forest, LSTM for time-series anomalies)
- Move secrets (API key, credentials) out of source code
- Add historical weather data per date instead of only current weather
- Add proper user authentication (hashed passwords, session management)

## 👤 Author

Nimra


Specify a license (e.g. MIT) before publishing publicly.
