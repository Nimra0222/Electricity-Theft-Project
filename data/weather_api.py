import requests

def fetch_weather_data(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    main = data.get('main', {})
    return {
        'temp': main.get('temp', 0),
        'humidity': main.get('humidity', 0),
        'pressure': main.get('pressure', 0),
    }
