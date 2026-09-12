import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(X, y, model_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Your model training code (replace with your actual model training)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
