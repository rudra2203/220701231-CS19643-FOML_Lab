import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
try:
    df = pd.read_csv("Crop_recommendation.csv")
    print("Columns in dataset:", df.columns.tolist())  # Show all columns
    
    # Verify required columns exist
    required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']

    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing columns: {missing}")
    
    # Proceed with modeling
    X = df.drop('label', axis=1)
    y = df['label']

    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    joblib.dump(model, 'model.pkl')


except Exception as e:
    print(f"Error: {str(e)}")
    print("Please check your dataset file and its contents.")