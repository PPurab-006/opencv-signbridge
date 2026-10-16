import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

def train():
    print("Loading data...")
    df = pd.read_csv("data/dataset.csv")
    
    # Check if df is empty
    if len(df) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    X = df.drop("Label", axis=1)
    y = df["Label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Accuracy: {acc:.4f}")
    
    # Save metrics
    metrics = {
        "accuracy": acc,
        "classification_report": report
    }
    with open("data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save model
    joblib.dump(clf, "data/model.pkl")
    print("Model saved to data/model.pkl")

if __name__ == "__main__":
    train()
