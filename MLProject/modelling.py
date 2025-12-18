"""
modelling.py
Wine Quality Classification menggunakan MLflow dengan Autolog (Basic)

Author: Achmad Azril
"""
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load preprocessed data
    train_data = pd.read_csv('WineQT_train.csv')
    test_data = pd.read_csv('WineQT_test.csv')
    
    # Split features dan target
    X_train = train_data.drop('quality', axis=1)
    y_train = train_data['quality']
    X_test = test_data.drop('quality', axis=1)
    y_test = test_data['quality']
    
    # Set experiment name
    mlflow.set_experiment("Wine_Quality_Classification")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Print hasil
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n[INFO] Model logged dengan MLflow Autolog")
        print(f"[INFO] Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='WineQT_train.csv')
    parser.add_argument('--test_data', type=str, default='WineQT_test.csv')
    args = parser.parse_args()
    main(args.train_data, args.test_data)