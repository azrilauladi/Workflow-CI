"""
modelling.py
Exam Score Regression menggunakan MLflow dengan Autolog (Basic)

Author: Achmad Azril
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def main():
    # Load preprocessed data
    train_data = pd.read_csv('ExamScore_train.csv')
    test_data = pd.read_csv('ExamScore_test.csv')
    
    # Split features dan target
    X_train = train_data.drop('Performance Index', axis=1)
    y_train = train_data['Performance Index']
    X_test = test_data.drop('Performance Index', axis=1)
    y_test = test_data['Performance Index']
    
    # Set experiment name
    mlflow.set_experiment("Exam_Score_Regression")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForestRegressor_Autolog"):
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Print hasil
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        print(f"\n[INFO] Model logged dengan MLflow Autolog")
        print(f"[INFO] Run ID: {mlflow.active_run().info.run_id}")

        # Simpan model dengan joblib
        joblib.dump(model, "model.joblib")
        print("[INFO] Model saved as model.joblib")

if __name__ == "__main__":
    main()