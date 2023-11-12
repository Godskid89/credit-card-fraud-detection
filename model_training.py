# model_training.py

from data_processing import get_processed_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    X, y = get_processed_data('credit_card_transaction_data_de.parquet')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Save the model
    joblib.dump(model, 'model/credit_card_fraud_detection_model.pkl')

if __name__ == '__main__':
    main()
