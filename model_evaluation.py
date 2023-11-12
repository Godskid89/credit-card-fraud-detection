import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_processing import get_processed_data
from sklearn.model_selection import train_test_split

def load_model(model_path):
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    model = load_model('model/credit_card_fraud_detection_model.pkl')
    X, y = get_processed_data('credit_card_transaction_data_de.parquet')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
