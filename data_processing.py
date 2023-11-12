# data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

def load_data(file_path):
    return pd.read_parquet(file_path)

def preprocess_data(transactions_df):
    transactions_df['Amount'] = transactions_df['Amount'].replace('[\$,]', '', regex=True).astype(float)
    
    # Identify categorical and numerical columns
    categorical_cols = transactions_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = transactions_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols.remove('Is Fraud?')  # Remove the target variable

    # Imputers for numerical and categorical data
    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Create transformers for categorical and numerical data
    numerical_transformer = Pipeline(steps=[('imputer', numerical_imputer), ('scaler', StandardScaler()), ('pca', PCA(n_components=0.95))])
    categorical_transformer = Pipeline(steps=[('imputer', categorical_imputer), ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Apply preprocessing
    X = preprocessor.fit_transform(transactions_df.drop('Is Fraud?', axis=1))
    y = transactions_df['Is Fraud?'].map({'Yes': 1, 'No': 0})

    return X, y

def get_processed_data(file_name):
    file_path = f'data/{file_name}'
    raw_data = load_data(file_path)
    X, y = preprocess_data(raw_data)

    # Handling class imbalance with undersampling
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    return X_resampled, y_resampled
