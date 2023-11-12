# Credit Card Fraud Detection Project

## Overview
This project focuses on building a predictive model to identify fraudulent credit card transactions. Using historical transaction data, the model aims to assist financial institutions in mitigating the risks associated with credit card fraud.

## Data
The project utilizes three main datasets:
- `credit_card_transaction_data_de.parquet`: Transaction details including amount, merchant information, and whether the transaction was flagged as fraudulent.
- `credit_card_users_de.parquet`: User demographics and financial information.
- `sd254_cards_de.parquet`: Credit card details for each user.

## Methodology
1. **Data Preprocessing**: Includes handling missing values, encoding categorical variables, and normalizing numerical features.
2. **Exploratory Data Analysis (EDA)**: Analysis of transaction patterns, user behavior, and identification of key features influencing fraudulent activities.
3. **Feature Engineering**: Creation of new features to improve model performance.
4. **Model Development**: Implementation of a RandomForestClassifier with strategies to address class imbalance in the data.
5. **Model Evaluation**: Evaluation of the model using metrics like accuracy, precision, recall, F1-score, and a confusion matrix. Analysis of feature importance to understand key predictors of fraud.

## Files in the Repository
- `data_processing.py`: Script for data loading, cleaning, and preprocessing.
- `eda.ipynb`: Jupyter notebook containing the exploratory data analysis and modeling.
- `model_training.py`: Script for model training, including handling imbalanced data and hyperparameter tuning.
- `model_evaluation.py`: Script for evaluating the model and analyzing feature importance.
- `requirements.txt`: List of Python libraries required to run the code.

## How to Run the Project
1. Install the required packages: `pip install -r requirements.txt`
2. Run the data processing script: `python data_processing.py`
3. Explore the EDA notebook: `jupyter notebook eda.ipynb`
4. Train the model: `python model_training.py`
5. Evaluate the model: `python model_evaluation.py`

## Results
The model achieved an accuracy of [insert accuracy], with a recall of [insert recall] for fraudulent transactions. Key predictors of fraud include [list key features].

## Future Work
- Explore alternative models and resampling techniques.
- Deepen the analysis of transaction patterns over time.
- Implement real-time fraud detection mechanisms.

## Contact
For any further questions or contributions, please contact [Your Name] at [Your Email].

---

Remember to replace placeholders (like `[insert accuracy]` or `[insert recall]`) with actual values from your project. You may also want to include additional sections or details specific to your project, such as challenges faced, insights gained, or specific conclusions drawn from your analysis.