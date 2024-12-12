import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Set up Faker and logging
fake = Faker()
logging.basicConfig(level=logging.INFO)

# Enhanced fake data generator with BlueGuard HealthPay-themed placeholder values
def generate_fake_user_data():
    try:
        user_data = {
            'patient_id': fake.uuid4(),
            'patient_name': fake.name(),
            'email': fake.email(),
            'phone_number': fake.phone_number(),
            'address': fake.address(),
            'birthdate': fake.date_of_birth(minimum_age=18, maximum_age=85).strftime("%Y-%m-%d"),
            'appointment_date': fake.date_time_this_year().strftime("%Y-%m-%d %H:%M:%S"),
            'transaction_amount': round(fake.random_number(digits=5, fix_len=True) / 100, 2),  # Amount in dollars
            'healthcare_provider': fake.company(),
            'medical_condition': fake.random_element(elements=('Diabetes', 'Hypertension', 'Asthma', 'Heart Disease')),
            'treatment': fake.random_element(elements=('Medication', 'Surgery', 'Physical Therapy', 'Consultation')),
            'insurance_plan': fake.random_element(elements=('Gold', 'Silver', 'Bronze')),
            'payment_method': fake.random_element(elements=('Credit Card', 'Debit Card', 'Bank Transfer', 'Insurance')),
            'transaction_currency': fake.currency_code(),
            'Class': fake.random_element(elements=('Fraud', 'Valid'))  # 'Fraud' or 'Valid'
        }
        return user_data
    except Exception as e:
        logging.error(f"Error in generate_fake_user_data function: {e}")
        raise

# Generate a sample dataset
sample_data = [generate_fake_user_data() for _ in range(1000)]
data = pd.DataFrame(sample_data)

# Show a sample of the generated data
print("Sample Data:")
print(data.head())

# Data Cleaning: Handle missing values and outliers
data['transaction_amount'].fillna(data['transaction_amount'].median(), inplace=False)
data = data[(np.abs(data['transaction_amount'] - data['transaction_amount'].mean()) <= (3 * data['transaction_amount'].std()))]

# Feature Engineering: Create new features based on existing ones
data['appointment_date'] = pd.to_datetime(data['appointment_date'])
data['appointment_timestamp'] = data['appointment_date'].astype('int64') // 10**9
data['appointment_year'] = data['appointment_date'].dt.year
data['appointment_month'] = data['appointment_date'].dt.month
data['appointment_day'] = data['appointment_date'].dt.day
data = pd.get_dummies(data, columns=['transaction_currency', 'healthcare_provider', 'medical_condition', 'treatment', 'insurance_plan', 'payment_method'])

# Convert Class to binary format
data['Class'] = data['Class'].map({'Fraud': 1, 'Valid': 0})

# Select features and target variable
features = data.drop(columns=['patient_id', 'patient_name', 'email', 'phone_number', 'address', 'birthdate', 'Class', 'appointment_date'])
target = data['Class']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# Model Training with Different Algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "conf_matrix": conf_matrix
    }

# Evaluate and compare the models
for model_name, result in results.items():
    logging.info(f"\nModel: {model_name}")
    logging.info(f"Accuracy: {result['accuracy']:.2f}")
    logging.info(f"Precision: {result['precision']:.2f}")
    logging.info(f"Recall: {result['recall']:.2f}")
    logging.info(f"F1 Score: {result['f1']:.2f}")
    logging.info(f"ROC AUC Score: {result['roc_auc']:.2f}")
    logging.info(f"Confusion Matrix:\n{result['conf_matrix']}")

# Plot ROC Curve
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {results[model_name]['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Display results summary
for model_name, result in results.items():
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print(f"Precision: {result['precision']:.2f}")
    print(f"Recall: {result['recall']:.2f}")
    print(f"F1 Score: {result['f1']:.2f}")
    print(f"ROC AUC Score: {result['roc_auc']:.2f}")
    print(f"Confusion Matrix:\n{result['conf_matrix']}")
