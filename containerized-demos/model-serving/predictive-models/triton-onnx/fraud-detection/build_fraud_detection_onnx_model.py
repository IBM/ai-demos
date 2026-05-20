import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from pathlib import Path

# Download dataset from the kaggle hub
path = kagglehub.dataset_download("dhanushnarayananr/credit-card-fraud")

csv_file_path = os.path.join(path, "card_transdata.csv")
df = pd.read_csv(csv_file_path)
print(f"Dataset loaded successfully from: {csv_file_path}")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nClass distribution (0: Not Fraud, 1: Fraud):")
print(df['fraud'].value_counts())
print(f"Fraudulent transactions make up: {df['fraud'].value_counts()[1]/df['fraud'].value_counts()[0]*100:.4f}% of non-fraudulent transactions.")


X = df.drop('fraud', axis=1)
y = df['fraud']

# Stratify by 'y' to ensure both training and testing sets have a similar proportion of fraud cases.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


print(f"\nTraining set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"Class distribution in y_train: {Counter(y_train)}")
print(f"Class distribution in y_test: {Counter(y_test)}")

print('Training model using RandomForestClassifier')
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of being the positive class (fraud)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# This helps understand which features are most influential in predicting fraud.
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

print("\nTop 10 Feature Importances:")
print(sorted_importances.head(10))

initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
options = {id(model): {'zipmap': False}}
model_onnx = convert_sklearn(model, initial_types=initial_type, options=options)

# Create model directory structure
model_dir = Path('model-repository/fraud-detection/1')
model_dir.mkdir(parents=True, exist_ok=True)

# Save model directly to the target directory
model_path = model_dir / 'model.onnx'
with open(model_path, 'wb') as f:
    f.write(model_onnx.SerializeToString())

print(f"\nModel saved successfully to: {model_path}")
