import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    df = pd.read_csv('LungCapData.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Convert categorical variables to numerical format for the model
    df_model = pd.get_dummies(df, columns=['Smoke', 'Gender', 'Caesarean'], drop_first=True)

except FileNotFoundError:
    print("Error: 'LungCapData.csv' not found. Please ensure the file is in the correct folder.")
    exit()

median_lung_cap = df_model['LungCap'].median()
df_model['High_Lung_Cap'] = (df_model['LungCap'] > median_lung_cap).astype(int)
print(f"Binary target variable 'High_Lung_Cap' created based on the median value of {median_lung_cap:.2f}")

X = df_model.drop(['LungCap', 'High_Lung_Cap'], axis=1)
y = df_model['High_Lung_Cap']

# An 80/20 split.
# 'stratify=y' ensures the train and test sets have a similar proportion of high/low cases.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)
print("Logistic Regression model has been trained.")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Low Lung Cap', 'High Lung Cap'])

print("\n--- Model Evaluation on Unseen Test Data ---")
print(f"Accuracy: {accuracy:.2%}")

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nBreakdown of the Confusion Matrix:")
print(f"True Negatives (TN): {conf_matrix[0, 0]} (Correctly predicted 'Low')")
print(f"False Positives (FP): {conf_matrix[0, 1]} (Incorrectly predicted 'High')")
print(f"False Negatives (FN): {conf_matrix[1, 0]} (Incorrectly predicted 'Low')")
print(f"True Positives (TP): {conf_matrix[1, 1]} (Correctly predicted 'High')")

print("\nFull Classification Report:")
print(class_report)