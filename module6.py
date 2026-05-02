# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("/Users/hadisunny/Desktop/school/senior/414/module 6/student-mat.csv", sep=";")

# Create pass/fail label
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Drop G3 (target leakage)
df = df.drop("G3", axis=1)

# Encode categorical variables
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop("pass", axis=1)
y = df["pass"]

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")  # saves image
plt.show()

importance = pd.Series(model.coef_[0], index=X.columns)
importance = importance.sort_values()

importance.tail(10).plot(kind="barh")
plt.title("Top Features Predicting Student Success")
plt.show()