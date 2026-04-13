#Fraud detection in transaction::

#importing the libraires:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#data collection & preprocessing::
df = pd.read_csv('bank_transactions_data.csv')
print(df)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
df=df.dropna()

#conversion of date column::
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], dayfirst=True)
print(df.info())

#FEATURE ENGINEERING:
#Transaction Hour
df['Hour'] = df['TransactionDate'].dt.hour

#Amount Range
df['Amount_Range'] = pd.cut(df['TransactionAmount'],
                           bins=[0, 10, 50, 100],
                           labels=['Low', 'Medium', 'High'])

#create flag
df['FraudFlag'] = (
    (df['TransactionAmount'] > 400) |
    (df['Hour'] < 5) |
    (df['LoginAttempts'] > 3)
).astype(int)

df['High_Amount'] = (df['TransactionAmount'] > 500).astype(int)
print(df.info())

print("Fraud count:")
print(df['FraudFlag'].value_counts())

#EXPLORATORY DATA ANALYSIS:
fraud_counts = df['FraudFlag'].value_counts()
print("Fraud vs Normal Count:", fraud_counts)

plt.figure()
fraud_counts.plot(kind='bar')
plt.title("Fraud vs Normal Transactions")
plt.xlabel("FraudFlag")
plt.ylabel("Count")
plt.show()

#amount distribution:
sns.boxplot(x='FraudFlag', y='TransactionAmount', data=df)
plt.title("Transaction Amount by Fraud Flag")       
plt.show()

#average amount:
avg_amount=df.groupby('FraudFlag')['TransactionAmount'].mean()
print("Average Transaction Amount by Fraud Flag:",avg_amount)


#simple risk analysis::
risk_data = df.groupby('FraudFlag')['TransactionAmount'].agg(['mean', 'max', 'min'])
print("Risk Analysis Data:", risk_data)

#MACHINE LEARNING MODEL:
#LOGISTIC REGRESSION::

X = df[['TransactionAmount', 'Hour', 'High_Amount']]
y = df['FraudFlag']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#scaling
scaler = StandardScaler()
x_train =scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)   

#logistic regression model
model = LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)

#predictions
y_pred = model.predict(x_test)

#evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:", conf_matrix)
print("Classification Report:",classification_report(y_test, y_pred,zero_division=0))

#visualization ::
#confusion matrix:
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#correlation heatmap:
plt.figure()
sns.heatmap(df[['TransactionAmount', 'Hour', 'High_Amount', 'FraudFlag']].corr(), annot=True, cmap='pink')
plt.title("Correlation Heatmap")
plt.show()


df.to_csv('updated_bank_transactions_data.csv', index=False )


