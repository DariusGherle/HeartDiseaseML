import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load training data
train_data = pd.read_csv('heart_disease_uci.csv')

'''
# Vizualizare rapidă
print(train_data.head(2))
print(train_data.dtypes)
print(train_data.columns)
print(train_data.shape)
print(train_data.isnull().sum())
print(train_data['dataset'].value_counts())
'''

# Drop coloane irelevante sau problematice
train_data.drop(['id', 'ca', 'thal'], axis=1, inplace=True)

#Inlocuire valori lipsa
train_data['fbs'] = train_data['fbs'].fillna(train_data['fbs'].mode().iloc[0])

# Mapare categorii text în valori numerice
train_data['sex'] = train_data['sex'].map({'Male': 0, 'Female': 1})
train_data['dataset'] = train_data['dataset'].map({'Cleveland': 0, 'Hungary': 1, 'Switzerland': 2, 'VA Long Beach': 3})
train_data['cp'] = train_data['cp'].map({'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3})
train_data['fbs'] = train_data['fbs'].map({'True': 1, 'False': 0})
train_data['fbs'] = train_data['fbs'].fillna(0)
train_data['restecg'] = train_data['restecg'].map({'normal': 0, 'stt abnormality': 1, 'lv hypertrophy': 2})
train_data['exang'] = train_data['exang'].map({'TRUE': 1, 'FALSE': 0})
train_data['slope'] = train_data['slope'].map({'flat': 0, 'upsloping': 1, 'downsloping': 2})

# Înlocuire valori lipsă cu mediană sau modă
train_data['trestbps'] = train_data['trestbps'].fillna(train_data['trestbps'].median())
train_data['chol'] = train_data['chol'].fillna(train_data['chol'].median())
train_data['restecg'] = train_data['restecg'].fillna(train_data['restecg'].mode().iloc[0])
train_data['thalch'] = train_data['thalch'].fillna(train_data['thalch'].median())
if not train_data['exang'].mode().empty:
    train_data['exang'] = train_data['exang'].fillna(train_data['exang'].mode().iloc[0])
else:
    train_data['exang'] = train_data['exang'].fillna(0)  # fallback dacă nu există modă
train_data['oldpeak'] = train_data['oldpeak'].fillna(train_data['oldpeak'].median())
train_data['slope'] = train_data['slope'].fillna(train_data['slope'].mode().iloc[0])

# Mapare target: 0 = fără boală, 1 = orice formă de boală
train_data['num'] = train_data['num'].apply(lambda x: 0 if x == 0 else 1)

# Features și target
features = ['age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalch', 'exang', 'oldpeak', 'slope']
X = train_data[features]
print(X.isnull().sum())
y = train_data['num']

# Împărțire în seturi de antrenare și validare
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Standardizare
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(np.isnan(X_train_scaled).sum(), "NaN în X_train_scaled")
print(np.isnan(X_val_scaled).sum(), "NaN în X_val_scaled")

#verificam daca talch are valori
print(train_data['thalch'].isnull().sum())

# Model Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=1)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, predictions)

print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Model Logistic Regression
log_model = LogisticRegression(C=1.0, random_state=1)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_val_scaled)

log_accuracy = accuracy_score(y_val, log_preds)
print(f"Logistic regression accuracy: {log_accuracy * 100:.2f}%")

#train model SVM
svm_model = SVC(C=1.0, kernel='linear', gamma='scale')
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_val_scaled)

svm_accuracy = accuracy_score(y_val, svm_preds)
print(f"SVM accuracy: {svm_accuracy *100:.2f}%")

#Train model K-nearest neighbours (KNN)
knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_val_scaled)

knn_accuracy = accuracy_score(y_val, knn_preds)
print(f"KNN accuracy: {knn_accuracy * 100:.2f}%")
