# import packages
import pandas as pd
import cleaning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# fetch data
data = pd.read_csv('data/diabetic_data.csv')
clean_data = cleaning.load_and_clean(data)

# imputing missing values
clean_data['max_glu_serum'] = clean_data['max_glu_serum'].apply(lambda x: 'Unknown' if type(x) != str else x)
clean_data['A1Cresult'] = clean_data['A1Cresult'].apply(lambda x: 'Unknown' if type(x) != str else x)

# frequency encoding
categorical_columns = clean_data.select_dtypes(include=['object', 'category']).columns.tolist()
for cat_column in categorical_columns:
    frequency_encoding = clean_data[cat_column].value_counts(normalize=True).to_dict()
    clean_data[f'encoded_{cat_column}'] = clean_data[cat_column].map(frequency_encoding)
    clean_data = clean_data.drop(cat_column, axis=1)
print(clean_data.head())

# Split into covariates and target
X = clean_data.drop('readmit30', axis = 1)
y = clean_data['readmit30']

# split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=815)

# feature scaling
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# create model and train
logreg_model = LogisticRegression(solver='lbfgs', max_iter=100) # maybe try sag as well
logreg_model.fit(X_train_scale, y_train)

# predict
y_pred = logreg_model.predict(X_test_scale)
# get probabilities for readmission = 1
y_pred_prob = logreg_model.predict_proba(X_test_scale)[:, 1]

# Evaluate model
print("Model Coefficients: ", logreg_model.coef_)
print("Model Intercept: ", logreg_model.intercept_)
print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("\nClassification Report of Model: \n", classification_report(y_test, y_pred))
print("\n Model AUC Score")

# Feature importance based on coefficient magnitude
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logreg_model.coef_[0],
    'Abs_Coefficient': abs(logreg_model.coef_[0])
}).sort_values(by='Abs_Coefficient', ascending=False)

print("\nModel Feature importance:\n", coef_df)
