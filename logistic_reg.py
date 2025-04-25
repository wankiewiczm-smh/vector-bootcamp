# import packages
import pandas as pd
import cleaning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, \
    roc_auc_score, accuracy_score, recall_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import timeit
import matplotlib.pyplot as plt


# fetch data
data = pd.read_csv('data/diabetic_data.csv')
clean_data = cleaning.load_and_clean(data)

# imputing missing values from t
clean_data['max_glu_serum'] = clean_data['max_glu_serum'].\
    apply(lambda x: 'Unknown' if type(x) != str else x)
clean_data['A1Cresult'] = clean_data['A1Cresult'].\
    apply(lambda x: 'Unknown' if type(x) != str else x)

# reduced model 1: using total previous visits
#clean_data = clean_data[['readmit30', 'total_previous_visits', 'diag_1_ccs',
#                         'discharge_disposition_id', 'num_lab_procedures',
#                         'time_in_hospital', 'number_diagnoses', 'age', 'race']]
# reduced model 2: using total_inpatient_visits
clean_data = clean_data[['readmit30', 'number_inpatient', 'diag_1_ccs',
                         'discharge_disposition_id', 'num_lab_procedures',
                         'time_in_hospital', 'number_diagnoses', 'age', 'race']]
# reduced model 3: using all diagnoses and total_previous visits
#clean_data = clean_data[['readmit30', 'total_previous_visits', 'diag_1_ccs',
#                         'diag_2_ccs', 'diag_3_ccs', 'discharge_disposition_id',
#                         'num_lab_procedures', 'time_in_hospital',
#                         'number_diagnoses', 'age', 'race']]

# Split into covariates and target
X = clean_data.drop('readmit30', axis=1)
y = clean_data['readmit30']

# encoding categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, drop_first=True)

# split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,
                                                    test_size=0.2,
                                                    random_state=815)

# feature scaling
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# start time counter
start = timeit.default_timer()

# create model and train
logreg_model = LogisticRegression(C=0.01,
                                  penalty='l2',
                                  solver='newton-cg',
                                  max_iter=1000)
logreg_model.fit(X_train_scale, y_train)

# Calibrate using sigmoid calibration
calibrated_logreg = CalibratedClassifierCV(estimator=logreg_model,
                                           method='sigmoid',
                                           cv='prefit')

calibrated_logreg.fit(X_train_scale, y_train)

# evaluate base model and calibrated model
y_pred_base = logreg_model.predict(X_test_scale)
y_proba_base = logreg_model.predict_proba(X_test_scale)[:, 1]

y_pred_calib = calibrated_logreg.predict(X_test_scale)
y_proba_calib = calibrated_logreg.predict_proba(X_test_scale)[:, 1]

print("Base model Accuracy: ", accuracy_score(y_test, y_pred_base))
print("\nBase model AUC score: ", roc_auc_score(y_test, y_pred_base))
print("\nBase model Recall: ", recall_score(y_test, y_pred_base))
print("\nBase model Confusion Matrix: ", confusion_matrix(y_test, y_pred_base))

print("Calibrated model Accuracy: ", accuracy_score(y_test, y_pred_calib))
print("\nCalibrated model AUC score: ", roc_auc_score(y_test, y_pred_calib))
print("\nCalibrated model Recall: ", recall_score(y_test, y_pred_calib))
print("\nCalibrated model Confusion Matrix: ", confusion_matrix(y_test,
                                                                y_pred_calib))

# Plot calibration curve
plt.figure(figsize=(10, 6))
probas_list = [y_proba_base, y_proba_calib]
names = ['Uncalibrated', 'Sigmoid calibration']
for probas, name in zip(probas_list, names):
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, probas, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)

plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.legend()
plt.title("Calibration curves")
plt.show()
'''
# predict
y_pred = logreg_model.predict(X_test_scale)
# get probabilities for readmission = 1
y_pred_prob = logreg_model.predict_proba(X_test_scale)[:, 1]

# end time counter, and get time taken to train
stop = timeit.default_timer()
print('Time: ', stop - start)

# Evaluate model
print("Model Coefficients: ", logreg_model.coef_)
print("Model Intercept: ", logreg_model.intercept_)
print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("\nClassification Report of Model: \n",
      classification_report(y_test, y_pred))
print("\n Model AUC Score: \n", roc_auc_score(y_test, y_pred))

# Feature importance based on coefficient magnitude
coef_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Coefficient': logreg_model.coef_[0],
    'Abs_Coefficient': abs(logreg_model.coef_[0])
}).sort_values(by='Abs_Coefficient', ascending=False)

print("\nModel Feature importance:\n", coef_df)
'''
