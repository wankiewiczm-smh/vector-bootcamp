# GAM for predicting readmission outcome in admitted diabetic patients
import pandas as pd
from imblearn.over_sampling import SMOTE
import math
import cleaning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pygam import LogisticGAM, s, f
from sklearn.metrics import recall_score, classification_report, \
    roc_auc_score

# Just using vector's code

def preprocess_data(data):
    # Make copies to preserve original numerical values
    data_processed = data.copy()
    data_numerical_orig = data_processed.copy()

    # Identify categorical and numerical columns.
    categorical_columns = data.select_dtypes(
        include=['object']).columns.tolist()
    numerical_columns = data.select_dtypes(
        include=['int64', 'float64']).columns.tolist()

    # Remove the target variable 'readmit30' from numerical_columns
    if 'readmit30' in numerical_columns:
        numerical_columns.remove('readmit30')

    # For categorical features, use LabelEncoder and store mapping.
    cat_mapping = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])
        # Save the original class labels (in order of encoding)
        cat_mapping[col] = list(le.classes_)

    # For numerical features, store original values.
    data_numerical_orig = data_numerical_orig[numerical_columns]

    # Scale numerical features (excluding the target)
    scaler = StandardScaler()
    data_processed[numerical_columns] = scaler.fit_transform(
        data_processed[numerical_columns])

    return data_processed, scaler, cat_mapping, data_numerical_orig, \
        numerical_columns


np.random.seed(185)

# fetch data
data = pd.read_csv('data/diabetic_data.csv')
clean_data = cleaning.load_and_clean(data)


df_processed, scaler, cat_mapping, data_numerical_orig, numerical_columns = \
    preprocess_data(clean_data)
print("Processed data head:")
print(df_processed.head())

#df_processed = df_processed[['readmit30', 'number_inpatient', 'diag_1_ccs',
#                             'discharge_disposition_id',
#                             'num_medications',
#                             'num_lab_procedures', 'time_in_hospital',
#                             'number_diagnoses', 'age', 'race']]
df_processed = df_processed.drop(['diag_1', 'diag_2', 'diag_3', 'diabetesMed',
                                  'total_previous_visits', 'diag_1_clean',
                                  'diag_2_clean', 'diag_3_clean',
                                  'metformin-pioglitazone',
                                  'metformin-rosiglitazone',
                                  'glimepiride-pioglitazone',
                                  'glipizide-metformin', 'citoglipton',
                                  'examide', 'troglitazone'], axis=1)
print("Processed data Columns: \n")
print(df_processed.columns.tolist())

# Split into covariates and target
X = df_processed.drop('readmit30', axis=1)
y = df_processed['readmit30']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# oversample minority class (readmitted = T)
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)

# Build model specification: smooth for numerical, factor for categorical.
terms = None
for i, col in enumerate(X_train.columns):
    if col in numerical_columns:
        new_term = s(i)
    else:
        new_term = f(i)
    terms = new_term if terms is None else terms + new_term

gam = LogisticGAM(terms, n_splines=8).gridsearch(X_train_resampled.values,
                                                 y_train_resampled.values)

#gam = LogisticGAM(terms, n_splines=8).fit(X_train_resampled.values,
#                                          y_train_resampled.values)

print(gam.summary())
print(gam.accuracy(X_test, y_test))

y_pred = gam.predict(X_test)
mse = np.mean((y_test - y_pred)**2)
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print("\n Model AUC Score: ", roc_auc_score(y_test, y_pred))

#
y_pred_proba = gam.predict_proba(X_test)
y_pred_binary = (y_pred_proba >= 0.5).astype(int)

# calculate recall
recall = recall_score(y_test, y_pred_binary)
print(f"Recall: {recall:.4f}")

print("\n Classification Report: ")
print(classification_report(y_test, y_pred_binary))

feature_names = X_train.columns
n_features = len(feature_names)

# Create a mapping for numerical features: feature -> (mean, std)
num_stats = {}
for col in numerical_columns:
    j = numerical_columns.index(col)
    mean = scaler.mean_[j]
    std = np.sqrt(scaler.var_[j])
    num_stats[col] = (mean, std)

# Baseline row: use the mean of the scaled X_train for all features.
baseline_row = X_train.values.mean(axis=0)

# Determine subplot grid dimensions
ncols = math.ceil(np.sqrt(n_features))
nrows = math.ceil(n_features / ncols)

fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axs = axs.ravel()

# For each feature, create a full X matrix based on baseline values.
for idx in range(n_features):
    feature = feature_names[idx]
    term_idx = int(idx)

    if feature in numerical_columns:
        grid_points = 100
        # Create baseline matrix with grid_points rows.
        X_baseline = np.tile(baseline_row, (grid_points, 1))
        # Get original range for this numerical feature
        orig_vals = data_numerical_orig[feature]
        orig_min, orig_max = orig_vals.min(), orig_vals.max()
        grid_orig = np.linspace(orig_min, orig_max, grid_points)
        # Convert grid to scaled values using (mean, std)
        mean, std = num_stats[feature]
        grid_scaled = (grid_orig - mean) / std
        X_baseline[:, term_idx] = grid_scaled
        pdep, confi = gam.partial_dependence(term=term_idx, X=X_baseline,
                                             width=0.95)
        axs[idx].plot(grid_orig, pdep)
        axs[idx].fill_between(grid_orig, confi[:, 0], confi[:, 1], color='r',
                              alpha=0.3)
        axs[idx].set_xlabel(feature)
    else:
        # For categorical features
        levels = cat_mapping[feature]
        grid_points = len(levels)
        X_baseline = np.tile(baseline_row, (grid_points, 1))
        grid_factor = np.arange(grid_points)
        X_baseline[:, term_idx] = grid_factor
        pdep, confi = gam.partial_dependence(term=term_idx, X=X_baseline,
                                             width=0.95)
        axs[idx].plot(grid_factor, pdep, marker='o')
        axs[idx].fill_between(grid_factor, confi[:, 0], confi[:, 1], color='r',
                              alpha=0.3)
        axs[idx].set_xticks(grid_factor)
        axs[idx].set_xticklabels(levels)
        axs[idx].set_xlabel(feature)

    axs[idx].set_ylabel("Target")
    axs[idx].set_title(f"Shape Function of {feature}")

# Hide any extra subplots if n_features is not a perfect grid.
for k in range(n_features, len(axs)):
    fig.delaxes(axs[k])

#plt.tight_layout()
plt.savefig('GAM_regression.png')
plt.show()
