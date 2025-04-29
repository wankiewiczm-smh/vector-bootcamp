import torch
import pandas as pd
import cleaning
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from model.nbm_model import ConceptNBMNary
from utils.plot_shapefunc import plot_nbm_shape_functions_with_feature_density
import utils

# set seeds for reproducibility
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load and preprocess the data
data = pd.read_csv('data/diabetic_data.csv')
clean_data = cleaning.load_and_clean(data)

# imputing missing values
clean_data['max_glu_serum'] = clean_data['max_glu_serum'].\
    apply(lambda x: 'Unknown' if type(x) != str else x)
clean_data['A1Cresult'] = clean_data['A1Cresult'].\
    apply(lambda x: 'Unknown' if type(x) != str else x)
u
# select variables of interest
clean_data = clean_data[['readmit30', 'number_inpatient', 'diag_1_ccs',
                         'discharge_disposition_id', 'num_lab_procedures',
                         'max_glu_serum', 'A1Cresult',
                         'time_in_hospital', 'number_diagnoses', 'age', 'race']]

# Split into covariates and target
X = clean_data.drop('readmit30', axis=1)
y = clean_data['readmit30']


categorical_columns = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, drop_first=True)

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                    random_state=815)

X_train, y_train, = SMOTE(random_state=815).fit_resample(X_train, y_train)

# Convert data to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoader objects for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ------------------------------
# Set up the model, loss, and optimizer
# ------------------------------
num_concepts = X_train_tensor.shape[1]  # num features
num_classes = 1
num_bases = 100
hidden_dims = (256, 128, 128)
num_subnets = 1
dropout = 0.0
bases_dropout = 0.2
batchnorm = True

# Instantiate the model (nary is left as None so that it uses all unary interactions)
model = ConceptNBMNary(
    num_concepts=num_concepts,
    num_classes=num_classes,
    nary=None,
    num_bases=num_bases,
    hidden_dims=hidden_dims,
    num_subnets=num_subnets,
    dropout=dropout,
    bases_dropout=bases_dropout,
    batchnorm=batchnorm
)

# Use Mean Squared Error for regression
# but try NLLLoss after
criterion = nn.BCEWithLogitsLoss() #nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------
# Training loop
# ------------------------------
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        # In training mode, the model returns a tuple: (output, features)
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    # ------------------------------
    # Evaluation on the test set
    # ------------------------------
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for inputs, targets in test_loader:
            # In eval mode, the model returns only the output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
        test_loss = total_loss / len(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")
    # RMSE = sqrt(test_loss)
    print(f"RMSE: {test_loss ** 0.5:.4f}")

'''
# Plot the shape functions of the model along with feature density

device = "cpu"
model.eval()
model.to(device)

feature_names = X_encoded.columns.tolist()

plot_nbm_shape_functions_with_feature_density(
    model,
    X_test,
    feature_names=feature_names,
    n_points=50,   # more points for a smoother curve
    bins=50,        # more histogram bins
    device=device,
    plot_cols=4,
    red_alpha=0.4
)
'''

# Put model in evaluation mode
model.eval()

# Get predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)

        # If outputs is tuple, unpack it
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Convert probabilities to binary predictions (threshold = 0.5)
        preds = (outputs > 0.5).int().squeeze()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy().astype(int))

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Compute performance metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

# Print them nicely
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
