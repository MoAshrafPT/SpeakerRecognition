import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import joblib


# Load the dataset
df = pd.read_csv("data/features.csv")

# Map original labels to gender
# 0 -> M20, 1 -> F20, 2 -> M50, 3 -> F50
# Gender: 0 = Male, 1 = Female
df["gender"] = df["label"].map({0: 0, 1: 1, 2: 0, 3: 1})

# Define selected features (93 total)
features = []

# Pitch
features += ['pitch_mean', 'pitch_std']

# MFCCs
features += [f"mfcc_mean_{i}" for i in range(1, 14)]
features += [f"mfcc_std_{i}" for i in range(1, 14)]

# Chroma
features += [f"chroma_mean_{i}" for i in range(1, 13)]
features += [f"chroma_std_{i}" for i in range(1, 13)]

# Spectral shape
features += ['centroid_mean', 'centroid_std', 'bandwidth_mean', 'bandwidth_std']
features += ['rolloff_mean', 'rolloff_std']

# Energy / voicing
features += ['rms_mean', 'rms_std', 'zcr_mean', 'zcr_std']

# Mel (only first 13)
features += [f"mel_mean_{i}" for i in range(1, 14)]
features += [f"mel_std_{i}" for i in range(1, 14)]

# Extract features and target
X = df[features]
y = df["gender"]

# Undersample to balance genders
rus = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = rus.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize GPU-accelerated XGBoost classifier
xgb_model = XGBClassifier(
    tree_method='gpu_hist',  # Enables GPU training
    predictor='gpu_predictor',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=200,
    
)

# Train the model
xgb_model.fit(X_train_scaled, y_train)

# Predict
y_pred = xgb_model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#save model
joblib.dump(xgb_model, 'gender_model.joblib')
#save scaler
joblib.dump(scaler, 'gender_scaler.joblib') 