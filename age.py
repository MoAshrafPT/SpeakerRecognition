import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("data/features.csv")

# Map original labels to age group
# 0 -> M20, 1 -> F20, 2 -> M50, 3 -> F50
# Age: 0 = Young (20s), 1 = Older (50s)
df["age"] = df["label"].map({0: 0, 1: 0, 2: 1, 3: 1})

# Define age-relevant features
features = [
    'pitch_mean', 'pitch_std',
    'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4', 'mfcc_mean_5',
    'mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3', 'mfcc_std_4', 'mfcc_std_5', 'mfcc_std_6',
    'mfcc_std_7', 'mfcc_std_8', 'mfcc_std_9', 'mfcc_std_10', 'mfcc_std_11', 'mfcc_std_12',
    'centroid_mean', 'centroid_std',
    'bandwidth_mean', 'bandwidth_std',
    'rolloff_mean', 'rolloff_std',
    'rms_mean', 'rms_std'
]

# Add chroma std
features += [f"chroma_std_{i}" for i in range(1, 13)]

# Add mel mean and std (first 30)
features += [f"mel_mean_{i}" for i in range(1, 31)]
features += [f"mel_std_{i}" for i in range(1, 31)]

# Add formant features
formant_features = [f'formant_{i}' for i in range(1, 5)]
for feature in formant_features:
    if feature in df.columns:
        features.append(feature)
        print(f"Including {feature} in model training")


# Extract features and target
X = df[features]
y = df["age"]

# Balance age groups with undersampling
rus = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = rus.fit_resample(X, y)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize GPU-accelerated XGBoost
xgb_model = XGBClassifier(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=200,
    reg_alpha=0.01,
)

lightgbm_model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='binary_logloss',
    n_estimators=200,
    random_state=42,
    reg_alpha=0.01,
)

# Initialize KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

#logistic regression model
logistic_model = LogisticRegression(max_iter=1000)

# Stacking classifier
stacking_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lightgbm_model),
        ('knn', knn_model)
    ],
    final_estimator=logistic_model,
    cv=5,
    n_jobs=-1
)

# Train model
stacking_model.fit(X_train_scaled, y_train)

# Predict
y_pred = stacking_model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Age Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler
joblib.dump(stacking_model, 'age_model.joblib')
joblib.dump(scaler, 'age_scaler.joblib')
