import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")
print("=== Gender-Aware Age Classification Training ===")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("data/features.csv")

# Map original labels to age and gender
# 0 -> M20, 1 -> F20, 2 -> M50, 3 -> F50
# Age: 0 = Young (20s), 1 = Older (50s)
df["age"] = df["label"].map({0: 0, 1: 0, 2: 1, 3: 1})
# Gender: 0 = Male, 1 = Female
df["gender"] = df["label"].map({0: 0, 1: 1, 2: 0, 3: 1})

# Define common age-relevant features
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

# Define male-specific features (could be expanded with male voice-specific features)
male_features = features.copy()

# Define female-specific features (could be expanded with female voice-specific features)
female_features = features.copy()

# Split data by gender
male_df = df[df['gender'] == 0]
female_df = df[df['gender'] == 1]

print(f"\nData distribution:")
print(f"Total samples: {len(df)}")
print(f"Male samples: {len(male_df)}")
print(f"Female samples: {len(female_df)}")
print(f"Young (20s) samples: {len(df[df['age'] == 0])}")
print(f"Older (50s) samples: {len(df[df['age'] == 1])}")
print(f"Young male samples: {len(male_df[male_df['age'] == 0])}")
print(f"Older male samples: {len(male_df[male_df['age'] == 1])}")
print(f"Young female samples: {len(female_df[female_df['age'] == 0])}")
print(f"Older female samples: {len(female_df[female_df['age'] == 1])}")

# Function to create and train a model
def create_model(X_train, y_train, X_test, y_test, model_name="age"):
    # Initialize GPU-accelerated XGBoost
    xgb_model = XGBClassifier(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.1,
    )
    
    lightgbm_model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        metric='binary_logloss',
        n_estimators=200,
        random_state=42,
        reg_alpha=0.01,
        reg_lambda=0.01,
    )
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
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
    print(f"\nTraining {model_name} model...")
    stacking_model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = stacking_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{model_name.title()} Model Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # If XGBoost has feature importance, plot it
    if hasattr(xgb_model, 'feature_importances_'):
        xgb_model.fit(X_train, y_train)
        importances = xgb_model.feature_importances_
        
        # Get feature names
        feature_names = []
        if model_name == "male age":
            feature_names = male_features
        elif model_name == "female age":
            feature_names = female_features
        else:
            feature_names = features
            
        # Create a DataFrame for better visualization
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 6))
        plt.title(f"Top 20 Feature Importances for {model_name.title()} Model")
        plt.bar(range(20), feature_importance['Importance'][:20])
        plt.xticks(range(20), feature_importance['Feature'][:20], rotation=90)
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f"plots/{model_name.replace(' ', '_')}_feature_importance.png")
        print(f"Feature importance plot saved to plots/{model_name.replace(' ', '_')}_feature_importance.png")
    
    return stacking_model, accuracy, f1

# Function to preprocess data and train a gender-specific model
def train_gender_specific_model(gender_df, features_list, gender_name):
    X = gender_df[features_list]
    y = gender_df["age"]
    
    # Balance age groups with undersampling
    rus = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = rus.fit_resample(X, y)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model, accuracy, f1 = create_model(
        X_train_scaled, y_train, X_test_scaled, y_test, f"{gender_name} age"
    )
    
    # Save model and scaler
    joblib.dump(model, f'age_model_{gender_name.lower()}.joblib')
    joblib.dump(scaler, f'age_scaler_{gender_name.lower()}.joblib')
    print(f"Saved {gender_name}-specific age model and scaler")
    
    return model, scaler, accuracy, f1

# Train male-specific model
print("\n=== Training Male-Specific Age Model ===")
male_model, male_scaler, male_acc, male_f1 = train_gender_specific_model(
    male_df, male_features, "male"
)

# Train female-specific model
print("\n=== Training Female-Specific Age Model ===")
female_model, female_scaler, female_acc, female_f1 = train_gender_specific_model(
    female_df, female_features, "female"
)

# Train combined model
print("\n=== Training Combined Age Model ===")
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

# Create and train combined model
combined_model, combined_acc, combined_f1 = create_model(
    X_train_scaled, y_train, X_test_scaled, y_test, "combined age"
)

# Save combined model
joblib.dump(combined_model, 'age_model_combined.joblib')
joblib.dump(scaler, 'age_scaler_combined.joblib')
print("Saved combined age model and scaler")

# Create a configuration file for the model pipeline
config = {
    'male_model_path': 'age_model_male.joblib',
    'female_model_path': 'age_model_female.joblib',
    'combined_model_path': 'age_model_combined.joblib',
    'male_scaler_path': 'age_scaler_male.joblib',
    'female_scaler_path': 'age_scaler_female.joblib',
    'combined_scaler_path': 'age_scaler_combined.joblib',
    'use_gender_specific': True
}

joblib.dump(config, 'age_model_config.joblib')
print("Saved model configuration")

# Print performance comparison
print("\n=== Performance Comparison ===")
print(f"Male-specific model: Accuracy = {male_acc * 100:.2f}%, F1 = {male_f1 * 100:.2f}%")
print(f"Female-specific model: Accuracy = {female_acc * 100:.2f}%, F1 = {female_f1 * 100:.2f}%")
print(f"Combined model: Accuracy = {combined_acc * 100:.2f}%, F1 = {combined_f1 * 100:.2f}%")

# Calculate theoretical combined performance (weighted by gender distribution)
male_ratio = len(male_df) / len(df)
female_ratio = len(female_df) / len(df)
weighted_accuracy = (male_acc * male_ratio) + (female_acc * female_ratio)
weighted_f1 = (male_f1 * male_ratio) + (female_f1 * female_ratio)

print(f"\nTheoretical gender-aware pipeline performance:")
print(f"Weighted Accuracy = {weighted_accuracy * 100:.2f}%")
print(f"Weighted F1 = {weighted_f1 * 100:.2f}%")
print(f"Improvement over combined model: {(weighted_accuracy - combined_acc) * 100:.2f}%")

print("\nTraining complete!")