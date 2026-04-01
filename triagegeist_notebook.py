"""
Triagegeist: AI-Powered Emergency Triage Prediction
====================================================
A machine learning solution for predicting Emergency Severity Index (ESI) 
levels from structured patient intake data.

Author: Triagegeist Competition Entry
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("Loading data...")
train = pd.read_csv('/kaggle/input/triagegeist/train.csv')
test = pd.read_csv('/kaggle/input/triagegeist/test.csv')
chief_complaints = pd.read_csv('/kaggle/input/triagegeist/chief_complaints.csv')
patient_history = pd.read_csv('/kaggle/input/triagegeist/patient_history.csv')
sample_submission = pd.read_csv('/kaggle/input/triagegeist/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['triage_acuity'].value_counts().sort_index()}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """
    Create clinically meaningful features for triage prediction.
    
    Features engineered based on:
    - ESI triage criteria (physiologic stability, vital signs)
    - Clinical risk stratification (NEWS2, shock index)
    - Patient risk factors (comorbidities, age)
    - Healthcare utilization patterns
    """
    df = df.copy()
    
    # Merge with patient history
    df = df.merge(patient_history, on='patient_id', how='left')
    
    # --- VITAL SIGN ABNORMALITIES (Core ESI Criteria) ---
    # Hypotension: SBP < 90 mmHg indicates shock/hypoperfusion
    df['hypotension'] = (df['systolic_bp'] < 90).astype(int)
    
    # Tachycardia: HR > 100 bpm indicates physiologic stress
    df['tachycardia'] = (df['heart_rate'] > 100).astype(int)
    
    # Tachypnea: RR > 20/min indicates respiratory distress
    df['tachypnea'] = (df['respiratory_rate'] > 20).astype(int)
    
    # Hypoxemia: SpO2 < 94% indicates inadequate oxygenation
    df['hypoxemia'] = (df['spo2'] < 94).astype(int)
    df['severe_hypoxemia'] = (df['spo2'] < 90).astype(int)
    
    # Altered mental status: GCS < 15 indicates CNS impairment
    df['gcs_abnormal'] = (df['gcs_total'] < 15).astype(int)
    df['gcs_severe'] = (df['gcs_total'] < 9).astype(int)
    
    # Shock index: HR/SBP > 0.9 indicates shock state
    df['shock_index_high'] = (df['shock_index'] > 0.9).astype(int)
    
    # --- CLINICAL RISK SCORES ---
    # NEWS2 high risk: ≥7 points indicates critical illness
    df['news2_high'] = (df['news2_score'] >= 7).astype(int)
    df['news2_medium'] = ((df['news2_score'] >= 5) & (df['news2_score'] < 7)).astype(int)
    
    # --- DEMOGRAPHIC RISK FACTORS ---
    df['age_elderly'] = (df['age'] >= 75).astype(int)
    df['age_pediatric'] = (df['age'] < 18).astype(int)
    
    # --- COMORBIDITY BURDEN ---
    comorb_cols = [c for c in df.columns if c.startswith('hx_')]
    df['comorbidity_count'] = df[comorb_cols].sum(axis=1)
    df['multiple_comorbidities'] = (df['comorbidity_count'] >= 3).astype(int)
    
    # High-risk comorbidity clusters
    df['cardiac_risk'] = ((df['hx_coronary_artery_disease'] == 1) | 
                          (df['hx_heart_failure'] == 1) | 
                          (df['hx_atrial_fibrillation'] == 1)).astype(int)
    
    df['respiratory_risk'] = ((df['hx_copd'] == 1) | 
                              (df['hx_asthma'] == 1)).astype(int)
    
    df['immunocompromised'] = ((df['hx_immunosuppressed'] == 1) | 
                               (df['hx_malignancy'] == 1) | 
                               (df['hx_hiv'] == 1)).astype(int)
    
    # --- TEMPORAL FEATURES ---
    df['is_night'] = ((df['arrival_hour'] < 7) | (df['arrival_hour'] >= 19)).astype(int)
    df['is_weekend'] = df['arrival_day'].isin(['Saturday', 'Sunday']).astype(int)
    
    # --- HEALTHCARE UTILIZATION ---
    df['frequent_flyer'] = (df['num_prior_ed_visits_12m'] >= 3).astype(int)
    df['high_utilizer'] = ((df['num_prior_ed_visits_12m'] >= 3) | 
                           (df['num_prior_admissions_12m'] >= 2)).astype(int)
    
    # --- PAIN ASSESSMENT ---
    df['severe_pain'] = (df['pain_score'] >= 8).astype(int)
    df['moderate_pain'] = ((df['pain_score'] >= 4) & (df['pain_score'] < 8)).astype(int)
    
    # --- BMI CATEGORIES ---
    df['bmi_obese'] = (df['bmi'] >= 30).astype(int)
    df['bmi_underweight'] = (df['bmi'] < 18.5).astype(int)
    
    # --- COMPOSITE VITAL ABNORMALITY SCORE ---
    vital_cols = ['hypotension', 'tachycardia', 'tachypnea', 'hypoxemia', 'gcs_abnormal']
    df['vital_abnormality_count'] = df[vital_cols].sum(axis=1)
    
    # --- CLINICAL IMPUTATION ---
    # Missing vitals imputed with normal values (reflects triage practice)
    normal_values = {
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'mean_arterial_pressure': 93,
        'pulse_pressure': 40,
        'respiratory_rate': 16,
        'temperature_c': 36.8,
        'shock_index': 0.7
    }
    for col, val in normal_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    
    # Pain score: -1 indicates missing, impute with 0 (no pain)
    df['pain_score'] = df['pain_score'].replace(-1, np.nan).fillna(0)
    
    return df

# Apply feature engineering
print("\nEngineering features...")
train_fe = engineer_features(train)
test_fe = engineer_features(test)
print(f"Feature engineering complete. Shape: {train_fe.shape}")

# =============================================================================
# 3. DATA PREPARATION
# =============================================================================
# Exclude non-predictive columns
exclude_cols = ['patient_id', 'site_id', 'triage_nurse_id', 'disposition', 
                'ed_los_hours', 'triage_acuity', 'age_group']

# Encode categorical features
categorical_cols = [c for c in train_fe.columns 
                    if c not in exclude_cols and train_fe[c].dtype == 'object']

print(f"\nEncoding {len(categorical_cols)} categorical features...")
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]], axis=0).astype(str)
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col] = le.transform(test_fe[col].astype(str))

# Prepare feature matrix
feature_cols = [c for c in train_fe.columns if c not in exclude_cols]
X = train_fe[feature_cols]
y = train_fe['triage_acuity']
X_test = test_fe[feature_cols]

print(f"Feature matrix shape: {X.shape}")
print(f"Number of features: {len(feature_cols)}")

# =============================================================================
# 4. MODEL TRAINING
# =============================================================================
# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Random Forest Classifier
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Validation performance
val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred, average='weighted')

print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"Validation Weighted F1: {val_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, val_pred, 
                           target_names=['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5']))

# =============================================================================
# 5. CLINICAL SAFETY ANALYSIS
# =============================================================================
cm = confusion_matrix(y_val, val_pred)

print("\n=== CLINICAL SAFETY METRICS ===")
print("\nUndertriage Rates (CRITICAL - Patient Safety Concern):")
for true_esi in range(1, 6):
    undertriage = sum(cm[true_esi-1, true_esi:])
    total = cm[true_esi-1].sum()
    pct = undertriage / total * 100 if total > 0 else 0
    status = "✓ SAFE" if pct < 5 else "⚠ WARNING"
    print(f"  ESI-{true_esi}: {undertriage}/{total} ({pct:.1f}%) {status}")

print("\nOvertriage Rates (Resource Utilization Concern):")
for true_esi in range(1, 6):
    overtriaged = sum(cm[true_esi-1, :true_esi-1]) if true_esi > 1 else 0
    total = cm[true_esi-1].sum()
    pct = overtriaged / total * 100 if total > 0 else 0
    print(f"  ESI-{true_esi}: {overtriaged}/{total} ({pct:.1f}%)")

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== TOP 15 CLINICAL FEATURES ===")
print(feature_importance.head(15).to_string(index=False))

# =============================================================================
# 7. FINAL PREDICTION & SUBMISSION
# =============================================================================
print("\nGenerating final predictions...")
# Train on full dataset
model_final = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model_final.fit(X, y)

# Predict test set
test_predictions = model_final.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'patient_id': test_fe['patient_id'],
    'triage_acuity': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved! Shape: {submission.shape}")
print(f"Prediction distribution:\n{pd.Series(test_predictions).value_counts().sort_index()}")

# =============================================================================
# 8. VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Triagegeist: AI-Powered Emergency Triage', fontsize=16, fontweight='bold')

# Confusion Matrix
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5'],
            yticklabels=['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5'])
ax1.set_title('Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')

# Feature Importance
ax2 = axes[0, 1]
top_features = feature_importance.head(15)
ax2.barh(range(len(top_features)), top_features['importance'], color='steelblue')
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'])
ax2.set_xlabel('Importance')
ax2.set_title('Top 15 Features')
ax2.invert_yaxis()

# Undertriage Analysis
ax3 = axes[1, 0]
undertriage_rates = []
for true_esi in range(1, 6):
    rate = sum(cm[true_esi-1, true_esi:]) / cm[true_esi-1].sum() * 100
    undertriage_rates.append(rate)
colors = ['#d62728', '#ff7f0e', '#ffdd44', '#2ca02c', '#1f77b4']
ax3.bar(['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5'], undertriage_rates, color=colors, alpha=0.7)
ax3.axhline(y=5, color='red', linestyle='--', label='5% Safety Threshold')
ax3.set_ylabel('Undertriage Rate (%)')
ax3.set_title('Patient Safety: Undertriage Rates')
ax3.legend()

# Prediction Distribution
ax4 = axes[1, 1]
train_dist = y.value_counts(normalize=True).sort_index() * 100
pred_dist = pd.Series(test_predictions).value_counts(normalize=True).sort_index() * 100
x = np.arange(5)
width = 0.35
ax4.bar(x - width/2, train_dist.values, width, label='Training', alpha=0.8)
ax4.bar(x + width/2, pred_dist.values, width, label='Test Predictions', alpha=0.8)
ax4.set_xlabel('ESI Level')
ax4.set_ylabel('Percentage (%)')
ax4.set_title('ESI Distribution Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5'])
ax4.legend()

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n=== NOTEBOOK COMPLETE ===")
print("Files generated:")
print("  - submission.csv: Final predictions for submission")
print("  - model_performance.png: Performance visualization")
