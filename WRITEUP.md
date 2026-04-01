# Triagegeist: AI-Powered Emergency Triage Prediction

## Clinical Problem Statement

Emergency department triage is one of the most critical decisions in healthcare. Every minute counts when patients present with life-threatening conditions, yet triage nurses must make rapid severity assessments under extreme cognitive load, with incomplete information, and in chronically understaffed environments. Errors in triage—particularly undertriage of high-acuity patients—lead to delayed care, adverse outcomes, and preventable deaths.

The Emergency Severity Index (ESI), the most widely used triage system in the United States, relies primarily on unaided human judgment. Inter-rater variability is well-documented, with studies showing agreement rates as low as 60% between experienced nurses. This project addresses a specific, clinically meaningful question: **Can machine learning accurately predict ESI acuity levels from structured patient intake data, and can such a system improve patient safety by reducing undertriage errors?**

## Methodology

### Data Overview

This analysis uses the Triagegeist synthetic dataset comprising 80,000 training records and 20,000 test records from a simulated 24-month period across a fictional multi-site hospital network. The dataset includes:

- **Structured intake data**: Vital signs (BP, HR, RR, SpO2, temperature, GCS), demographics, arrival characteristics
- **Patient history**: 25 binary comorbidity flags (hypertension, diabetes, cardiac disease, etc.)
- **Target variable**: ESI level (1-5), where 1 = immediate life threat, 5 = non-urgent

The training data shows a realistic distribution: ESI-1 (4.0%), ESI-2 (16.8%), ESI-3 (36.2%), ESI-4 (28.8%), ESI-5 (14.2%).

### Feature Engineering

Features were engineered based on established clinical criteria and ESI guidelines:

**Physiologic Abnormalities (Core ESI Criteria)**
- Binary flags for hypotension (SBP <90), tachycardia (HR >100), tachypnea (RR >20)
- Hypoxemia indicators (SpO2 <94%, severe <90%)
- Altered mental status (GCS <15, severe <9)
- Shock index elevation (HR/SBP >0.9)

**Clinical Risk Stratification**
- NEWS2 score categories (high ≥7, medium 5-6)
- Vital abnormality count (composite score)

**Patient Risk Factors**
- Age categories (elderly ≥75, pediatric <18)
- Comorbidity burden (count and clusters: cardiac, respiratory, immunocompromised)
- Healthcare utilization patterns (frequent ED visitors)

**Temporal Context**
- Night/weekend arrival indicators

**Missing Value Imputation**
Missing vital signs were imputed with clinically normal values (SBP 120, HR 70, RR 16, SpO2 98%), reflecting the realistic triage scenario where missing vitals often indicate stable patients for whom complete assessment was deferred.

### Model Architecture

A Random Forest classifier was selected for its:
- Strong performance on tabular medical data
- Built-in feature importance for clinical interpretability
- Robustness to outliers and missing values
- Ability to capture non-linear relationships between vitals and acuity

**Hyperparameters**: 150 estimators, max depth 18, balanced class weights to address the 4% prevalence of ESI-1 cases.

### Validation Strategy

Stratified train-validation split (85/15) preserved the imbalanced class distribution. Model performance was evaluated using accuracy, weighted F1-score, and per-class precision/recall. Clinical safety was assessed through undertriage and overtriaged rates by ESI level.

## Results

### Model Performance

The model achieved **82.4% accuracy** and **0.825 weighted F1-score** on the validation set:

| ESI Level | Precision | Recall | F1-Score | Clinical Interpretation |
|-----------|-----------|--------|----------|------------------------|
| ESI-1 (Immediate) | 0.93 | 0.95 | 0.94 | Excellent - Critical patients correctly identified |
| ESI-2 (Emergent) | 0.97 | 0.95 | 0.96 | Excellent - Emergent cases accurately captured |
| ESI-3 (Urgent) | 0.88 | 0.84 | 0.86 | Good - Most urgent cases correctly triaged |
| ESI-4 (Less Urgent) | 0.74 | 0.69 | 0.71 | Moderate - Some confusion with ESI-3 |
| ESI-5 (Non-urgent) | 0.68 | 0.87 | 0.76 | Good - High recall prevents overtriaged |

### Clinical Safety Analysis

**Undertriage Rates** (predicted lower acuity than actual - PATIENT SAFETY CONCERN):
- ESI-1: **4.8%** (23/483 cases) ✓ Below 5% safety threshold
- ESI-2: **3.0%** (61/2,016 cases) ✓ Excellent safety profile
- ESI-3: **15.7%** (681/4,338 cases)
- ESI-4: **19.2%** (664/3,453 cases)
- ESI-5: **0.0%** (0/1,710 cases)

The model demonstrates exceptional safety for the most critical patients: **only 4.8% of ESI-1 patients would be undertriaged**, well below the 5% safety threshold. This is the most important metric for clinical deployment.

**Overtriage Rates** (predicted higher acuity than actual - RESOURCE CONCERN):
- ESI-1: 0.0% (no overtriaged possible)
- ESI-2: 1.8% (minimal resource impact)
- ESI-3: 0.6% (excellent)
- ESI-4: 11.6% (moderate - some resource inefficiency)
- ESI-5: 12.9% (moderate - some resource inefficiency)

### Feature Importance

The top predictive features align with clinical expectations:

1. **GCS Total (15.2%)** - Altered mental status is the strongest predictor of high acuity
2. **NEWS2 Score (10.7%)** - Validated early warning score for critical illness
3. **Pain Score (10.0%)** - Patient-reported symptom severity
4. **SpO2 (7.5%)** - Oxygen saturation, critical for respiratory assessment
5. **GCS Abnormal (5.3%)** - Binary indicator of altered mental status

Vital signs (respiratory rate, MAP, temperature, heart rate) collectively contribute 25% of predictive power, confirming the model's reliance on physiologic stability—the core principle of ESI triage.

### Bias Analysis

The model was evaluated for demographic bias:

| Group | Accuracy | Sample Size |
|-------|----------|-------------|
| Female | 82.7% | 5,799 |
| Male | 82.3% | 5,905 |
| Other Sex | 79.1% | 296 |
| Non-elderly (<75) | 82.4% | 9,850 |
| Elderly (≥75) | 82.6% | 2,150 |

Performance is consistent across sex and age groups, with the "Other" sex category showing slightly lower accuracy (79.1% vs ~82.5%), likely due to smaller sample size. No systematic bias against protected groups was detected.

## Limitations

1. **Synthetic Data**: This analysis uses simulated data from the Laitinen-Fredriksson Foundation. Real-world validation on clinical datasets (MIMIC-IV-ED, NHAMCS) is needed before deployment.

2. **Missing NLP**: Chief complaint free-text narratives were not incorporated. Future work should apply clinical NLP (ClinicalBERT, MedSpaCy) to extract symptom severity and temporal patterns from unstructured notes.

3. **Static Prediction**: The model predicts acuity at a single timepoint. A deterioration risk model for patients already in the waiting room would provide additional clinical value.

4. **ESI-4/ESI-5 Confusion**: The model shows moderate confusion between lower acuity levels (ESI-4 recall 69%, ESI-5 precision 68%). This reflects the inherent subjectivity in distinguishing "less urgent" from "non-urgent" cases—an area where human judgment may remain valuable.

5. **External Validity**: The model was trained on a single simulated hospital network. Generalizability to different populations, geographic regions, and healthcare systems requires validation.

## Reproducibility

All code is available in the attached Kaggle Notebook. The pipeline runs end-to-end in approximately 3 minutes on standard hardware. Key dependencies: pandas, scikit-learn, matplotlib, seaborn. Random seed (42) ensures reproducible results.

To reproduce:
1. Load train.csv, test.csv, chief_complaints.csv, patient_history.csv
2. Run feature engineering pipeline (engineer_features function)
3. Encode categoricals with LabelEncoder
4. Train RandomForestClassifier with specified hyperparameters
5. Generate predictions and submit

## Clinical Impact and Future Directions

This proof-of-concept demonstrates that machine learning can accurately predict ESI acuity with strong safety margins for critical patients. A deployed system could:

1. **Reduce Undertriage**: Flag high-risk patients for immediate physician review
2. **Support Novice Triage Nurses**: Provide decision support during training
3. **Quality Assurance**: Identify cases for retrospective review and feedback
4. **Resource Planning**: Predict acuity mix for staffing optimization

Future work should integrate:
- Real-time vital sign streams from bedside monitors
- Chief complaint NLP for symptom extraction
- Deterioration risk models for waiting room patients
- Prospective clinical validation in emergency departments

## Conclusion

The Triagegeist model achieves 82.4% accuracy in predicting ESI acuity levels with exceptional safety performance: only 4.8% of ESI-1 patients would be undertriaged. The model's reliance on clinically interpretable features (GCS, NEWS2, vital signs) aligns with established triage principles, supporting its credibility for clinical decision support. While limitations exist—particularly the need for real-world validation—this work demonstrates the potential for AI to enhance emergency triage safety and consistency.

---

**Keywords**: Emergency medicine, triage, ESI, machine learning, clinical decision support, patient safety

**Data Source**: Triagegeist Dataset, Laitinen-Fredriksson Foundation (synthetic data for research purposes)
