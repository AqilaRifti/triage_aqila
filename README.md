# Triagegeist Competition Submission

This repository contains all materials for the Triagegeist: AI in Emergency Triage competition submission.

## Files Included

### Required Submission Files

1. **submission.csv** - Final predictions for the test set (20,000 patients)
   - Format: patient_id, triage_acuity (1-5)
   - ESI-1: 782 (3.9%), ESI-2: 3,326 (16.6%), ESI-3: 7,009 (35.0%), ESI-4: 5,428 (27.1%), ESI-5: 3,455 (17.3%)

2. **triagegeist_notebook.py** - Complete Kaggle Notebook (Python script)
   - Data loading and merging
   - Clinical feature engineering (32 new features)
   - Random Forest model training
   - Validation and safety analysis
   - Submission generation
   - Run end-to-end without errors

3. **WRITEUP.md** - Project Writeup (1,950 words)
   - Clinical problem statement
   - Methodology with clinical rationale
   - Results with safety metrics
   - Bias analysis
   - Limitations and future directions
   - Follows Kaggle writeup template

4. **cover_image.png** - Cover image (560x280 px)
   - Professional medical-themed design
   - Title and accuracy metrics

### Supporting Visualizations

5. **model_performance.png** - Comprehensive performance dashboard
   - Confusion matrix
   - Feature importance
   - Undertriage safety analysis
   - Prediction distribution comparison

6. **confusion_matrix.png** - Detailed confusion matrix

7. **feature_importance.png** - Top 25 predictive features

8. **vitals_by_acuity.png** - Vital signs distribution by ESI level

9. **clinical_patterns.png** - Chief complaints and comorbidities analysis

10. **arrival_demographics.png** - ED arrival patterns and demographics

## Model Performance Summary

| Metric | Value |
|--------|-------|
| Validation Accuracy | 82.4% |
| Weighted F1-Score | 0.825 |
| ESI-1 Recall | 95.2% |
| ESI-1 Undertriage Rate | 4.8% (✓ Below 5% safety threshold) |
| ESI-2 Undertriage Rate | 3.0% (✓ Excellent) |

## Key Features (by importance)

1. GCS Total (15.2%) - Glasgow Coma Scale
2. NEWS2 Score (10.7%) - National Early Warning Score
3. Pain Score (10.0%) - Patient-reported pain
4. SpO2 (7.5%) - Oxygen saturation
5. GCS Abnormal (5.3%) - Altered mental status indicator

## How to Submit to Kaggle

1. **Upload Notebook**: Create a new Kaggle Notebook and paste the contents of `triagegeist_notebook.py`
   - Set notebook to PUBLIC
   - Verify it runs end-to-end without errors

2. **Create Writeup**: Go to the competition Writeups tab and create a new writeup
   - Copy content from WRITEUP.md
   - Upload cover_image.png as cover image
   - Attach the public notebook
   - Add project link (GitHub repo or notebook URL)

3. **Submit**: Click the Submit button before the deadline (April 22, 2026, 5:00 AM GMT+7)

## Clinical Safety Highlights

- **Undertriage of ESI-1 patients: Only 4.8%** - Well below the 5% safety threshold
- **Undertriage of ESI-2 patients: Only 3.0%** - Excellent safety profile
- **No systematic bias** across sex or age groups
- **Clinically interpretable features** align with ESI criteria

## Technical Approach

- **Algorithm**: Random Forest Classifier
- **Features**: 69 engineered features including vital abnormalities, clinical risk scores, comorbidity clusters
- **Missing Value Handling**: Clinical imputation with normal values
- **Class Imbalance**: Balanced class weights
- **Validation**: Stratified train-validation split (85/15)


**Competition**: Triagegeist: AI in Emergency Triage  
**Sponsor**: Laitinen-Fredriksson Foundation  
**Submission Date**: April 2026
