"""
NCA Framework: Unified NeuroCognitive Age Predictor
--------------------------------------------------
Author: Elise Roger, PhD
Affiliation: Assistant Professor, Dept. of Medical Imaging & Radiation Sciences
             Faculty of Medicine and Health Sciences, University of Sherbrooke
Laboratory: IRIS Neurolab / CdRV
Contact: elise.roger@usherbrooke.ca
Date: 2026
"""

import joblib
import pandas as pd
import numpy as np
from scipy.stats import boxcox

def run_nca_pipeline(master_csv_path, brain_model_path, cog_model_path):
    # 1. Load Data
    # The master file must contain: id, chron_age, sex, fluency, education, language + 624 MRI features
    df = pd.read_csv(master_csv_path, sep=';')
    
    # Required column validation
    required = ['id', 'chron_age', 'sex', 'fluency', 'education', 'language']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in CSV: {missing}")

    results_df = df[['id', 'chron_age']].copy()

    # --- PART A: BRAIN AGE (MRI-based) ---
    print("Computing Brain Age (BA)...")
    ba_pipeline = joblib.load(brain_model_path)
    
    # Extract the 624 MRI features directly from the joblib's saved feature list
    X_brain = df[ba_pipeline['features']]
    X_brain_scaled = ba_pipeline['scaler'].transform(X_brain)
    raw_brain_age = ba_pipeline['model'].predict(X_brain_scaled)
    
    # Apply Beheshti Bias Correction (Brain)
    alpha_b = ba_pipeline['beheshti_correction']['alpha']
    beta_b = ba_pipeline['beheshti_correction']['beta']
    results_df['brain_age'] = raw_brain_age - (alpha_b * df['chron_age'] + beta_b)

    # --- PART B: COGNITIVE AGE (Fluency-based) ---
    print("Computing Cognitive Age (CA)...")
    ca_pipeline = joblib.load(cog_model_path)
    
    # Pre-processing (Box-Cox transformation using trained lambda)
    df['fluency_bc'] = boxcox(df['fluency'] + 1, lmbda=ca_pipeline['lambda_bc'])
    
    # Categorical encoding for Sex and Language
    df_cog = pd.get_dummies(df, columns=['sex', 'language'], drop_first=True)
    
    # Feature alignment check (Ensures model-expected columns exist)
    for col in ca_pipeline['features']:
        if col not in df_cog.columns:
            df_cog[col] = 0
            
    X_cog = df_cog[ca_pipeline['features']]
    X_cog_scaled = ca_pipeline['scaler'].transform(X_cog)
    raw_cog_age = ca_pipeline['model'].predict(X_cog_scaled)
    
    # Apply Beheshti Bias Correction (Cognition)
    alpha_c = ca_pipeline['bias_correction']['alpha']
    beta_c = ca_pipeline['bias_correction']['beta']
    results_df['cognitive_age'] = raw_cog_age - (alpha_c * df['chron_age'] + beta_c)

    # --- PART C: MULTIMODAL INTEGRATION (NCA Index) ---
    print("Synthesizing Multimodal NCA Index...")
    
    # Optimized weights anchored on MoCA Z-scores
    w_ca = 0.754
    w_ba = 0.246
    
    # Final NCA Synthesis
    results_df['nca_index'] = (w_ca * results_df['cognitive_age']) + (w_ba * results_df['brain_age'])
    
    # Calculate Gaps (Deviation from Chronological Age)
    results_df['brain_gap'] = results_df['brain_age'] - results_df['chron_age']
    results_df['cognitive_gap'] = results_df['cognitive_age'] - results_df['chron_age']
    results_df['nca_gap'] = results_df['nca_index'] - results_df['chron_age']
    
    # Include MoCA for clinical context
    if 'moca' in df.columns:
        results_df['moca'] = df['moca']

    return results_df.round(2)

if __name__ == "__main__":
    # File configuration
    CSV_FILE = 'demo_nca_master.csv'
    BRAIN_JOB = 'nca_brain_pipeline.joblib'
    COG_JOB = 'nca_cognitive_pipeline.joblib'

    try:
        final_results = run_nca_pipeline(CSV_FILE, BRAIN_JOB, COG_JOB)
        
        print("\n" + "="*70)
        print("          NEUROCOGNITIVE AGE (NCA) MULTIMODAL REPORT")
        print("="*70)
        # Select key columns for terminal display
        display_cols = ['id', 'chron_age', 'brain_age', 'cognitive_age', 'nca_index', 'nca_gap', 'moca']
        print(final_results[display_cols].to_string(index=False))
        print("="*70)
        
        # Save output
        final_results.to_csv('nca_final_results.csv', index=False, sep=';')
        print(f"\n[OK] Results successfully saved to 'nca_final_results.csv'")
        
    except Exception as e:
        print(f"\n[!] Pipeline Error: {e}")

