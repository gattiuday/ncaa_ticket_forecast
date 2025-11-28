import pytest
import pandas as pd
import numpy as np
import os
import joblib

# Test Data Cleaning Logic (Mocking the function or testing a small dataframe)
def test_data_cleaning_logic():
    # Create a dummy raw dataframe
    data = {
        'id': [1, 2],
        'season': [2024, 2024],
        'date': ['2024-01-01', '2024-01-02'],
        'attendance': [1000, None], # One missing
        'home_name': ['Team A', 'Team B'],
        'home_id': [101, 102],
        'home_conference_id': [1, 1],
        'home_current_rank': [1, 99],
        'away_name': ['Team C', 'Team D'],
        'away_id': [201, 202],
        'away_conference_id': [2, 2],
        'away_current_rank': [99, 5],
        'venue_id': [301, 302],
        'venue_full_name': ['Arena A', 'Arena B'],
        'venue_address_city': ['City A', 'City B'],
        'venue_address_state': ['State A', 'State B'],
        'neutral_site': [False, False],
        'conference_competition': [True, True],
        'notes_headline': ['', '']
    }
    df = pd.DataFrame(data)
    
    # Apply cleaning logic similar to clean_data.py
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    df['attendance'] = df['attendance'].fillna(0)
    
    assert df['attendance'].isna().sum() == 0
    assert df.loc[1, 'attendance'] == 0
    assert 'day_of_week' in df.columns
    assert 'month' in df.columns

def test_model_loading():
    # Check if model exists
    model_path = "models/xgboost_attendance_model.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert model is not None
    else:
        pytest.skip("Model not found, skipping loading test")

def test_encoders_loading():
    encoders_path = "models/encoders.joblib"
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
        assert 'home_team' in encoders
        assert 'away_team' in encoders
    else:
        pytest.skip("Encoders not found, skipping test")
