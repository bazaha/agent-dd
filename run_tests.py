#!/usr/bin/env python3

import sys
import pandas as pd
from bbb_predictor import BBBPredictor
import json

def test_single_predictions():
    """Test single SMILES predictions"""
    print("=== Testing Single Predictions ===")
    
    predictor = BBBPredictor()
    predictor.train_ml_model()
    
    test_smiles = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)OC1=CC=CC=C1C(=O)O",      # Aspirin
        "CN1CCCC1C2=CN=CC=C2",           # Nicotine
    ]
    
    for smiles in test_smiles:
        print(f"\nTesting SMILES: {smiles}")
        result = predictor.predict_single(smiles)
        
        if result.get('Error'):
            print(f"  Error: {result['Error']}")
        else:
            print(f"  Rule-based: {result['Rule_Based']['BBB_Permeable']}")
            print(f"  ML-based: {result['ML_Based']['ML_Prediction']}")
            print(f"  Confidence: {result['ML_Based']['Confidence']:.3f}")

def test_batch_predictions():
    """Test batch predictions from CSV file"""
    print("\n=== Testing Batch Predictions ===")
    
    predictor = BBBPredictor()
    predictor.train_ml_model()
    
    # Read test data
    df = pd.read_csv('test_data.csv')
    smiles_list = df['SMILES'].tolist()
    
    print(f"Testing {len(smiles_list)} compounds from CSV file")
    
    results = predictor.predict_batch(smiles_list)
    
    correct_rule = 0
    correct_ml = 0
    total = len(results)
    
    for i, result in enumerate(results):
        known_bbb = df.iloc[i]['Known_BBB']
        compound_name = df.iloc[i]['Name']
        
        print(f"\n{compound_name} ({known_bbb}):")
        
        if result.get('Error'):
            print(f"  Error: {result['Error']}")
            continue
            
        rule_pred = result['Rule_Based']['BBB_Permeable']
        ml_pred = result['ML_Based']['ML_Prediction']
        
        print(f"  Rule-based: {rule_pred}")
        print(f"  ML-based: {ml_pred}")
        
        # Check rule-based accuracy
        if known_bbb == 'Permeable' and rule_pred in ['High', 'Medium']:
            correct_rule += 1
        elif known_bbb == 'Not Permeable' and rule_pred == 'Low':
            correct_rule += 1
            
        # Check ML accuracy
        if known_bbb == 'Permeable' and 'Permeable' in ml_pred:
            correct_ml += 1
        elif known_bbb == 'Not Permeable' and 'Not Permeable' in ml_pred:
            correct_ml += 1
    
    print(f"\n=== Accuracy Results ===")
    print(f"Rule-based accuracy: {correct_rule}/{total} ({correct_rule/total*100:.1f}%)")
    print(f"ML-based accuracy: {correct_ml}/{total} ({correct_ml/total*100:.1f}%)")

def test_molecular_descriptors():
    """Test molecular descriptor calculation"""
    print("\n=== Testing Molecular Descriptors ===")
    
    predictor = BBBPredictor()
    
    test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    
    descriptors = predictor.calculate_molecular_descriptors(test_smiles)
    
    if descriptors:
        print(f"Calculated {len(descriptors)} descriptors for caffeine:")
        key_descriptors = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA']
        for key in key_descriptors:
            if key in descriptors:
                print(f"  {key}: {descriptors[key]:.3f}")
    else:
        print("Failed to calculate descriptors")

def test_error_handling():
    """Test error handling for invalid SMILES"""
    print("\n=== Testing Error Handling ===")
    
    predictor = BBBPredictor()
    predictor.train_ml_model()
    
    invalid_smiles = [
        "INVALID",
        "",
        "123ABC",
    ]
    
    for smiles in invalid_smiles:
        print(f"\nTesting invalid SMILES: '{smiles}'")
        result = predictor.predict_single(smiles)
        if result.get('Error'):
            print(f"  Correctly caught error: {result['Error']}")
        else:
            print("  Error not caught - this might be a problem")

if __name__ == "__main__":
    try:
        print("Blood-Brain Barrier Predictor Test Suite")
        print("=" * 50)
        
        test_molecular_descriptors()
        test_single_predictions()
        test_batch_predictions()
        test_error_handling()
        
        print("\n=== All Tests Completed ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)