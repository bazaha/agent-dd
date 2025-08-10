# Blood-Brain Barrier Permeability Predictor

A comprehensive web application for predicting blood-brain barrier (BBB) permeability of small molecules using molecular descriptors and machine learning approaches.

## Features

- **Dual Prediction Methods**:
  - **Rule-based prediction**: Uses molecular property thresholds based on BBB-specific rules
  - **Machine learning prediction**: Ensemble model combining Random Forest and Gradient Boosting classifiers

- **Web Interface**:
  - Single compound prediction via SMILES input
  - Batch prediction from CSV/Excel file uploads
  - Example compounds with known BBB permeability
  - Interactive results visualization
  - Downloadable CSV results

- **Molecular Descriptors**: Calculates 32+ molecular descriptors including:
  - Basic properties (MW, LogP, TPSA, HBD, HBA)
  - Topological descriptors (Chi indices, Kappa shape indices)
  - Electronic descriptors (PEOE_VSA, EState_VSA)
  - 3D descriptors (MolMR, LabuteASA)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-dd
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The web application will be available at `http://localhost:8080`

## Main Prediction Methods

### Rule-Based Prediction

The rule-based method evaluates compounds against BBB-specific molecular property ranges:

- **Molecular Weight**: ≤ 450 Da
- **LogP**: -1 to 4
- **Topological Polar Surface Area (TPSA)**: ≤ 90 Ų  
- **Hydrogen Bond Donors**: ≤ 3
- **Hydrogen Bond Acceptors**: ≤ 8
- **Rotatable Bonds**: ≤ 8

**Scoring System**:
- **High BBB Permeability**: Score ≥ 5/6 rules satisfied
- **Medium BBB Permeability**: Score 3-4/6 rules satisfied  
- **Low BBB Permeability**: Score < 3/6 rules satisfied

### Machine Learning Prediction

The ML method uses an ensemble approach:

1. **Feature Engineering**: 32 molecular descriptors calculated using RDKit
2. **Model Architecture**: 
   - Random Forest Classifier (100 trees)
   - Gradient Boosting Classifier (100 estimators)
   - Ensemble voting for final prediction
3. **Training Data**: Synthetic dataset with known BBB permeable/non-permeable compounds
4. **Output**: 
   - Binary classification (Permeable/Not Permeable)
   - Confidence score
   - Individual class probabilities

## API Endpoints

### Single Prediction
```bash
POST /predict_single
Content-Type: application/json

{
  "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}
```

### Batch Prediction
```bash
POST /predict_batch
Content-Type: multipart/form-data

Form data: file (CSV/Excel with SMILES column)
```

### Example Response
```json
{
  "SMILES": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
  "Rule_Based": {
    "BBB_Permeable": "High",
    "Rule_Score": 5,
    "Max_Score": 6,
    "Violations": ["LogP not in range -1 to 4"]
  },
  "ML_Based": {
    "ML_Prediction": "Permeable",
    "Confidence": 0.91,
    "Probability_Permeable": 0.91,
    "Probability_Non_Permeable": 0.09
  },
  "Molecular_Descriptors": {
    "MW": 194.194,
    "LogP": -1.0293,
    "TPSA": 61.82,
    "HBD": 0,
    "HBA": 6
  }
}
```

## File Formats

### Input Files (Batch Prediction)
- **Supported formats**: CSV, Excel (.xlsx, .xls)
- **Required column**: 'SMILES', 'smiles', 'Smiles', or similar
- **Limits**: Maximum 1000 compounds per file, 16MB file size

### Output Files
- **Format**: CSV with comprehensive results
- **Columns**: SMILES, molecular descriptors, rule-based predictions, ML predictions, confidence scores

## Example Compounds

The application includes pre-loaded examples:
- **Caffeine**: Known BBB permeable stimulant
- **Aspirin**: Known non-BBB permeable analgesic
- **Morphine**: Known BBB permeable opioid
- **Dopamine**: Known non-BBB permeable neurotransmitter
- **Diazepam**: Known BBB permeable benzodiazepine

## Dependencies

- Flask 3.0.0 - Web framework
- RDKit 2023.9.2 - Molecular descriptor calculation
- Scikit-learn 1.3.2 - Machine learning models
- Pandas 2.1.4 - Data manipulation
- NumPy 1.24.3 - Numerical computing

## License

This project is for educational and research purposes.
