import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class BBBPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def calculate_molecular_descriptors(self, smiles):
        """Calculate molecular descriptors from SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            descriptors = {}
            
            # Basic descriptors
            descriptors['MW'] = Descriptors.MolWt(mol)
            descriptors['LogP'] = Descriptors.MolLogP(mol)
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['HBD'] = Descriptors.NumHDonors(mol)
            descriptors['HBA'] = Descriptors.NumHAcceptors(mol)
            descriptors['RotBonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['AromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['SaturatedRings'] = Descriptors.NumSaturatedRings(mol)
            descriptors['AliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            
            # Advanced descriptors
            descriptors['FractionCsp3'] = rdMolDescriptors.CalcFractionCSP3(mol)
            descriptors['MolMR'] = Descriptors.MolMR(mol)
            descriptors['BalabanJ'] = Descriptors.BalabanJ(mol)
            descriptors['BertzCT'] = Descriptors.BertzCT(mol)
            descriptors['Chi0v'] = Descriptors.Chi0v(mol)
            descriptors['Chi1v'] = Descriptors.Chi1v(mol)
            descriptors['Chi2v'] = Descriptors.Chi2v(mol)
            descriptors['Chi3v'] = Descriptors.Chi3v(mol)
            descriptors['Chi4v'] = Descriptors.Chi4v(mol)
            descriptors['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
            descriptors['Kappa1'] = Descriptors.Kappa1(mol)
            descriptors['Kappa2'] = Descriptors.Kappa2(mol)
            descriptors['Kappa3'] = Descriptors.Kappa3(mol)
            descriptors['LabuteASA'] = Descriptors.LabuteASA(mol)
            descriptors['PEOE_VSA1'] = Descriptors.PEOE_VSA1(mol)
            descriptors['PEOE_VSA2'] = Descriptors.PEOE_VSA2(mol)
            descriptors['SMR_VSA1'] = Descriptors.SMR_VSA1(mol)
            descriptors['SMR_VSA2'] = Descriptors.SMR_VSA2(mol)
            descriptors['SlogP_VSA1'] = Descriptors.SlogP_VSA1(mol)
            descriptors['SlogP_VSA2'] = Descriptors.SlogP_VSA2(mol)
            descriptors['EState_VSA1'] = Descriptors.EState_VSA1(mol)
            descriptors['EState_VSA2'] = Descriptors.EState_VSA2(mol)
            descriptors['VSA_EState1'] = Descriptors.VSA_EState1(mol)
            descriptors['VSA_EState2'] = Descriptors.VSA_EState2(mol)
            
            return descriptors
            
        except Exception as e:
            print(f"Error calculating descriptors for {smiles}: {e}")
            return None
    
    def apply_bbb_rules(self, descriptors):
        """Apply rule-based BBB permeability prediction"""
        if descriptors is None:
            return {"BBB_Permeable": "Unknown", "Rule_Score": 0, "Violations": []}
        
        violations = []
        score = 0
        
        # Lipinski's Rule of Five (modified for BBB)
        if descriptors['MW'] > 450:
            violations.append("Molecular weight > 450 Da")
        else:
            score += 1
            
        if descriptors['LogP'] < -1 or descriptors['LogP'] > 4:
            violations.append("LogP not in range -1 to 4")
        else:
            score += 1
            
        if descriptors['TPSA'] > 90:
            violations.append("TPSA > 90 Ų")
        else:
            score += 1
            
        if descriptors['HBD'] > 3:
            violations.append("H-bond donors > 3")
        else:
            score += 1
            
        if descriptors['HBA'] > 8:
            violations.append("H-bond acceptors > 8")
        else:
            score += 1
            
        # Additional BBB-specific rules
        if descriptors['RotBonds'] > 8:
            violations.append("Rotatable bonds > 8")
        else:
            score += 1
            
        # Predict based on score
        if score >= 5:
            prediction = "High"
        elif score >= 3:
            prediction = "Medium"
        else:
            prediction = "Low"
            
        return {
            "BBB_Permeable": prediction,
            "Rule_Score": score,
            "Max_Score": 6,
            "Violations": violations
        }
    
    def train_ml_model(self, training_data=None):
        """Train machine learning model for BBB prediction"""
        if training_data is None:
            # Create synthetic training data for demonstration
            training_data = self.create_synthetic_training_data()
        
        X = training_data.drop(['SMILES', 'BBB_Label'], axis=1)
        y = training_data['BBB_Label']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Create ensemble predictions
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        
        # Simple ensemble voting
        ensemble_pred = np.where((rf_pred + gb_pred) >= 1, 1, 0)
        
        accuracy = accuracy_score(y_test, ensemble_pred)
        print(f"Model trained with accuracy: {accuracy:.3f}")
        
        # Store both models
        self.model = {'rf': rf_model, 'gb': gb_model}
        
        return accuracy
    
    def create_synthetic_training_data(self):
        """Create synthetic training data for demonstration"""
        # Known BBB permeable compounds (SMILES)
        permeable_smiles = [
            "CCN(CC)CCCC(C)NC1=C2C=CC(Cl)=CC2=NC=C1",  # Chloroquine
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)NCC(C1=CC(O)=CC(O)=C1)O",  # Isoprenaline
            "CCCN1CCC(CC1)(C(=O)O)C2=CC=CC=C2",  # Pethidine
            "CN1CCCC1C2=CN=CC=C2",  # Nicotine
        ]
        
        # Known non-BBB permeable compounds
        non_permeable_smiles = [
            "CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NN2CCCCCC2",  # Large molecule
            "NCCCC[C@@H](C(=O)O)N",  # Lysine
            "NC(CCC(=O)O)C(=O)O",  # Glutamic acid
            "C1=CC=C(C=C1)C(C(=O)O)N",  # Phenylalanine
            "OC(=O)CCCCCCCCCCCCCCC",  # Palmitic acid
        ]
        
        data = []
        
        # Process permeable compounds
        for smiles in permeable_smiles:
            descriptors = self.calculate_molecular_descriptors(smiles)
            if descriptors:
                descriptors['SMILES'] = smiles
                descriptors['BBB_Label'] = 1
                data.append(descriptors)
        
        # Process non-permeable compounds
        for smiles in non_permeable_smiles:
            descriptors = self.calculate_molecular_descriptors(smiles)
            if descriptors:
                descriptors['SMILES'] = smiles
                descriptors['BBB_Label'] = 0
                data.append(descriptors)
        
        # Create additional synthetic data points
        np.random.seed(42)
        for _ in range(40):  # Add more synthetic data
            if np.random.random() > 0.5:  # BBB permeable
                descriptors = {
                    'MW': np.random.normal(300, 100),
                    'LogP': np.random.normal(2, 1),
                    'TPSA': np.random.normal(60, 20),
                    'HBD': np.random.poisson(2),
                    'HBA': np.random.poisson(4),
                    'RotBonds': np.random.poisson(4),
                    'AromaticRings': np.random.poisson(1),
                    'SaturatedRings': np.random.poisson(1),
                    'AliphaticRings': np.random.poisson(0),
                    'FractionCsp3': np.random.random(),
                    'MolMR': np.random.normal(80, 30),
                    'BalabanJ': np.random.normal(2, 0.5),
                    'BertzCT': np.random.normal(500, 200),
                    'Chi0v': np.random.normal(10, 5),
                    'Chi1v': np.random.normal(5, 2),
                    'Chi2v': np.random.normal(3, 1),
                    'Chi3v': np.random.normal(2, 1),
                    'Chi4v': np.random.normal(1, 0.5),
                    'HallKierAlpha': np.random.normal(0, 0.5),
                    'Kappa1': np.random.normal(5, 2),
                    'Kappa2': np.random.normal(3, 1),
                    'Kappa3': np.random.normal(2, 1),
                    'LabuteASA': np.random.normal(40, 15),
                    'PEOE_VSA1': np.random.normal(10, 5),
                    'PEOE_VSA2': np.random.normal(15, 7),
                    'SMR_VSA1': np.random.normal(20, 10),
                    'SMR_VSA2': np.random.normal(25, 12),
                    'SlogP_VSA1': np.random.normal(30, 15),
                    'SlogP_VSA2': np.random.normal(35, 17),
                    'EState_VSA1': np.random.normal(40, 20),
                    'EState_VSA2': np.random.normal(45, 22),
                    'VSA_EState1': np.random.normal(50, 25),
                    'VSA_EState2': np.random.normal(55, 27),
                    'SMILES': f"synthetic_permeable_{_}",
                    'BBB_Label': 1
                }
            else:  # Non-BBB permeable
                descriptors = {
                    'MW': np.random.normal(600, 150),
                    'LogP': np.random.normal(-1, 2),
                    'TPSA': np.random.normal(120, 40),
                    'HBD': np.random.poisson(5),
                    'HBA': np.random.poisson(8),
                    'RotBonds': np.random.poisson(8),
                    'AromaticRings': np.random.poisson(2),
                    'SaturatedRings': np.random.poisson(2),
                    'AliphaticRings': np.random.poisson(1),
                    'FractionCsp3': np.random.random(),
                    'MolMR': np.random.normal(150, 50),
                    'BalabanJ': np.random.normal(3, 1),
                    'BertzCT': np.random.normal(800, 300),
                    'Chi0v': np.random.normal(15, 7),
                    'Chi1v': np.random.normal(8, 3),
                    'Chi2v': np.random.normal(5, 2),
                    'Chi3v': np.random.normal(3, 1.5),
                    'Chi4v': np.random.normal(2, 1),
                    'HallKierAlpha': np.random.normal(1, 0.7),
                    'Kappa1': np.random.normal(8, 3),
                    'Kappa2': np.random.normal(5, 2),
                    'Kappa3': np.random.normal(3, 1.5),
                    'LabuteASA': np.random.normal(70, 25),
                    'PEOE_VSA1': np.random.normal(15, 8),
                    'PEOE_VSA2': np.random.normal(20, 10),
                    'SMR_VSA1': np.random.normal(35, 17),
                    'SMR_VSA2': np.random.normal(40, 20),
                    'SlogP_VSA1': np.random.normal(45, 22),
                    'SlogP_VSA2': np.random.normal(50, 25),
                    'EState_VSA1': np.random.normal(60, 30),
                    'EState_VSA2': np.random.normal(65, 32),
                    'VSA_EState1': np.random.normal(70, 35),
                    'VSA_EState2': np.random.normal(75, 37),
                    'SMILES': f"synthetic_non_permeable_{_}",
                    'BBB_Label': 0
                }
            data.append(descriptors)
        
        return pd.DataFrame(data)
    
    def predict_ml(self, descriptors):
        """Make ML-based BBB prediction"""
        if self.model is None or descriptors is None:
            return {"ML_Prediction": "Unknown", "Confidence": 0}
        
        # Convert to DataFrame for consistent feature ordering
        feature_df = pd.DataFrame([descriptors])
        feature_df = feature_df.reindex(columns=self.feature_names, fill_value=0)
        
        # Scale features
        features_scaled = self.scaler.transform(feature_df)
        
        # Get predictions from both models
        rf_pred = self.model['rf'].predict_proba(features_scaled)[0]
        gb_pred = self.model['gb'].predict_proba(features_scaled)[0]
        
        # Ensemble prediction
        avg_prob = (rf_pred + gb_pred) / 2
        prediction = 1 if avg_prob[1] > 0.5 else 0
        confidence = max(avg_prob)
        
        result = {
            "ML_Prediction": "Permeable" if prediction == 1 else "Not Permeable",
            "Confidence": confidence,
            "Probability_Permeable": avg_prob[1],
            "Probability_Non_Permeable": avg_prob[0]
        }
        
        return result
    
    def predict_single(self, smiles):
        """Predict BBB permeability for a single SMILES string"""
        descriptors = self.calculate_molecular_descriptors(smiles)
        
        if descriptors is None:
            return {
                "SMILES": smiles,
                "Error": "Invalid SMILES string",
                "Rule_Based": None,
                "ML_Based": None
            }
        
        rule_prediction = self.apply_bbb_rules(descriptors)
        ml_prediction = self.predict_ml(descriptors)
        
        return {
            "SMILES": smiles,
            "Molecular_Descriptors": descriptors,
            "Rule_Based": rule_prediction,
            "ML_Based": ml_prediction,
            "Error": None
        }
    
    def predict_batch(self, smiles_list):
        """Predict BBB permeability for multiple SMILES strings"""
        results = []
        for smiles in smiles_list:
            result = self.predict_single(smiles)
            results.append(result)
        return results
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, filepath)
    
    def load_model(self, filepath):
        """Load trained model from file"""
        try:
            saved_data = joblib.load(filepath)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_names = saved_data['feature_names']
            return True
        except:
            return False