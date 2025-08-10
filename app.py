from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import io
import json
from bbb_predictor import BBBPredictor
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'bbb_predictor_secret_key'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize BBB predictor
predictor = BBBPredictor()

# Train the model on startup
print("Training BBB prediction model...")
try:
    accuracy = predictor.train_ml_model()
    print(f"Model trained successfully with accuracy: {accuracy:.3f}")
except Exception as e:
    print(f"Error training model: {e}")

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Predict BBB permeability for a single SMILES string"""
    try:
        data = request.get_json()
        smiles = data.get('smiles', '').strip()
        
        if not smiles:
            return jsonify({'error': 'SMILES string is required'}), 400
        
        result = predictor.predict_single(smiles)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict BBB permeability for multiple SMILES from uploaded file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV or Excel files.'}), 400
        
        # Read the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Read file based on extension
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Check if SMILES column exists
            smiles_column = None
            possible_smiles_columns = ['SMILES', 'smiles', 'Smiles', 'SMILE', 'smile']
            
            for col in possible_smiles_columns:
                if col in df.columns:
                    smiles_column = col
                    break
            
            if smiles_column is None:
                # If no standard SMILES column found, use the first column
                smiles_column = df.columns[0]
            
            smiles_list = df[smiles_column].dropna().astype(str).tolist()
            
            if len(smiles_list) == 0:
                return jsonify({'error': 'No valid SMILES strings found in the file'}), 400
            
            if len(smiles_list) > 1000:
                return jsonify({'error': 'File contains too many compounds. Maximum 1000 compounds allowed.'}), 400
            
            # Make predictions
            results = predictor.predict_batch(smiles_list)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return jsonify({
                'results': results,
                'total_compounds': len(results),
                'smiles_column_used': smiles_column
            })
            
        except Exception as e:
            # Clean up uploaded file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_results', methods=['POST'])
def download_results():
    """Download prediction results as CSV file"""
    try:
        data = request.get_json()
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'No results to download'}), 400
        
        # Convert results to DataFrame
        rows = []
        for result in results:
            row = {'SMILES': result.get('SMILES', '')}
            
            # Add molecular descriptors
            if result.get('Molecular_Descriptors'):
                for key, value in result['Molecular_Descriptors'].items():
                    row[f'Descriptor_{key}'] = value
            
            # Add rule-based predictions
            if result.get('Rule_Based'):
                rule_data = result['Rule_Based']
                row['Rule_Based_Prediction'] = rule_data.get('BBB_Permeable', '')
                row['Rule_Based_Score'] = rule_data.get('Rule_Score', '')
                row['Rule_Based_Max_Score'] = rule_data.get('Max_Score', '')
                row['Rule_Based_Violations'] = '; '.join(rule_data.get('Violations', []))
            
            # Add ML predictions
            if result.get('ML_Based'):
                ml_data = result['ML_Based']
                row['ML_Prediction'] = ml_data.get('ML_Prediction', '')
                row['ML_Confidence'] = ml_data.get('Confidence', '')
                row['ML_Prob_Permeable'] = ml_data.get('Probability_Permeable', '')
                row['ML_Prob_Non_Permeable'] = ml_data.get('Probability_Non_Permeable', '')
            
            # Add error information
            if result.get('Error'):
                row['Error'] = result['Error']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        return send_file(tmp_file_path, as_attachment=True, download_name='bbb_predictions.csv', mimetype='text/csv')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_example_data')
def get_example_data():
    """Get example SMILES data for testing"""
    example_compounds = [
        {
            'name': 'Caffeine',
            'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'known_bbb': 'Permeable'
        },
        {
            'name': 'Aspirin',
            'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'known_bbb': 'Not Permeable'
        },
        {
            'name': 'Morphine',
            'smiles': 'CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5',
            'known_bbb': 'Permeable'
        },
        {
            'name': 'Dopamine',
            'smiles': 'NCCc1ccc(O)c(O)c1',
            'known_bbb': 'Not Permeable'
        },
        {
            'name': 'Diazepam',
            'smiles': 'CN1C(=O)CN=C(C2=CC=CC=C2)C2=C1C=CC(Cl)=C2',
            'known_bbb': 'Permeable'
        }
    ]
    
    return jsonify(example_compounds)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)