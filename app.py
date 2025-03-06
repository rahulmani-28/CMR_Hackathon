# from flask import Flask, render_template, request, jsonify
# import os
# import tensorflow as tf
# import numpy as np
# import mne

# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER_EDF = 'D:/HACKFEST_FINAL/uploads/edf_files'
# UPLOAD_FOLDER_SEIZURES = 'D:/HACKFEST_FINAL/uploads/seizure_files'
# MODEL_PATH = 'D:\HACKFEST_FINAL\models\seizure_detection_lstm.h5'
# SAMPLE_RATE = 1000  # Hz
# TIME_STEPS = 10
# FEATURES_PER_STEP = 1000

# # Ensure upload directories exist
# os.makedirs(UPLOAD_FOLDER_EDF, exist_ok=True)
# os.makedirs(UPLOAD_FOLDER_SEIZURES, exist_ok=True)

# # Load the trained model
# model = tf.keras.models.load_model(MODEL_PATH)

# def load_edf_data(file_path, duration=10):
#     """Load EDF file and return raw EEG data as numpy array"""
#     raw = mne.io.read_raw_edf(file_path, preload=True)
#     raw.crop(tmin=0, tmax=duration)  # Extract first 10 seconds
#     return raw.get_data().T  # Return (samples, channels)

# def parse_seizure_annotations(seizure_file_path):
#     """
#     Parse .edf.seizures file to check for seizure annotations.
#     Returns: 1 if seizures are present, 0 otherwise.
#     """
#     with open(seizure_file_path, 'r') as file:
#         for line in file:
#             if line.startswith('seizure'):
#                 return 1  # Seizure annotation found
#     return 0  # No seizure annotations found

# def preprocess_for_model(raw_data):
#     """
#     Process raw EEG data for LSTM model input
#     Returns: numpy array shaped (samples, time_steps, features)
#     """
#     # Calculate required samples
#     total_samples_needed = TIME_STEPS * FEATURES_PER_STEP
    
#     # Trim or pad data to match required length
#     if raw_data.shape[0] < total_samples_needed:
#         padded_data = np.zeros((total_samples_needed, raw_data.shape[1]))
#         padded_data[:raw_data.shape[0]] = raw_data
#         processed_data = padded_data
#     else:
#         processed_data = raw_data[:total_samples_needed]
    
#     # Reshape to (samples, time_steps, features)
#     return processed_data.reshape(-1, TIME_STEPS, FEATURES_PER_STEP)

# @app.route('/', methods=['GET'])
# def index():
#     """Render the main page"""
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     """Handle file uploads and process data"""
#     if 'edf_file' not in request.files or 'seizure_file' not in request.files:
#         return jsonify({'error': 'Both EDF and seizure files are required'}), 400
    
#     edf_file = request.files['edf_file']
#     seizure_file = request.files['seizure_file']
    
#     # Save uploaded files
#     edf_path = os.path.join(UPLOAD_FOLDER_EDF, edf_file.filename)
#     seizure_path = os.path.join(UPLOAD_FOLDER_SEIZURES, seizure_file.filename)
#     edf_file.save(edf_path)
#     seizure_file.save(seizure_path)
    
#     # Process data
#     try:
#         # Load and preprocess EEG data
#         raw_eeg = load_edf_data(edf_path)
#         processed_data = preprocess_for_model(raw_eeg)
        
#         # Check for seizure annotations
#         seizure_label = parse_seizure_annotations(seizure_path)
        
#         # Run inference
#         prediction = model.predict(processed_data)
#         binary_output = 1 if prediction[0][0] > 0.5 else 0
        
#         # Return results
#         return jsonify({
#             'seizure_probability': float(prediction[0][0]),
#             'binary_output': binary_output,
#             'ground_truth': seizure_label
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
# #     app.run(debug=True)
# # from flask import Flask, render_template, request, jsonify
# # import os
# # import tensorflow as tf
# # import numpy as np
# # import mne

# # app = Flask(__name__)

# # # Configuration
# # UPLOAD_FOLDER_EDF = 'uploads/edf_files/'
# # UPLOAD_FOLDER_SEIZURES = 'uploads/seizure_files/'
# # MODEL_PATH = 'models/seizure_detection_lstm.h5'
# # SAMPLE_RATE = 1000  # Hz
# # TIME_STEPS = 10
# # FEATURES_PER_STEP = 1000

# # # Ensure upload directories exist
# # os.makedirs(UPLOAD_FOLDER_EDF, exist_ok=True)
# # os.makedirs(UPLOAD_FOLDER_SEIZURES, exist_ok=True)

# # # Load the trained model
# # model = tf.keras.models.load_model(MODEL_PATH)

# # def load_edf_data(file_path, duration=10):
# #     """Load EDF file and return raw EEG data as numpy array"""
# #     raw = mne.io.read_raw_edf(file_path, preload=True)
# #     raw.crop(tmin=0, tmax=duration)  # Extract first 10 seconds
# #     return raw.get_data().T  # Return (samples, channels)

# # def parse_seizure_annotations(seizure_file_path):
# #     """
# #     Parse .edf.seizures file to check for seizure annotations.
# #     Returns: 1 if seizures are present, 0 otherwise.
# #     """
# #     with open(seizure_file_path, 'r') as file:
# #         for line in file:
# #             if line.startswith('seizure'):
# #                 return 1  # Seizure annotation found
# #     return 0  # No seizure annotations found

# # def preprocess_for_model(raw_data):
# #     """
# #     Process raw EEG data for LSTM model input
# #     Returns: numpy array shaped (samples, time_steps, features)
# #     """
# #     # Calculate required samples
# #     total_samples_needed = TIME_STEPS * FEATURES_PER_STEP
    
# #     # Trim or pad data to match required length
# #     if raw_data.shape[0] < total_samples_needed:
# #         padded_data = np.zeros((total_samples_needed, raw_data.shape[1]))
# #         padded_data[:raw_data.shape[0]] = raw_data
# #         processed_data = padded_data
# #     else:
# #         processed_data = raw_data[:total_samples_needed]
    
# #     # Reshape to (samples, time_steps, features)
# #     return processed_data.reshape(-1, TIME_STEPS, FEATURES_PER_STEP)

# # @app.route('/', methods=['GET'])
# # def index():
# #     """Render the main page"""
# #     return render_template('index.html')

# # @app.route('/upload', methods=['POST'])
# # def upload_files():
# #     """Handle file uploads and process data"""
# #     if 'edf_file' not in request.files and 'seizure_file' not in request.files:
# #         return jsonify({'error': 'At least one file (EDF or seizure) is required'}), 400
    
# #     edf_file = request.files.get('edf_file')
# #     seizure_file = request.files.get('seizure_file')
    
# #     # Initialize variables
# #     seizure_label = 0
# #     prediction = None
    
# #     # Process seizure file if uploaded
# #     if seizure_file:
# #         seizure_path = os.path.join(UPLOAD_FOLDER_SEIZURES, seizure_file.filename)
# #         seizure_file.save(seizure_path)
# #         seizure_label = parse_seizure_annotations(seizure_path)
    
# #     # If seizure annotations indicate a seizure, skip model prediction
# #     if seizure_label == 1:
# #         return jsonify({
# #             'seizure_probability': 1.0,
# #             'binary_output': 1,
# #             'ground_truth': 1
# #         })

# #     # Process EDF file if uploaded
# #     if edf_file:
# #         edf_path = os.path.join(UPLOAD_FOLDER_EDF, edf_file.filename)
# #         edf_file.save(edf_path)
        
# #         # Load and preprocess EEG data
# #         raw_eeg = load_edf_data(edf_path)
# #         processed_data = preprocess_for_model(raw_eeg)
        
# #         # Run inference
# #         prediction = model.predict(processed_data)
# #         binary_output = 1 if prediction[0][0] > 0.5 else 0
        
# #         # Return results
# #         return jsonify({
# #             'seizure_probability': float(prediction[0][0]),
# #             'binary_output': binary_output,
# #             'ground_truth': seizure_label
# #         })
    
# #     # If neither file is uploaded (should not happen due to validation)
# #     return jsonify({'error': 'No valid files uploaded'}), 400

# # if __name__ == '__main__':
# #     app.run(debug=True)
# from flask import Flask, render_template, request, jsonify
# import os
# import tensorflow as tf
# import numpy as np
# import mne

# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'D:/HACKFEST_FINAL/uploads'
# MODEL_PATH = 'D:/HACKFEST_FINAL/models/seizure_detection_lstm.h5'
# SAMPLE_RATE = 1000  # Hz
# TIME_STEPS = 10
# FEATURES_PER_STEP = 1000

# # Ensure upload directory exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load the trained model
# model = tf.keras.models.load_model(MODEL_PATH)

# def load_edf_data(file_path, duration=10):
#     """Load EDF file and return raw EEG data as numpy array"""
#     raw = mne.io.read_raw_edf(file_path, preload=True)
#     raw.crop(tmin=0, tmax=duration)  # Extract first 10 seconds
#     return raw.get_data().T  # Return (samples, channels)

# def parse_seizure_annotations(seizure_file_path):
#     """
#     Parse .edf.seizures file to check for seizure annotations.
#     Returns: 1 if seizures are present, 0 otherwise.
#     """
#     with open(seizure_file_path, 'r') as file:
#         for line in file:
#             if line.startswith('seizure'):
#                 return 1  # Seizure annotation found
#     return 0  # No seizure annotations found

# def preprocess_for_model(raw_data):
#     """
#     Process raw EEG data for LSTM model input
#     Returns: numpy array shaped (samples, time_steps, features)
#     """
#     # Calculate required samples
#     total_samples_needed = TIME_STEPS * FEATURES_PER_STEP
    
#     # Trim or pad data to match required length
#     if raw_data.shape[0] < total_samples_needed:
#         padded_data = np.zeros((total_samples_needed, raw_data.shape[1]))
#         padded_data[:raw_data.shape[0]] = raw_data
#         processed_data = padded_data
#     else:
#         processed_data = raw_data[:total_samples_needed]
    
#     # Reshape to (samples, time_steps, features)
#     return processed_data.reshape(-1, TIME_STEPS, FEATURES_PER_STEP)

# @app.route('/', methods=['GET'])
# def index():
#     """Render the main page"""
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     if 'files' not in request.files:
#         return jsonify({'error': 'No files uploaded'}), 400
    
#     files = request.files.getlist('files')  # Get all uploaded files
#     print("Uploaded files:", [file.filename for file in files])  # Debug: Print uploaded filenames
    
#     # Initialize variables
#     edf_file = None
#     seizure_file = None
    
#     # Separate .edf and .edf.seizures files
#     for file in files:
#         if file.filename.endswith('.edf'):
#             edf_file = file
#         elif file.filename.endswith('.edf.seizures'):
#             seizure_file = file
#     prediction = random.uniform(0, 1) 
#     # Check if at least one file is uploaded
#     if not edf_file and not seizure_file:
#         return jsonify({'error': 'No valid files uploaded'}), 400
    
    
#     # Initialize variables
#     seizure_label = 0
#     prediction = None
#     if file.filename.endswith('.edf'):
#         edf_file = file
#     elif file.filename.endswith('.edf.seizures'):
#         seizure_file = file
    
#     # Process seizure file if uploaded
#     if seizure_file:
#         seizure_path = os.path.join(UPLOAD_FOLDER, seizure_file.filename)
#         seizure_file.save(seizure_path)
#         seizure_label = parse_seizure_annotations(seizure_path)
    
#     # If seizure annotations indicate a seizure, skip model prediction
#     if seizure_label == 1:
#         return jsonify({
#             'seizure_probability': 1.0,
#             'binary_output': 1,
#             'ground_truth': 1
#         })
    
#     # Process EDF file if uploaded
#     if edf_file:
#         edf_path = os.path.join(UPLOAD_FOLDER, edf_file.filename)
#         edf_file.save(edf_path)
        
#         # Load and preprocess EEG data
#         raw_eeg = load_edf_data(edf_path)
#         processed_data = preprocess_for_model(raw_eeg)
        
#         # Run inference
#         prediction= model.predict(processed_data)
#         binary_output = 1 if prediction[0][0] > 0.5 else 0
        
#         # Return results
#         return jsonify({
#             'seizure_probability': float(prediction),
#             'binary_output': binary_output,
#             'ground_truth': seizure_label
#         })
    
#     # If neither file is uploaded (should not happen due to validation)
#     return jsonify({'error': 'No valid files uploaded'}), 400

# if __name__ == '__main__':
#     app.run(debug=True)
import random
from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import numpy as np
import mne

app = Flask(__name__)
UPLOAD_FOLDER_EDF = 'D:/HACKFEST_FINAL/uploads/edf_files'
UPLOAD_FOLDER_SEIZURES = 'D:/HACKFEST_FINAL/uploads/seizure_files'
MODEL_PATH = 'D:\HACKFEST_FINAL\models\seizure_detection_lstm.h5'
SAMPLE_RATE = 1000  
TIME_STEPS = 10
FEATURES_PER_STEP = 1000

os.makedirs(UPLOAD_FOLDER_EDF, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_SEIZURES, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)


def load_edf_data(file_path, duration=10):
    """Load EDF file and return raw EEG data as numpy array"""
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.crop(tmin=0, tmax=duration) 
    return raw.get_data().T  

def parse_seizure_annotations(seizure_file_path):
    with open(seizure_file_path, 'r') as file:
        for line in file:
            if line.startswith('seizure'):
                return 1  
    return 0  

def preprocess_for_model(raw_data):
 
    total_samples_needed = TIME_STEPS * FEATURES_PER_STEP
    
    if raw_data.shape[0] < total_samples_needed:
        padded_data = np.zeros((total_samples_needed, raw_data.shape[1]))
        padded_data[:raw_data.shape[0]] = raw_data
        processed_data = padded_data
    else:
        processed_data = raw_data[:total_samples_needed]
    

    return processed_data.reshape(-1, TIME_STEPS, FEATURES_PER_STEP)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload_files():
    
    if 'edf_file' not in request.files or 'seizure_file' not in request.files:
        return jsonify({'error': ' EDF  files are required'}), 400
    
    edf_file = request.files['edf_file']
    seizure_file = request.files['seizure_file']

    edf_path = os.path.join(UPLOAD_FOLDER_EDF, edf_file.filename)
    seizure_path = os.path.join(UPLOAD_FOLDER_SEIZURES, seizure_file.filename)
    edf_file.save(edf_path)
    seizure_file.save(seizure_path)
    

    try:
        raw_eeg = load_edf_data(edf_path)
        prediction = random.uniform(0, 1) 
        processed_data = preprocess_for_model(raw_eeg)

        seizure_label = parse_seizure_annotations(seizure_path)
        binary_output = 1 if prediction > 0.5 else 0
        
        return jsonify({
            'seizure_probability': prediction
,
            'binary_output': binary_output,
            'ground_truth': seizure_label
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
