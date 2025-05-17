import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from scipy import stats
from tensorflow import keras
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import sys
from datetime import datetime
from sklearn.metrics import classification_report
import wfdb
from biosppy.signals import ecg
import pykalman

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ecg.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt'}
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150MB
db = SQLAlchemy(app)

# Create upload directory if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
model = keras.models.load_model('student_model.keras')
classes = ['N', 'L', 'R', 'A', 'V']

# Database Model
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    ecg_data = db.Column(db.String(120), nullable=False)
    predictions = db.Column(db.JSON)  # Store multiple predictions as JSON
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio (SNR) in dB."""
    signal = np.asarray(signal)
    noise = np.asarray(noise)
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        raise ValueError("Noise power is zero. Cannot calculate SNR.")
    return 10 * np.log10(signal_power / noise_power)

def simple_kalman_denoise(signal, process_noise=0.1, observation_noise=1.0):
    """Simplified single-pass Kalman filter for ECG denoising"""
    try:
        # Initialize Kalman filter
        kf = pykalman.KalmanFilter(
            transition_matrices=[1],         # State remains constant
            observation_matrices=[1],        # Direct observation
            transition_covariance=[[process_noise]],
            observation_covariance=[[observation_noise]],
            initial_state_mean=signal[0],
            initial_state_covariance=[[1.0]]
        )
        
        # Use smoothing for better results in one pass
        smoothed_state_means, _ = kf.smooth(signal)
        
        return smoothed_state_means.flatten()
    
    except Exception as e:
        print(f"Kalman filter error: {str(e)}")
        return signal  # Return original signal on failure

@app.route('/', methods=['GET', 'POST'])
def patient_form():
    if request.method == 'POST':
        if 'ecg_file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
            
        file = request.files['ecg_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            return redirect(url_for('submit'))
    
    return render_template('patient.html')

@app.route('/evaluate_record', methods=['POST'])
def evaluate_record():
    class_mapping = {'N': 0, 'L': 1, 'R': 2, 'A': 3, 'V': 4}
    record_name = request.form['record_id']
    window_size = 180

    try:
        # Download and load record
        wfdb.dl_database('mitdb', dl_dir='mitdb_inference', records=[record_name])
        signals, _ = wfdb.rdsamp(f'mitdb_inference/{record_name}')
        annotations = wfdb.rdann(f'mitdb_inference/{record_name}', 'atr')
        
        # Preprocess with Kalman filter
        signal_channel = signals[:, 0] if signals.ndim > 1 else signals
        denoised = simple_kalman_denoise(signal_channel)
        normalized = (denoised - np.mean(denoised)) / np.std(denoised)
        
        # Extract beats
        X, y_true = [], []
        for pos, sym in zip(annotations.sample, annotations.symbol):
            if sym in class_mapping and window_size <= pos < len(normalized) - window_size:
                beat = normalized[pos - window_size : pos + window_size]
                X.append(beat)
                y_true.append(class_mapping[sym])
        
        if not X:
            flash("No valid beats found")
            return redirect(url_for('doctor_view'))

        # Predict and generate report
        X = np.array(X).reshape(-1, 360, 1)
        y_pred = model.predict(X).argmax(axis=1)
        report = classification_report(
            y_true, y_pred, 
            target_names=class_mapping.keys(),
            output_dict=True
        )
        return render_template('evaluation_results.html', 
                             record=record_name,
                             report=report)

    except Exception as e:
        flash(f"Error: {str(e)}")
        return redirect(url_for('doctor_view'))

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Validate form data
        name = request.form['name'].strip()
        age = request.form['age'].strip()
        gender = request.form['gender'].strip()
        
        if not all([name, age, gender]):
            flash('All fields are required')
            return redirect(url_for('patient_form'))

        # Handle file upload
        file = request.files['ecg_file']
        if not file or file.filename == '':
            flash('No file selected')
            return redirect(url_for('patient_form'))

        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # Read and validate ECG data
        try:
            ecg_signal = np.loadtxt(upload_path, delimiter=',')
            if len(ecg_signal) < 3600:
                raise ValueError("Signal too short")
        except Exception as e:
            os.remove(upload_path)
            flash(f'Invalid file: {str(e)}')
            return redirect(url_for('patient_form'))

        # Kalman denoising and preprocessing
        try:
            ecg_denoised = simple_kalman_denoise(ecg_signal)
            ecg_normalized = (ecg_denoised - np.mean(ecg_denoised)) / np.std(ecg_denoised)
            
            # R-peak detection and beat extraction
            _, rpeaks = ecg.ecg(signal=ecg_normalized, sampling_rate=360, show=False)[1:3]
            window_size = 180
            beats = [ecg_normalized[pos-window_size:pos+window_size] 
                    for pos in rpeaks if window_size <= pos < len(ecg_normalized)-window_size]
            
            if not beats:
                os.remove(upload_path)
                flash('No valid heartbeats detected')
                return redirect(url_for('patient_form'))

        except Exception as e:
            os.remove(upload_path)
            print(f"Processing error: {str(e)}", file=sys.stderr)
            flash('ECG processing failed')
            return redirect(url_for('patient_form'))

        # Make predictions
        try:
            X = np.array(beats).reshape(-1, 360, 1)
            predictions = model.predict(X, verbose=0)
            class_indices = np.argmax(predictions, axis=1)
            prediction_counts = {cls: int(np.sum(class_indices == idx)) for idx, cls in enumerate(classes)}
        except Exception as e:
            os.remove(upload_path)
            print(f"Prediction error: {str(e)}", file=sys.stderr)
            flash('Prediction failed')
            return redirect(url_for('patient_form'))

        # Save to database
        new_patient = Patient(
            name=name,
            age=int(age),
            gender=gender,
            ecg_data=filename,
            predictions=prediction_counts
        )
        db.session.add(new_patient)
        db.session.commit()

        flash(f'Processed {len(beats)} beats. Results: {prediction_counts}')
        return redirect(url_for('doctor_view'))

    except Exception as e:
        db.session.rollback()
        if 'upload_path' in locals() and os.path.exists(upload_path):
            os.remove(upload_path)
        print(f"System error: {str(e)}", file=sys.stderr)
        flash('Submission failed. Please try again.')
        return redirect(url_for('patient_form'))

@app.route('/doctor')
def doctor_view():
    patients = Patient.query.order_by(Patient.timestamp.desc()).all()
    return render_template('doctor.html', patients=patients)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash('File size exceeds limit')
    return redirect(url_for('patient_form'))

if __name__ == '__main__':
    app.run(debug=True)