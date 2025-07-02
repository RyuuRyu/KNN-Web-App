from flask import Flask, render_template, request
import joblib
import numpy as np
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100))
    absensi = db.Column(db.Float)
    tugas = db.Column(db.Float)
    uts = db.Column(db.Float)
    uas = db.Column(db.Float)
    ipk = db.Column(db.Float)
    result = db.Column(db.String(20))

# Load model and scaler
model = joblib.load('model/knn_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# @app.before_first_request
# def create_tables():
#     db.create_all()

first_request = True
@app.before_request
def before_first_request():
    global first_request
    if first_request:
        db.create_all()
        first_request = False

@app.route('/')
def index():
    all_predictions = Prediction.query.all()
    return render_template('index.html', all_predictions=all_predictions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        absensi = float(request.form['absensi'])
        tugas = float(request.form['tugas'])
        uts = float(request.form['uts'])
        uas = float(request.form['uas'])
        ipk = float(request.form['ipk'])
        nama = request.form['nama']

        features = np.array([[absensi, tugas, uts, uas, ipk]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        result = "Lulus" if prediction[0] == 1 else "Tidak Lulus"

        # Save to database
        pred = Prediction(
            nama=nama,
            absensi=absensi,
            tugas=tugas,
            uts=uts,
            uas=uas,
            ipk=ipk,
            result=result
        )
        db.session.add(pred)
        db.session.commit()

        input_data = {
            'nama': nama,
            'absensi': absensi,
            'tugas': tugas,
            'uts': uts,
            'uas': uas,
            'ipk': ipk
        }

        all_predictions = Prediction.query.all()
        return render_template('index.html', prediction=result, input_data=input_data, all_predictions=all_predictions)
    except Exception as e:
        all_predictions = Prediction.query.all()
        return render_template('index.html', prediction=f"Error: {e}", all_predictions=all_predictions)

if __name__ == '__main__':
    app.run(debug=True)