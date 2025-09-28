from flask import Flask, request, render_template, redirect, session, url_for
from functools import wraps
from flask_session import Session
from datetime import timedelta
import pandas as pd
import joblib
import numpy as np
import sqlite3
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'fc15185ac6a4b32d0a3e19cfdcb54945'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=30)  # Session lasts 30 seconds
Session(app)

knn_model = joblib.load('model/knn_model.pkl')
scaler = joblib.load('model/scaler.pkl')
DB_FILE = 'data/student_inputs.db'


# Database initialization
def init_db():

    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama TEXT,
            semester1 REAL,
            semester2 REAL,
            semester3 REAL,
            semester4 REAL,
            semester5 REAL,
            semester6 REAL,
            semester7 REAL,
            stddev REAL,
            trend REAL,
            slope REAL,
            weighted_avg REAL,
            prediction TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()
    

# Login required decorator & Login Route
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/qr_login', methods=['POST'])
def qr_login():
    qr_data = request.form['qr_data']
    # Validate qr_data (e.g., check against user database)
    if qr_data == "user1":  # Replace with your validation logic
        session.permanent = True  # Make session use the lifetime above
        session['user'] = qr_data
        return redirect(url_for('index'))
    return "Invalid QR Code", 401

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# Index Route
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    nama = request.form['nama']
    semester_grades = [
        float(request.form['semester1']),
        float(request.form['semester2']),
        float(request.form['semester3']),
        float(request.form['semester4']),
        float(request.form['semester5']),
        float(request.form['semester6']),
        float(request.form['semester7']),
    ]
    stddev = np.std(semester_grades)
    trend = semester_grades[-1] - semester_grades[0]
    X = np.arange(7).reshape(-1, 1)
    y = np.array(semester_grades).reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0][0]
    weights = [0.1, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25]
    weighted_avg = np.dot(semester_grades, weights)
    input_data = pd.DataFrame([
    semester_grades + [stddev, trend, slope, weighted_avg]
    ], columns=[
        'SEMESTER 1', 'SEMESTER 2', 'SEMESTER 3', 'SEMESTER 4',
        'SEMESTER 5', 'SEMESTER 6', 'SEMESTER 7',
        'STDEV_IPK', 'TREND_IPK', 'SLOPE_IPK', 'WEIGHTED_AVG_IPK'
    ])
    input_data_scaled = scaler.transform(input_data)
    prediction_result = knn_model.predict(input_data_scaled)
    status = 'LULUS' if prediction_result[0] == 1 else 'TIDAK LULUS'
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO inputs (
            nama, semester1, semester2, semester3, semester4, semester5, semester6, semester7,
            stddev, trend, slope, weighted_avg, prediction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (nama, *semester_grades, stddev, trend, slope, weighted_avg, status))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return redirect(url_for('result', record_id=new_id))


# Result Route
@app.route('/result/<int:record_id>')
def result(record_id):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    record = c.execute('SELECT * FROM inputs WHERE id = ?', (record_id,)).fetchone()
    conn.close()
    if record is None:
        return "Record not found", 404
    prediction = f"{record['prediction']}"
    summary = {
        'stddev': record['stddev'],
        'trend': record['trend'],
        'slope': record['slope']
    }
    semester_grades = [
        record['semester1'], record['semester2'], record['semester3'],
        record['semester4'], record['semester5'], record['semester6'], record['semester7']
    ]
    semesters = [f'Semester {i+1}' for i in range(7)]
    plt.figure(figsize=(8, 4.5))
    plt.plot(semesters, semester_grades, marker='o', linestyle='-', color='#2563eb', label='IPK')
    plt.title(f"Grafik Perkembangan IPK: {record['nama']}", fontsize=14)
    plt.xlabel('Semester', fontsize=12)
    plt.ylabel('IPK', fontsize=12)
    plt.ylim(0, 4.2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('index.html', prediction=prediction, summary=summary, plot_url=plot_url)

@app.route('/table')
def table():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT nama, semester1, semester2, semester3, semester4, semester5, semester6, semester7,
               weighted_avg, prediction
        FROM inputs
    ''')
    records = c.fetchall()
    conn.close()
    return render_template('table.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)