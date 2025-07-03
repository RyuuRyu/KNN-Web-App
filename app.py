from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import numpy as np
import sqlite3

app = Flask(__name__)

knn_model = joblib.load('model/knn_model.pkl')
scaler = joblib.load('model/scaler.pkl')

DB_FILE = 'data/student_inputs.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        semester1 = float(request.form['semester1'])
        semester2 = float(request.form['semester2'])
        semester3 = float(request.form['semester3'])
        semester4 = float(request.form['semester4'])
        semester5 = float(request.form['semester5'])
        semester6 = float(request.form['semester6'])
        semester7 = float(request.form['semester7'])

        # Calculate features
        ipks = [semester1, semester2, semester3, semester4, semester5, semester6, semester7]
        stddev = np.std(ipks)
        trend = semester7 - semester1

        # Slope calculation (linear regression)
        X = np.arange(7).reshape(-1, 1)
        y = np.array(ipks).reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0][0]

        # Weighted average
        weights = [0.1, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25]
        weighted_avg = np.dot(ipks, weights)

        # Prepare input for prediction (order must match training)
        input_data = pd.DataFrame([[
            semester1, semester2, semester3, semester4, semester5, semester6, semester7,
            stddev, trend, slope, weighted_avg
        ]], columns=[
            'SEMESTER 1', 'SEMESTER 2', 'SEMESTER 3', 'SEMESTER 4',
            'SEMESTER 5', 'SEMESTER 6', 'SEMESTER 7',
            'STD_DEV_IPK', 'TREND_IPK', 'SLOPE_IPK', 'WEIGHTED_AVG_IPK'
        ])
        input_data_scaled = scaler.transform(input_data)

        prediction_result = knn_model.predict(input_data_scaled)
        status = 'LULUS' if prediction_result[0] == 1 else 'TIDAK LULUS'
        prediction = f'Prediksi Status: {status}'

        # Save to database
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO inputs (
                semester1, semester2, semester3, semester4, semester5, semester6, semester7,
                stddev, trend, slope, weighted_avg, prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (semester1, semester2, semester3, semester4, semester5, semester6, semester7,
              stddev, trend, slope, weighted_avg, status))
        conn.commit()
        conn.close()
        return redirect(url_for('home'))

    # Fetch all records for display
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT semester1, semester2, semester3, semester4, semester5, semester6, semester7,
               stddev, trend, slope, weighted_avg, prediction
        FROM inputs
    ''')
    records = c.fetchall()
    conn.close()

    return render_template('index.html', prediction=prediction, records=records)

if __name__ == '__main__':
    app.run(debug=True)