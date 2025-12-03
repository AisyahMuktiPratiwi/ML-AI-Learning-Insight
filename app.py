from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- CONFIG ---
# Gunakan path dinamis agar aman di server manapun
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'no insight/model_gaya_belajar_rf_augmented.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'no insight/scaler_gaya_belajar_augmented.pkl')


# --- LOAD MODEL ---
print(f"Loading model from... {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model & Scaler loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None
    scaler = None

# --- KAMUS INSIGHT (SARAN BELAJAR) ---
# Ini bagian yang menerjemahkan Label menjadi Saran
INSIGHTS = {
    "Fast Learner": {
        "deskripsi": "Kamu cepat memahami konsep baru dan belajar dengan efisien.",
        "saran": [
            "Ambil tantangan coding level Advanced agar tidak bosan setelah menyelesaikan modul.",
            "Luangkan waktu untuk meninjau detail kecil yang mungkin terlewat karena proses belajar yang cepat.",
            "Gunakan sesi belajar singkat sekitar 25 menit untuk menjaga fokus dan konsentrasi."
        ]
    },
    "Reflective": {
        "deskripsi": "Kamu butuh waktu merenung untuk paham mendalam. Kualitas adalah kuncimu.",
        "saran": [
            "Jangan terburu-buru, pastikan kamu memahami konsepnya sampai benar-benar jelas.",
            "Cobalah menulis rangkuman singkat dari materi yang dipelajari untuk mengecek pemahamanmu.",
            "Diskusikan materi di forum untuk perspektif serta pemahaman baru."
        ]
    },
    "Consistent": {
        "deskripsi": "Kamu disiplin dan punya rutinitas stabil.",
        "saran": [
            "Tetap jaga ritme belajarmu, pertahankan, konsistensi berharga.",
            "Tambahkan durasi belajar sekitar 10 menit secara bertahap untuk meningkatkan kapasitasmu.",
            "Sesekali eksplorasi topik baru agar rutinitas belajar tetap terasa segar dan menarik."
        ]
    }
}

@app.route('/', methods=['GET'])
def home():
    return "API Machine Learning (With Insights) is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'status': 'error', 'message': 'Model not loaded correctly on server.'})

    try:
        data = request.get_json()

        # Ambil nilai input
        total_active_days = float(data['total_active_days'])
        avg_study_duration = float(data['avg_study_duration'])
        avg_exam_duration = float(data['avg_exam_duration'])
        avg_submission_rating = float(data['avg_submission_rating'])
        avg_exam_score = float(data['avg_exam_score'])

        # Buat DataFrame sesuai fitur model
        features = ['total_active_days', 'avg_study_duration', 'avg_exam_duration',
                    'avg_submission_rating', 'avg_exam_score']

        df_input = pd.DataFrame([{
            'total_active_days': total_active_days,
            'avg_study_duration': avg_study_duration,
            'avg_exam_duration': avg_exam_duration,
            'avg_submission_rating': avg_submission_rating,
            'avg_exam_score': avg_exam_score
        }], columns=features)

        # Scaling & Prediksi
        X_scaled = scaler.transform(df_input)
        prediction_label = model.predict(X_scaled)[0]

        # Jika model mendukung probabilitas
        probabilities = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            # Ambil semua label kelas
            classes = model.classes_
            probabilities = {str(cls): float(proba[i]) for i, cls in enumerate(classes)}

        # Ambil Insight dari Kamus
        insight_data = INSIGHTS.get(prediction_label, {
            "deskripsi": "Gaya belajar belum teridentifikasi.",
            "saran": []
        })

        # Output JSON Lengkapz
        return jsonify({
            'status': 'success',
            'gaya_belajar': prediction_label,
            'deskripsi': insight_data['deskripsi'],
            'saran': insight_data['saran'],
            'probabilities': probabilities  # Tambahkan probabilitas jika ada
        })

    except KeyError as e:
        return jsonify({'status': 'error', 'message': f'Field {str(e)} tidak ditemukan dalam input.'}), 400
    except ValueError as e:
        return jsonify({'status': 'error', 'message': f'Input harus berupa angka: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# WSGI Entry point
application = app

if __name__ == '__main__':
    app.run(debug=True)