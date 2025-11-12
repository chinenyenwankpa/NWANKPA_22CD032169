from flask import Flask, render_template, request
from deepface import DeepFace
import sqlite3
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---- Database initialization ----
def init_db():
    """Create the users table if it does not exist."""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            image_path TEXT NOT NULL,
            emotion TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database when app starts
init_db()


# ---- Home page ----
@app.route('/')
def index():
    return render_template('index.html')


# ---- Prediction route ----
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    name = request.form.get('name')
    email = request.form.get('email')
    image = request.files.get('image')

    if not image or not name or not email:
        return "Missing name, email, or image.", 400

    # Save uploaded image
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    # Analyze emotion using DeepFace
    try:
        result = DeepFace.analyze(img_path=filepath, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
    except Exception as e:
        emotion = f"Error detecting emotion: {e}"

    # Save user info and emotion into database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users (name, email, image_path, emotion)
        VALUES (?, ?, ?, ?)
    ''', (name, email, filepath, emotion))
    conn.commit()
    conn.close()

    return f"Hi {name}, your detected emotion is: {emotion}"


if __name__ == '__main__':
    # Only for local testing
    app.run(debug=True, host='0.0.0.0', port=5000)
