from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
import threading

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt')

model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')


# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    stopword_list = nltk.corpus.stopwords.words('english')
    tokens = text.split()

    filtered_tokens = [t for t in tokens if t not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)

    ps = nltk.porter.PorterStemmer()

    return ' '.join([ps.stem(word) for word in filtered_text.split()])


# =========================
# THREAD SAFE PREDICTION
# =========================
prediction_holder = {}

def threaded_predict(text, key):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]

    prediction_holder[key] = "Positive" if pred == 1 else "Negative"


# =========================
# HOME PAGE (HTML)
# =========================
@app.route('/')
def index():
    return render_template('index.html')


# =========================
# PREDICT (BOTH HTML + POSTMAN)
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # ========== POSTMAN (JSON) ==========
        if request.is_json:
            data = request.get_json()
            text = data.get('text')

        # ========== HTML FORM ==========
        else:
            text = request.form.get('text')

        if not text or text.strip() == "":
            if request.is_json:
                return jsonify({"error": "Please enter text"}), 400
            return render_template('index.html', prediction="Please enter text")

        # Threading (required for lab concept)
        key = "user1"

        thread = threading.Thread(
            target=threaded_predict,
            args=(text, key)
        )

        thread.start()
        thread.join()

        result = prediction_holder.get(key)

        # Return based on request type
        if request.is_json:
            return jsonify({"prediction": result})

        return render_template('index.html', prediction=result)

    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 500

        return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)