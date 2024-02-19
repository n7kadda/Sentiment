from flask import Flask, request, jsonify, render_template
import joblib
import os
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load(r"D:\Nikunj\Projects\New folder\sentiment\tinder_sentiment_model.pkl")
vectorizer = joblib.load(r'D:\Nikunj\Projects\New folder\sentiment\tinder_sentiment_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    content = data['content']
    # Vectorize the input text
    input_text = vectorizer.transform([content])
    # Make prediction
    prediction = model.predict(input_text)[0]
    # Map prediction to sentiment
    sentiment_map = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
    sentiment = sentiment_map[prediction]
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)