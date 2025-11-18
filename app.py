from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import re
import json
import random

app = Flask(__name__)
CORS(app)

# Preprocessamento del testo
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\']', ' ', text)
    return ' '.join(text.split())

# Carica intents.json
def load_intents():
    try:
        with open("intents.json", "r", encoding="utf-8") as f:
            intents_data = json.load(f)
            intents = intents_data["intents"]
        print(f"✅ Caricati {len(intents)} intents")
        return intents
    except FileNotFoundError:
        print("❌ ERRORE: File intents.json non trovato!")
        return []

intents = load_intents()

# Prepara i dati di training
X_train = []
y_train = []
all_patterns = []  # Memorizza tutti i pattern originali

for intent in intents:
    for pattern in intent["patterns"]:
        processed_pattern = preprocess_text(pattern)
        X_train.append(processed_pattern)
        y_train.append(intent["tag"])
        all_patterns.append(pattern)  # Salva il pattern originale

# Addestra il classificatore ML
if len(X_train) > 0:
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1)
    X_vectors = vectorizer.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_vectors, y_train)
    print("✅ Classificatore ML addestrato con successo!")
else:
    vectorizer = None
    clf = None
    print("⚠️  Avvertenza: Classificatore non addestrato")

def predict_intent_ml(user_message):
    if clf is None:
        return "errore", 0.0
    
    processed_msg = preprocess_text(user_message)
    print(f"🔍 ML Input: '{user_message}' -> Processato: '{processed_msg}'")
    
    try:
        X_test = vectorizer.transform([processed_msg])
        intent = clf.predict(X_test)[0]
        confidence = max(clf.predict_proba(X_test)[0])
        print(f"🎯 ML Predetto: {intent} (confidence: {confidence:.3f})")
        return intent, confidence
    except Exception as e:
        print(f"❌ Errore predizione ML: {e}")
        return "errore", 0.0

def find_similar_pattern(user_message):
    """Trova il pattern più simile usando cosine similarity"""
    if vectorizer is None:
        return None, 0.0
    
    processed_msg = preprocess_text(user_message)
    user_vector = vectorizer.transform([processed_msg])
    
    best_similarity = 0.0
    best_pattern_index = -1
    
    # Calcola similarità con tutti i pattern di training
    for i, pattern_vector in enumerate(X_vectors):
        similarity = cosine_similarity(user_vector, pattern_vector)[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_pattern_index = i
    
    if best_pattern_index != -1 and best_similarity > 0.4:
        best_intent = y_train[best_pattern_index]
        print(f"📏 Similarità: {best_similarity:.3f} con pattern: '{all_patterns[best_pattern_index]}'")
        return best_intent, best_similarity
    
    return None, 0.0

def generate_response(intent_tag):
    for intent in intents:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "Mi dispiace, non ho capito. Puoi riformulare la domanda?"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data_req = request.get_json()
        if not data_req or "message" not in data_req:
            return jsonify({"answer": "Non ho ricevuto alcun messaggio."})
        
        user_message = data_req["message"].strip()
        print(f"📩 Ricevuto: '{user_message}'")
        
        if len(user_message) < 2:
            return jsonify({"answer": "Il messaggio è troppo breve."})
        
        # Prova con il classificatore ML
        ml_intent, ml_confidence = predict_intent_ml(user_message)
        
        # Se confidence bassa, prova con similarità di pattern
        if ml_confidence < 0.4:
            similar_intent, similarity_score = find_similar_pattern(user_message)
            if similar_intent and similarity_score > ml_confidence:
                ml_intent, ml_confidence = similar_intent, similarity_score
                print(f"🔍 Usato similarità pattern: {ml_intent} ({ml_confidence:.3f})")
        
        print(f"✅ Intent finale: {ml_intent}, Confidence: {ml_confidence:.3f}")
        
        if ml_intent == "errore" or ml_confidence < 0.3:
            return jsonify({"answer": "Non ho capito bene, puoi essere più specifico riguardo a riciclo, energia, acqua o mobilità sostenibile?"})
        
        answer = generate_response(ml_intent)
        return jsonify({
            "intent": ml_intent,
            "confidence": round(float(ml_confidence), 2),
            "answer": answer
        })
        
    except Exception as e:
        print(f"❌ Errore in chat(): {e}")
        return jsonify({"answer": "Si è verificato un errore. Riprova più tardi."})

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)