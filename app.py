# GustAVO Chatbot dell'IIS Avogadro di Torino
# versione con rete neurale, cronologia e contesto
# Versione TF-IDF + MLP Classifier
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime

# Soppressione warning di convergenza
warnings.filterwarnings("ignore", category=ConvergenceWarning)

app = Flask(__name__, static_folder=".")
CORS(app)

# Carica intents.json
with open(os.path.dirname(os.path.abspath(__file__)) + "/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

# Prepara dataset (patterns e tag)
patterns = []
tags = []
responses = {}
for intent in intents:
    tag = intent["tag"]
    responses[tag] = intent["responses"]
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(tag)

# TF-IDF per rappresentare i patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Trasformiamo le tag in numeri per la rete neurale
tag_to_idx = {tag: i for i, tag in enumerate(set(tags))}
idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
y = [tag_to_idx[tag] for tag in tags]

# Rete neurale semplice
# mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42, solver='adam')
mlp = MLPClassifier(
        hidden_layer_sizes=(20, 10),  # Due layer, il primo con 20 neuroni, il secondo con 10
        max_iter=1000,
        random_state=42,
        solver='adam',
        alpha=0.001
)
mlp.fit(X, y)

# Cronologia dei messaggi per sessioni
conversations = {}  # session_id -> lista di messaggi
N_HISTORY = 2       # numero di messaggi precedenti da usare nel contesto
MAX_HISTORY = 10    # numero massimo di messaggi salvati per sessione

LOG_FILE = os.path.dirname(os.path.abspath(__file__)) + "/chat_log.json"

def save_chat_log(uuid, user_message, intent, confidence, answer):
    log_entry = {
        "id": str(datetime.now().timestamp()),
        "uuid": uuid,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_message": user_message,
        "intent": intent,
        "confidence": round(float(confidence), 3),
        "answer": answer
    }

    # Carica log esistenti
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append(log_entry)

    # Scrive su file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
        


def classify_intent(user_message, threshold=0.3):
    """Ritorna (intent, confidence) per un messaggio utente usando la rete neurale"""
    vec = vectorizer.transform([user_message])
    probs = mlp.predict_proba(vec)[0]
    best_idx = probs.argmax()
    best_score = probs[best_idx]

    if best_score < threshold:
        return None, best_score

    return idx_to_tag[best_idx], best_score

def generate_response(intent_tag):
    if intent_tag in responses:
        return random.choice(responses[intent_tag])
    return "Non ho capito bene, puoi riformulare?"

# Rotta per servire immagini statiche
@app.route('/img/<path:filename>')
def serve_images(filename):
    try:
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)) + '/img', filename)
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404

# Endpoint per debug dei file
@app.route("/debug-files")
def debug_files():
    img_files = []
    if os.path.exists('img'):
        img_files = os.listdir('img')
    return jsonify({
        "img_folder_exists": os.path.exists('img'),
        "images_in_img_folder": img_files,
        "logo_exists": os.path.dirname(os.path.abspath(__file__)) + "/logoAvogadro.png" in img_files,
        "current_directory": os.getcwd()
    })

# Route per servire la home page
@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")
    
# Route per le cronologie delle chat
@app.route("/feedback/<string:method>", methods=["GET"])
def feedback(method):
    match method:
        case "get":
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                try:
                    intents = json.load(f)
                    return intents
                except json.decoder.JSONDecodeError:
                    return jsonify({"status": "400", "message": "JSON non formattato correttamente o vuoto."})
        case "delete":
            open(LOG_FILE, "w", encoding="utf-8").close()
            return jsonify({"status": "ok", "message": "Feedback eliminati con successo."})
        case _:
            return jsonify({"status": "400", "message": "Method non consentito."})

# Route di test
@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "Orient-AvoBot è attivo!"})

# Route della chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    uuid = data.get("u_id", "-1")
    session_id = data.get("session_id", "default")  # default se non fornito

    if not user_message:
        return jsonify({"answer": "Per favore scrivi qualcosa."})

    # Prendi o inizializza la cronologia
    history = conversations.get(session_id, [])
    history.append({"role": "user", "text": user_message})

    # Prepara messaggio contestuale concatenando ultimi N_HISTORY messaggi dell'utente
    prev_texts = [m["text"] for m in history[-N_HISTORY:] if m["role"] == "user"]
    contextual_message = " ".join(prev_texts + [user_message])

    # Classifica intent usando il messaggio contestualizzato
    intent, confidence = classify_intent(contextual_message)
    if intent is None:
        bot_reply = "Non ho capito bene, puoi riformulare?"
    else:
        bot_reply = generate_response(intent)
        
    # salva su file JSON
    save_chat_log(uuid, user_message, intent, confidence, bot_reply)

    history.append({"role": "bot", "text": bot_reply})
    conversations[session_id] = history[-MAX_HISTORY:]  # salva solo ultimi MAX_HISTORY messaggi

    return jsonify({
        "answer": bot_reply,
        "intent": intent,
        "confidence": round(float(confidence), 2),
        "history": conversations[session_id]  # opzionale, utile per front-end
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5300))
    app.run(host="0.0.0.0", port=port)
