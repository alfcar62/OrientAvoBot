  # OrientAvoBot (GustAVO) (versione 2 -  leggera con scikit-learn)
  # TF-IDF + MLP Classifier
  Chatbot per orientarsi a scuola, sviluppata in IIS Avogadro (Torino)
  con il supporto degli studenti e di chatGPT (vibe Coding).
  Obiettivo: Sperimentare python e AI per un progetto concreto.
  Utilizza classificatore scikit-learn con MLPClassifier che prende TF-IDF come input e predice l'intents
  La rete neurale è piccola: un solo hidden layer da 10 neuroni.

## 🚀 Avvio rapido

1. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

2. Avvia il server:
   ```bash
   python3 app.py
   ```

3. Apri index.html in un browser  (port 5000)

## 📂 Struttura
- app.py → server Flask con scikit-learn e classificatore MLPClassifier che prende in input TF-IDF
- intents.json → base di conoscenza (intenti e risposte)
- index.html → interfaccia chat
- requirements.txt → librerie necessarie

## 📝 Note
- Modifica `intents.json` per aggiungere nuovi intenti o migliorare quelli esistenti.
- Le risposte vengono scelte casualmente dall'elenco `responses` per ogni intent.


## Tecnologie
- Python
- Flask
- scikit-learn
- TF-IDF
- Rete neurale MLP


