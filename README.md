# EcoAvoBot (versione leggera con scikit-learn)
#  classificatore scikit-learn + TF-IDF

## 🚀 Avvio rapido

1. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

2. Avvia il server:
   ```bash
   python3 app.py
   ```

3. Apri `index.html` in un browser.

## 📂 Struttura
- app.py → server Flask con classificatore TF-IDF
- intents.json → base di conoscenza (intenti e risposte)
- index.html → interfaccia chat
- requirements.txt → librerie necessarie

## 📝 Note
- Modifica `intents.json` per aggiungere nuovi intenti o migliorare quelli esistenti.
- Le risposte vengono scelte casualmente dall'elenco `responses` per ogni intent.
