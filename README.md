# RAG_VQA_Lite — Streamlit App Locale

Questa cartella contiene una semplice web app Streamlit che riutilizza la logica del notebook `RAG_VQA.ipynb` per fornire:

- Upload locale di un'immagine
- Campo per porre una domanda in italiano
- Inferenza VQA (classificazione) con il modello salvato in `file/vqa_model_best.pth`
- Retrieval testuale semplice usando gli embedding nel file `file/train_dataset_full.npz`
- Generazione RAG tramite Google Gemini se è impostata la variabile d'ambiente `GEMINI_API_KEY`, altrimenti un fallback testuale locale

File principali:

- `app.py`: Streamlit app (entrypoint)
- `requirements.txt`: dipendenze consigliate

Requisiti

1. Python 3.9+ (consigliato)
2. File con i pesi e NPZ già presenti nella cartella `file/` (già presenti nel repository):
   - `file/vqa_model_best.pth`
   - `file/train_dataset_full.npz`

Installazione (esempio rapido)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Avvio locale

```bash
streamlit run app.py
```

Note e suggerimenti

- Se vuoi che l'app usi l'API Gemini per la generazione RAG, imposta la variabile d'ambiente `GEMINI_API_KEY` prima di avviare l'app:

```bash
export GEMINI_API_KEY="la_tua_api_key"
streamlit run app.py
```

Alternativa (Streamlit secrets)

Se preferisci non esportare la variabile d'ambiente a livello di sistema, puoi usare i `secrets` di Streamlit.
1. Crea la cartella `.streamlit` nella root del progetto (se non esiste).
2. Crea un file `.streamlit/secrets.toml` con il seguente contenuto:

```toml
GEMINI_API_KEY = "la_tua_api_key"
```

3. In `app.py` l'app leggerà automaticamente `os.environ['GEMINI_API_KEY']` se la variabile d'ambiente è impostata; in alternativa puoi modificare `app.py` per leggere `st.secrets["GEMINI_API_KEY"]` se preferisci quella modalità.

- Se il modello VQA non viene trovato in `file/vqa_model_best.pth`, l'app mostrerà un messaggio e funzionerà almeno per la parte di retrieval (se `train_dataset_full.npz` è disponibile).

- L'approccio di retrieval usato qui è basato sugli embedding testuali salvati nel NPZ (non su un indice visivo), così evitano elaborazioni estese o la necessità di ricalcolare embedding visivi su grandi dataset.

Se vuoi che integri un pipeline più completa (es. costruzione di un indice visivo, saliency overlay, o integrazione con un LLM locale Hugging Face), dimmelo e lo aggiungo.
