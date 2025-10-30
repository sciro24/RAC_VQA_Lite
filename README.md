# RAG_VQA_Lite — Streamlit App Locale

Questo repository contiene una web app locale (Streamlit) che espone una pipeline ridotta di Visual Question Answering (VQA) e una generazione LLM (opzionale via Google Gemini).

L'obiettivo è: permettere a un utente di caricare un'immagine, porre una domanda in italiano e ottenere una risposta generata dall'LLM basata sull'analisi dell'immagine (classificazione VQANet + saliency map).

Contenuto del repository

- `app.py` — Streamlit application (entrypoint).
- `RAG_VQA.ipynb` — notebook originale usato come riferimento e per esperimenti.
- `requirements.txt` — dipendenze Python consigliate.
- `file/` — contiene i file locali utilizzati dall'app (qui puoi tenere i pesi e i dataset NPZ se vuoi).
- `.gitignore` — regole per evitare di committare file locali/temporanei.

Nota su cosa è incluso nel repo: per scelta tua i file pesi (`*.pth`) e gli NPZ (`*.npz`) nella cartella `file/` non vengono ignorati e quindi possono essere committati; valuta però dimensioni e policy del repository prima di includere asset molto grandi.

Prerequisiti

- Python 3.9+ (consigliato).
- Consiglio: creare e usare un virtual environment (es. `.venv`).

Installazione

1. Crea e attiva l'ambiente virtuale:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installa le dipendenze:

```bash
pip install -r requirements.txt
```

Uso (esempio rapido)

Per eseguire l'app in locale:

```bash
cd /path/to/RAG_VQA_Lite
streamlit run app.py
```

Funzionalità principali

- Upload immagine (jpg/png).
- Inserisci la domanda (in italiano).
- L'app esegue la classificazione VQANet usando il modello in `file/vqa_model_best.pth` (se presente) e mostra:
   - la classe predetta e la confidenza;
   - la saliency map (overlay) confrontata con l'immagine originale.
- Se è configurata la GEMINI_API_KEY (vedi sotto), l'app invierà la domanda + analisi immagine a Google Gemini per generare la risposta LLM. In mancanza della chiave, l'app mostrerà un messaggio di errore e non chiamerà l'LLM.

Configurare la GEMINI API KEY

L'app cerca la chiave in questo ordine (fallback automatico):

1. `st.secrets["GEMINI_API_KEY"]` (file `.streamlit/secrets.toml` oppure la UI secrets di Streamlit)
2. variabile d'ambiente `GEMINI_API_KEY`
