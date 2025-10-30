import os
import io
import textwrap
from pathlib import Path
import shutil
import time

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, models
from sentence_transformers import SentenceTransformer
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


class CIFARKBDataset(torch.utils.data.Dataset):
    """Dataset helper per costruire il KB visivo. Deve essere top-level per funzionare con multiprocessing."""
    def __init__(self, cifar_ds, transform):
        self.cifar_ds = cifar_ds
        self.transform = transform
    def __len__(self):
        return len(self.cifar_ds)
    def __getitem__(self, i):
        img, label = self.cifar_ds[i]
        return self.transform(img), label

try:
    import faiss
except Exception:
    faiss = None

# -------------------------
# Config
# -------------------------
PROJECT_ROOT = Path(__file__).parent
FILE_DIR = PROJECT_ROOT / "file"
MODEL_PATH = FILE_DIR / "vqa_model_best.pth"
TRAIN_NPZ = FILE_DIR / "train_dataset_full.npz"

ANSWER_VOCAB = ["aeroplano", "automobile", "uccello", "gatto", "cervo", "cane", "rana", "cavallo", "nave", "camion"]
CFG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model': {
        'question_dim': 384,
        'image_feature_dim': 256,
        'attention_hidden_dim': 128,
        'dropout': 0.3,
    },
    'embedding_model': 'all-MiniLM-L6-v2',
}

DEVICE = torch.device(CFG['device'])


def get_image_transform(is_training: bool = False) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_training:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])


class VQANet(nn.Module):
    def __init__(self, num_answers, question_dim, image_feature_dim, attention_hidden_dim, dropout: float = 0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(512, image_feature_dim, kernel_size=1)
        self.attention_conv = nn.Conv2d(image_feature_dim + question_dim, attention_hidden_dim, 1)
        self.attention_fc = nn.Conv2d(attention_hidden_dim, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(image_feature_dim + question_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(attention_hidden_dim, num_answers),
        )

    def forward(self, image, question_emb, temperature: float = 1.0):
        x = self.backbone(image)
        img_features = self.proj(x)
        B, C, H, W = img_features.shape
        question_emb_expanded = question_emb.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
        combined_features = torch.cat([img_features, question_emb_expanded], dim=1)
        attn_hidden = torch.tanh(self.attention_conv(combined_features))
        logits = self.attention_fc(attn_hidden).view(B, -1)
        logits = logits / max(temperature, 1e-6)
        attn_weights = F.softmax(logits, dim=1).view(B, 1, H, W)
        attended_img_vector = (attn_weights * img_features).sum(dim=[2, 3])
        final_combined = torch.cat([attended_img_vector, question_emb], dim=1)
        return self.fc(final_combined)


@st.cache_resource
def load_models():
    # Embedding model
    emb = SentenceTransformer(CFG['embedding_model'], device='cpu')

    # VQA model
    m_cfg = CFG['model']
    model = VQANet(len(ANSWER_VOCAB), m_cfg['question_dim'], m_cfg['image_feature_dim'], m_cfg['attention_hidden_dim'], dropout=m_cfg.get('dropout', 0.3))
    model.to(DEVICE)
    if MODEL_PATH.exists():
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state)
            loaded = True
        except Exception as e:
            st.warning(f"Impossibile caricare i pesi del modello: {e}")
            loaded = False
    else:
        st.warning(f"File modello non trovato in: {MODEL_PATH}. L'inferenza VQA non sarà disponibile.")
        loaded = False

    return emb, model, loaded


@st.cache_data(ttl=3600)
def build_textual_faiss_index(npz_path: str):
    if not Path(npz_path).exists():
        return None
    data = np.load(npz_path)
    emb = data['emb'].astype(np.float32)
    labels = data['y'].astype(np.int32)
    # embeddings were saved normalized in the notebook; still ensure float32
    if faiss is None:
        return {'emb': emb, 'labels': labels}

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    return {'index': index, 'labels': labels, 'emb': emb}


@st.cache_resource
def build_visual_faiss_index_on_demand(batch_size: int = 256):
    """Costruisce l'indice FAISS visivo usando CIFAR10 (train). Salva le embeddings in file/ per riutilizzo.
    Questo è costoso la prima volta (50k immagini). Viene eseguito solo se l'utente lo richiede dall'interfaccia.
    """
    kb_dir = FILE_DIR / 'cifar_data'
    kb_dir.mkdir(parents=True, exist_ok=True)

    # Carica dataset CIFAR10 (download se necessario)
    cifar_ds = CIFAR10(root=str(kb_dir), train=True, download=True)

    m_cfg = CFG['model']
    # ricrea visual_encoder come nel notebook
    vqa_m = VQANet(len(ANSWER_VOCAB), m_cfg['question_dim'], m_cfg['image_feature_dim'], m_cfg['attention_hidden_dim'], dropout=m_cfg.get('dropout', 0.3))
    # carica pesi se esistono per estrattore visivo
    if MODEL_PATH.exists():
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            vqa_m.load_state_dict(state)
        except Exception:
            pass
    visual_encoder = nn.Sequential(
        vqa_m.backbone,
        vqa_m.proj,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    ).to(DEVICE).eval()

    transform_kb = get_image_transform(is_training=False)

    class CIFARKBDataset(torch.utils.data.Dataset):
        def __init__(self, cifar_ds, transform):
            self.cifar_ds = cifar_ds
            self.transform = transform
        def __len__(self):
            return len(self.cifar_ds)
        def __getitem__(self, i):
            img, label = self.cifar_ds[i]
            return self.transform(img), label

    kb_dataset = CIFARKBDataset(cifar_ds, transform_kb)
    # Use num_workers=0 inside Streamlit to avoid multiprocessing pickling issues on macOS
    kb_loader = DataLoader(kb_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embeddings_list = []
    labels_list = []
    total = len(kb_dataset)

    # Estrai features
    for batch in kb_loader:
        imgs, labs = batch
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            feats = visual_encoder(imgs)
        embeddings_list.append(feats.cpu().numpy())
        labels_list.append(labs.numpy())

    emb_all = np.concatenate(embeddings_list, axis=0).astype(np.float32)
    labels_all = np.concatenate(labels_list, axis=0).astype(np.int32)

    # Salva su disco per riutilizzo
    np.save(FILE_DIR / 'kb_visual_embeddings.npy', emb_all)
    np.save(FILE_DIR / 'kb_visual_labels.npy', labels_all)

    if faiss is None:
        return {'emb': emb_all, 'labels': labels_all, 'dataset_root': str(kb_dir)}

    d = emb_all.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(emb_all)

    return {'index': index, 'labels': labels_all, 'emb': emb_all, 'dataset_root': str(kb_dir)}


@st.cache_resource
def load_visual_kb_from_disk():
    emb_path = FILE_DIR / 'kb_visual_embeddings.npy'
    labels_path = FILE_DIR / 'kb_visual_labels.npy'
    if not emb_path.exists() or not labels_path.exists():
        return None
    emb_all = np.load(emb_path).astype(np.float32)
    labels_all = np.load(labels_path).astype(np.int32)
    if faiss is None:
        return {'emb': emb_all, 'labels': labels_all}
    d = emb_all.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(emb_all)
    return {'index': index, 'labels': labels_all, 'emb': emb_all}


@st.cache_resource
def load_gemini_model(api_key: str):
    """Configura e ritorna il client Gemini (google.generativeai)."""
    try:
        import google.generativeai as genai
    except Exception as e:
        raise ImportError(f"google.generativeai non installato: {e}")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name="gemini-2.5-flash")


def get_gemini_key_from_toml(path: Path, key_name: str = "GEMINI_API_KEY"):
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding='utf-8')
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if key_name in line and '=' in line:
                # naive parse: KEY = "value"
                parts = line.split('=', 1)
                val = parts[1].strip()
                # remove inline comments
                if '#' in val:
                    val = val.split('#', 1)[0].strip()
                # strip quotes
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                return val
    except Exception:
        return None
    return None


def get_gemini_key():
    # 1) st.secrets
    try:
        if isinstance(st.secrets, dict) and "GEMINI_API_KEY" in st.secrets:
            return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    # 2) environment
    if os.environ.get('GEMINI_API_KEY'):
        return os.environ.get('GEMINI_API_KEY')

    # 3) project-level secrets (support both .streamlit and streamlit folders)
    candidates = [PROJECT_ROOT / '.streamlit' / 'secrets.toml', PROJECT_ROOT / 'streamlit' / 'secrets.toml']
    for p in candidates:
        val = get_gemini_key_from_toml(p)
        if val:
            return val
    return None


def retrieve_visual_context_from_kb(visual_features_np: np.ndarray, faiss_vis_data, k: int = 3):
    """Restituisce indici e classi delle immagini visivamente simili dal KB."""
    if faiss_vis_data is None:
        return []

    if visual_features_np.ndim == 1:
        visual_features_np = np.expand_dims(visual_features_np, axis=0)

    if faiss is None or 'index' not in faiss_vis_data:
        emb_store = faiss_vis_data['emb']
        # L2 distance fallback
        dists = np.sum((emb_store - visual_features_np) ** 2, axis=1)
        ids = np.argsort(dists)[:k]
    else:
        D, I = faiss_vis_data['index'].search(visual_features_np.astype(np.float32), k)
        ids = I[0]

    labels = faiss_vis_data['labels'][ids]
    return list(zip(ids.tolist(), labels.tolist()))


def get_vanilla_saliency(model: VQANet, question_embedding: torch.Tensor, input_tensor: torch.Tensor, target_class_idx: int):
    # Wrapper
    class Wrapper(nn.Module):
        def __init__(self, model, q_emb):
            super().__init__()
            self.model = model
            self.q_emb = q_emb
        def forward(self, x):
            B = x.size(0)
            q = self.q_emb.expand(B, -1)
            return self.model(x, q)

    input_copy = input_tensor.clone().detach().requires_grad_(True)
    wrapped = Wrapper(model, question_embedding.clone().detach().to(DEVICE)).to(DEVICE)
    wrapped.eval()
    out = wrapped(input_copy)
    score = out[0, target_class_idx]
    score.backward()
    saliency = input_copy.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)
    saliency = saliency.squeeze(0).cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def generate_saliency_overlay(pil_img: Image.Image, saliency_map: np.ndarray):
    # Resize saliency map to image size and create heatmap overlay
    img_np = np.array(pil_img.convert('RGB'))
    h, w = img_np.shape[:2]
    sal_resized = np.array(Image.fromarray((saliency_map * 255).astype(np.uint8)).resize((w, h)))

    # Prefer matplotlib colormap if available, otherwise fallback to red overlay
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('hot')
        heatmap = cmap(sal_resized / 255.0)[:, :, :3]
    except Exception:
        # Fallback: map saliency to red channel
        norm = (sal_resized.astype(np.float32) / 255.0)
        heatmap = np.zeros_like(img_np, dtype=np.float32) / 255.0
        heatmap[..., 0] = norm  # red channel
        heatmap[..., 1] = 0
        heatmap[..., 2] = 0

    overlay = (0.6 * img_np / 255.0 + 0.4 * heatmap)
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def ui_divider():
    """Insert a visually pleasing divider. Uses st.divider() when available, otherwise a styled markdown rule."""
    try:
        # Streamlit >=1.18
        st.divider()
    except Exception:
        # Fallback: subtle horizontal rule with spacing
        st.markdown("<hr style='border:none;border-top:1px solid #e6e6e6;margin:20px 0;'/>", unsafe_allow_html=True)


def safe_rerun():
    """Try to rerun the Streamlit script; if the API isn't available, show instructions to reload manually."""
    try:
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
            return
    except Exception:
        pass
    # Fallback: instruct user to reload manually
    st.info("Aggiornamento completato — ricarica la pagina del browser per applicare le modifiche (F5 o Ctrl/Cmd+R).")


def mask_key(key: str) -> str:
    """Return a masked representation of a secret (keep first/last 4 chars if possible)."""
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def run_vqa_classification_pil(pil_img: Image.Image, question: str, model: VQANet, emb_model: SentenceTransformer, device: torch.device):
    try:
        img = pil_img.convert('RGB')
        transform = get_image_transform(is_training=False)
        img_t = transform(img).unsqueeze(0).to(device).float()
    except Exception as e:
        return f"ERRORE Immagine: {e}", None, 0.0, None

    # question embedding for VQA (not normalized)
    q_emb_np = emb_model.encode(question, convert_to_numpy=True, normalize_embeddings=False)
    if q_emb_np.ndim == 1:
        q_emb_np = np.expand_dims(q_emb_np, axis=0)
    q_emb_torch = torch.from_numpy(q_emb_np).to(device).float()

    model.eval()
    with torch.no_grad():
        # visual encoder (like in notebook)
        visual_encoder = nn.Sequential(
            model.backbone,
            model.proj,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ).to(device).eval()

        vis_feat_np = visual_encoder(img_t).cpu().numpy()
        out = model(img_t, q_emb_torch)
        probabilities = F.softmax(out, dim=1)
        pred_idx = out.argmax(1).item()
        pred_class = ANSWER_VOCAB[pred_idx]
        confidence = probabilities[0, pred_idx].item() * 100.0

    # format answer similar to notebook
    q = question.strip().lower()
    if q.startswith("c'è un ") or q.startswith("c'è una "):
        prefix_len = len("c'è un ") if q.startswith("c'è un ") else len("c'è una ")
        asked_class = q[prefix_len:-1].strip() if q.endswith('?') else q[prefix_len:].strip()
        if asked_class == pred_class:
            formatted = f"Sì, c'è un/una {pred_class}."
        else:
            formatted = f"No, non c'è un/una {asked_class}. C'è un/una {pred_class}."
    elif q.startswith("che "):
        formatted = f"C'è un/una {pred_class}."
    else:
        formatted = pred_class

    return formatted, pred_class, confidence, vis_feat_np, pred_idx


def retrieve_textual_context(question: str, emb_model: SentenceTransformer, faiss_data, top_k: int = 3):
    if faiss_data is None:
        return "", []
    q_emb = emb_model.encode(question, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    if faiss is None or 'index' not in faiss_data:
        # fallback: compute cosine similarity with stored emb
        emb_store = faiss_data['emb']
        sims = emb_store @ q_emb
        ids = np.argsort(-sims)[:top_k]
    else:
        D, I = faiss_data['index'].search(np.expand_dims(q_emb, axis=0), top_k)
        ids = I[0]

    labels = faiss_data['labels'][ids]
    context = "Esempi simili (classi):\n"
    for i, lab in enumerate(labels):
        context += f"- Esempio {i+1}: classe '{ANSWER_VOCAB[int(lab)]}'\n"
    return context, labels.tolist()


def generate_rag_answer_local(visual_context: str, retrieved_context: str, question: str, formatted_vqa: str):
    # Basic templated fallback answer (when LLM API not configured)
    out = textwrap.dedent(f"""
    Analisi Immagine: {visual_context}

    Conoscenza di supporto (recuperata):
    {retrieved_context}

    Domanda: {question}

    Risposta: {formatted_vqa}
    """)
    return out


def main():
    st.title("RAG-VQA Lite con LLM")

    st.markdown("Carica un'immagine, scrivi una domanda in italiano e premi 'Chiedi' — l'app userà il modello VQA e un semplice retrieval per fornire una risposta.")

    # Sidebar: possibilità di caricare o aggiornare i pesi del modello
    st.sidebar.header("Modello VQA")
    FILE_DIR.mkdir(parents=True, exist_ok=True)
    model_present = MODEL_PATH.exists()
    if model_present:
        st.sidebar.success(f"File pesi trovato: {MODEL_PATH.name}")
        # uploader anche quando il file è presente per permettere aggiornamenti
        uploaded_weights = st.sidebar.file_uploader("Carica nuovo file pesi (.pth)", type=['pth'])
        if uploaded_weights is not None:
            if st.sidebar.button("Aggiorna / Sovrascrivi pesi"):
                try:
                    # crea backup con timestamp
                    ts = int(time.time())
                    backup_path = MODEL_PATH.with_name(f"{MODEL_PATH.name}.bak_{ts}")
                    try:
                        shutil.copy2(MODEL_PATH, backup_path)
                        st.sidebar.info(f"Backup creato: {backup_path.name}")
                    except Exception:
                        # backup non critico, prosegui comunque
                        pass
                    with open(MODEL_PATH, 'wb') as f:
                        f.write(uploaded_weights.getbuffer())
                    st.sidebar.success(f"Pesi aggiornati in: {MODEL_PATH}")
                    safe_rerun()
                except Exception as e:
                    st.sidebar.error(f"Errore aggiornamento pesi: {e}")
        # La funzionalità di rimozione manuale è stata rimossa per evitare cancellazioni accidentali.
    else:
        st.sidebar.warning(f"File pesi non trovato: {MODEL_PATH.name}")
        uploaded_weights = st.sidebar.file_uploader("Carica file pesi (.pth)", type=['pth'])
        if uploaded_weights is not None:
            if st.sidebar.button("Salva pesi nel progetto"):
                try:
                    with open(MODEL_PATH, 'wb') as f:
                        f.write(uploaded_weights.getbuffer())
                    st.sidebar.success(f"Pesi salvati in: {MODEL_PATH}")
                    safe_rerun()
                except Exception as e:
                    st.sidebar.error(f"Errore salvataggio pesi: {e}")
    # Divider e sezione per GEMINI API Key (sotto i pesi)
    ui_divider()
    st.sidebar.header("Google Gemini API Key")
    st.sidebar.markdown("Se vuoi usare Gemini per la generazione LLM, inserisci la GEMINI_API_KEY qui. Verrà salvata in `.streamlit/secrets.toml` nel progetto.")
    # verifica presenza chiave
    existing_key = get_gemini_key()
    if existing_key:
        st.sidebar.success(f"GEMINI_API_KEY presente")
        if st.sidebar.checkbox("Aggiorna / Sovrascrivi la GEMINI_API_KEY"):
            gemini_input = st.sidebar.text_input("Nuova GEMINI_API_KEY", type="password")
            if gemini_input and st.sidebar.button("Salva GEMINI_API_KEY nel progetto"):
                try:
                    secrets_dir = PROJECT_ROOT / '.streamlit'
                    secrets_dir.mkdir(parents=True, exist_ok=True)
                    secrets_path = secrets_dir / 'secrets.toml'
                    # Scrive o aggiorna il secrets.toml in modo semplice (sovrascrive la chiave se già presente)
                    existing = {}
                    if secrets_path.exists():
                        txt = secrets_path.read_text(encoding='utf-8')
                        for line in txt.splitlines():
                            if '=' in line:
                                k, v = line.split('=', 1)
                                existing[k.strip()] = v.strip()
                    existing['GEMINI_API_KEY'] = f'"{gemini_input}"'
                    with open(secrets_path, 'w', encoding='utf-8') as f:
                        for k, v in existing.items():
                            f.write(f"{k} = {v}\n")
                    st.sidebar.success(f"GEMINI_API_KEY salvata in: {secrets_path}")
                    safe_rerun()
                except Exception as e:
                    st.sidebar.error(f"Impossibile salvare la GEMINI_API_KEY: {e}")
    else:
        st.sidebar.warning("GEMINI_API_KEY non trovata")
        gemini_input = st.sidebar.text_input("GEMINI_API_KEY", type="password")
        if gemini_input and st.sidebar.button("Salva GEMINI_API_KEY nel progetto"):
            try:
                secrets_dir = PROJECT_ROOT / '.streamlit'
                secrets_dir.mkdir(parents=True, exist_ok=True)
                secrets_path = secrets_dir / 'secrets.toml'
                # Scrive o aggiorna il secrets.toml in modo semplice (sovrascrive la chiave se già presente)
                existing = {}
                if secrets_path.exists():
                    txt = secrets_path.read_text(encoding='utf-8')
                    for line in txt.splitlines():
                        if '=' in line:
                            k, v = line.split('=', 1)
                            existing[k.strip()] = v.strip()
                existing['GEMINI_API_KEY'] = f'"{gemini_input}"'
                with open(secrets_path, 'w', encoding='utf-8') as f:
                    for k, v in existing.items():
                        f.write(f"{k} = {v}\n")
                st.sidebar.success(f"GEMINI_API_KEY salvata in: {secrets_path}")
                safe_rerun()
            except Exception as e:
                st.sidebar.error(f"Impossibile salvare la GEMINI_API_KEY: {e}")

    emb_model, vqa_model, vqa_loaded = load_models()
    # Carica embedding testuali (NPZ) solo per compatibilità, ma non usiamo retrieval
    _ = build_textual_faiss_index(str(TRAIN_NPZ))

    # Layout: uploader + domanda a sinistra, selettore immagini di test + anteprima a destra
    col_left, col_right = st.columns([2, 1])
    uploaded = col_left.file_uploader("Carica immagine (jpg/png)", type=['jpg', 'jpeg', 'png'])
    question = col_left.text_input("Domanda (in italiano)")

    # Test images selector (if folder exists)
    test_dir = PROJECT_ROOT / 'test_images'
    sample_choices = ["--Nessuna--"]
    if test_dir.exists() and test_dir.is_dir():
        for p in sorted(test_dir.iterdir()):
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                sample_choices.append(p.name)
    selected_sample = col_right.selectbox("Usa immagine di test", sample_choices)
    # preview
    if selected_sample and selected_sample != "--Nessuna--":
        sample_path = test_dir / selected_sample
        try:
            img_preview = Image.open(sample_path)
            col_right.image(img_preview, caption=selected_sample, use_container_width=True)
        except Exception:
            col_right.write("Anteprima non disponibile")
    else:
        col_right.write("Seleziona un'immagine di test per anteprima")

    if st.button("Chiedi"):
        pil_img = None
        if selected_sample and selected_sample != "--Nessuna--":
            sample_path = test_dir / selected_sample
            try:
                pil_img = Image.open(sample_path)
            except Exception as e:
                st.error(f"Errore nel caricamento dell'immagine di test: {e}")
                return
        elif uploaded is not None:
            pil_img = Image.open(io.BytesIO(uploaded.read()))

        if pil_img is None:
            st.error("Per favore carica un'immagine o selezionane una di test.")
            return

        if question.strip() == "":
            st.error("Inserisci una domanda.")
            return

        # divider between the button and the analysis sections
        ui_divider()

        if not vqa_loaded:
            st.warning("Attenzione: modello VQA non caricato. Verrà comunque eseguito il retrieval testuale se disponibile.")

        # Prepara anche tensori per saliency
        transform = get_image_transform(is_training=False)
        img_t = transform(pil_img).unsqueeze(0).to(DEVICE).float()
        q_emb_np = emb_model.encode(question, convert_to_numpy=True, normalize_embeddings=False)
        if q_emb_np.ndim == 1:
            q_emb_np = np.expand_dims(q_emb_np, axis=0)
        q_emb_torch = torch.from_numpy(q_emb_np).to(DEVICE).float()


        # VQA classification (if model available)
        if vqa_loaded:
            formatted, pred_class, confidence, vis_feat, pred_idx = run_vqa_classification_pil(pil_img, question, vqa_model, emb_model, DEVICE)
            st.subheader("Analisi VQA (classificazione)")
            st.write(f"Classe predetta: **{pred_class}** — Confidenza: **{confidence:.2f}%**")
            # divider after VQA section
            ui_divider()
        else:
            formatted, pred_class, confidence, vis_feat, pred_idx = ("", None, 0.0, None, None)
        # Generazione LLM (Gemini) — mostrare la risposta LLM prima della saliency map
        # Recupera la chiave Gemini (st.secrets, env, o file .streamlit/streamlit)
        gemini_key = get_gemini_key()
        llm_answer = None
        if not gemini_key:
            # Quando manca la chiave, informare esplicitamente l'utente che non è possibile ottenere
            # una risposta dall'LLM e che verrà mostrata solo la classificazione VQA (e la saliency se disponibile).
            st.error("GEMINI_API_KEY mancante: non è possibile ottenere una risposta dall'LLM. Verrà mostrata solo la classificazione VQA e la saliency (se disponibile).")
            ui_divider()
        else:
            try:
                gen_model = load_gemini_model(gemini_key)
                # Instruire l'LLM a segnalare esplicitamente quando la confidenza è bassa (<=45%)
                prompt = textwrap.dedent(f"""
                Sei un assistente esperto di Visual Question Answering. Rispondi in italiano, in modo chiaro e completo, basandoti esclusivamente sull'analisi dell'immagine e sulla domanda.

                Se la confidenza della classificazione è inferiore o uguale al 68%, inserisci ESATTAMENTE all'inizio della risposta la frase:
                "⚠️ ATTENZIONE: il livello di confidenza della predizione è basso e il risultato potrebbe non essere accurato."

                Analisi immagine (output del classificatore VQANet): Classe predetta: {pred_class} (Confidenza: {confidence:.2f}%)

                Domanda: {question}

                Fornisci la risposta ora:
                """)
                resp = gen_model.generate_content(prompt)
                llm_answer = getattr(resp, 'text', str(resp))
                st.subheader("Risposta LLM (Gemini)")
                st.write(llm_answer)
                ui_divider()
            except Exception as e:
                st.error(f"Errore durante la chiamata a Gemini: {e}")
                # fallback locale se la chiamata a Gemini fallisce
                visual_context = f"Classe predetta: {pred_class} (Confidenza: {confidence:.2f}%)" if pred_class is not None else "N/D"
                llm_answer = generate_rag_answer_local(visual_context, "", question, formatted)
                st.subheader("Risposta LLM (fallback)")
                st.write(llm_answer)
                ui_divider()

        # Saliency (mostra confronto con immagine originale) — posizionata dopo la risposta LLM
        if vqa_loaded:
            if pred_idx is None:
                st.write("Nessuna predizione disponibile per calcolare la saliency.")
            else:
                saliency_map = get_vanilla_saliency(vqa_model, q_emb_torch, img_t, pred_idx)
                overlay = generate_saliency_overlay(pil_img, saliency_map)
                st.subheader("Saliency map — confronto")
                st.markdown("La saliency map evidenzia le regioni dell'immagine che hanno maggiormente influenzato la predizione del modello VQA. Può aiutare a capire perché il modello ha scelto una certa classe.")
                c1, c2 = st.columns(2)
                c1.image(pil_img, caption='Immagine originale', use_container_width=True)
                c2.image(overlay, caption='Saliency overlay', use_container_width=True)


if __name__ == '__main__':
    main()
