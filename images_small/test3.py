# extract_ingredients_from_captions.py
import os, re, json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# HF + SBERT + fuzzy
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

# Optional: spacy for lemmatization; fallback to simple regex
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:
    nlp = None

# --------------------------
# Config
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
SBERT_MODEL = "all-MiniLM-L6-v2"
NUM_CAPTIONS = 5    # generate 5 captions per image (diverse)
TEMPERATURE = 1.2   # >1 for more diversity
TOP_K = 50          # top-k for n-gram candidate selection in lexicon matching
ALPHA = 0.4         # weight fuzzy vs embedding
THRESH = 0.65       # acceptance threshold for ingredients (0..1)
NGRAM_MAX = 3       # use 1..3 grams

# --------------------------
# Load models
# --------------------------
print("Loading BLIP and SBERT...")
processor = BlipProcessor.from_pretrained(BLIP_MODEL)
blip = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(DEVICE)
blip.eval()
sbert = SentenceTransformer(SBERT_MODEL, device=DEVICE)  # uses GPU if available

# --------------------------
# Helper text functions
# --------------------------
_word_re = re.compile(r"[^\w\s\-']+")  # keep hyphens and apostrophes in tokens

def clean_text(s: str) -> str:
    s = s.lower().strip()
    s = _word_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def lemmatize_text(s: str) -> str:
    s = clean_text(s)
    if nlp:
        doc = nlp(s)
        return " ".join([tok.lemma_ for tok in doc])
    # fallback naive
    return s

def ngrams_from_caption(cap: str, n_max: int = 3) -> List[str]:
    tokens = cap.split()
    grams = []
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            grams.append(" ".join(tokens[i:i+n]))
    return list(dict.fromkeys(grams))  # preserve order & unique

# --------------------------
# Lexicon: load or build
# --------------------------
# Option A: load a prepared lexicon file (one ingredient per line)
def load_lexicon(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]
    items = [lemmatize_text(x) for x in items]
    return list(dict.fromkeys(items))

# Option B: small built-in fallback lexicon
FALLBACK_LEXICON = [
    "egg","bacon","cheese","tomato","lettuce","onion","garlic",
    "olive oil","butter","sugar","flour","milk","cream","banana",
    "apple","strawberry","chicken","beef","pork","fish","rice","noodles","potato",
    "salt","pepper","mayonnaise","vinegar","soy sauce","honey","lemon","lime",
    "cinnamon","nutmeg","buttermilk","yogurt","cream cheese"
]

# --------------------------
# Precompute lexicon embeddings
# --------------------------
def build_lexicon_embeddings(lexicon: List[str]):
    texts = [clean_text(x) for x in lexicon]
    emb = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = (emb**2).sum(axis=1, keepdims=True)**0.5 + 1e-12
    emb = emb / norms
    return texts, emb

# --------------------------
# Caption generation: produce m captions per image using sampling
# --------------------------
@torch.no_grad()
def generate_captions_for_image(image_path: str, num_return_sequences: int = NUM_CAPTIONS, temperature: float = TEMPERATURE):
    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(image_path)
    pil = Image.open(image).convert("RGB")
    inputs = processor(images=pil, return_tensors="pt").to(DEVICE)
    outputs = blip.generate(
        **inputs,
        max_length=30,
        num_beams=1,              # set to 1 when sampling
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
    )
    captions = [processor.decode(seq, skip_special_tokens=True).strip() for seq in outputs]
    captions = [lemmatize_text(c) for c in captions]
    return list(dict.fromkeys(captions))  # unique preserve order

# --------------------------
# Matching functions
# --------------------------
import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def best_match_score_for_lexicon(grams: List[str], lexicon_texts: List[str], lexicon_emb: np.ndarray, alpha: float=ALPHA) -> Dict[str, float]:
    """
    Returns dict: lexicon_item -> best combined score over grams
    """
    # embed grams in batch
    grams_clean = [clean_text(g) for g in grams]
    g_emb = sbert.encode(grams_clean, convert_to_numpy=True, show_progress_bar=False)
    # normalize
    g_norms = np.linalg.norm(g_emb, axis=1, keepdims=True) + 1e-12
    g_emb = g_emb / g_norms
    K = len(lexicon_texts)
    scores = {lex: 0.0 for lex in lexicon_texts}
    # iterate lexicon (usually small) and compute best over grams
    for k, lex in enumerate(lexicon_texts):
        lex_vec = lexicon_emb[k]
        best = 0.0
        for i, g in enumerate(grams_clean):
            # fuzzy ratio normalized 0..1
            r = fuzz.token_sort_ratio(g, lex) / 100.0
            s = (np.dot(g_emb[i], lex_vec) + 1.0) / 2.0  # map -1..1 -> 0..1
            comb = alpha * r + (1-alpha) * s
            if comb > best:
                best = comb
        scores[lex] = best
    return scores

# --------------------------
# Main ingredient extraction for one image
# --------------------------
def extract_ingredients_from_captions(captions: List[str], lexicon_texts: List[str], lexicon_emb: np.ndarray,
                                     ngram_max=NGRAM_MAX, alpha=ALPHA, thresh=THRESH) -> List[Tuple[str,float]]:
    # union of all ngrams from all captions
    all_grams = []
    for c in captions:
        all_grams += ngrams_from_caption(c, ngram_max)
    # deduplicate
    all_grams = list(dict.fromkeys(all_grams))
    scores = best_match_score_for_lexicon(all_grams, lexicon_texts, lexicon_emb, alpha=alpha)
    # filter by threshold
    found = [(lex, score) for lex, score in scores.items() if score >= thresh]
    # sort by score desc
    found.sort(key=lambda x: x[1], reverse=True)
    return found

# --------------------------
# Example run on a folder of images
# --------------------------
if __name__ == "__main__":
    import sys
    from PIL import Image

    image_folder = sys.argv[1] if len(sys.argv) > 1 else "data/images-small"
    image_paths = sorted([str(p) for p in Path(image_folder).glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    print("Images:", len(image_paths), "from", image_folder)

    # load lexicon (try local file lexicon.txt else fallback)
    if os.path.exists("lexicon.txt"):
        lexicon = load_lexicon("lexicon.txt")
    else:
        lexicon = [lemmatize_text(x) for x in FALLBACK_LEXICON]

    lex_texts, lex_emb = build_lexicon_embeddings(lexicon)

    out = []
    for p in tqdm(image_paths):
        # generate diverse captions
        caps = generate_captions_for_image(p, num_return_sequences=NUM_CAPTIONS, temperature=TEMPERATURE)
        print("Image:", p, "Captions:", caps)
        found = extract_ingredients_from_captions(caps, lex_texts, lex_emb, ngram_max=NGRAM_MAX, alpha=ALPHA, thresh=THRESH)
        print("Ingredients detected:", found)
        out.append({"image": p, "captions": caps, "ingredients": found})

    with open("ingredients_results.json", "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=2)
    print("Saved results to ingredients_results.json")
