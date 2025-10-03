# run_small_pipeline.py
import os, json, math
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Torch + HF + SBERT
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# === CONFIG ===
IMAGE_DIR = "data/images-small"   # place 10-20 images here
USDA_JSON = "data/FoodData_Central_foundation_food_json_2025-04-24.json"  # uploaded file. :contentReference[oaicite:2]{index=2}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
SBERT_MODEL = "all-MiniLM-L6-v2"  # small & fast
TOP_K = 3   # retrieve top-3 USDA items
NUM_CAPTIONS_PER_IMAGE = 1  # set to >1 if you want caption-sampling
# Health scoring config (example RDI/weights)
REF = {"Energy":2000, "Saturated fat":20, "Sugars":50, "Sodium":2300}
WEIGHTS = {"Energy":0.25, "Saturated fat":0.35, "Sugars":0.25, "Sodium":0.15}
THRESHOLDS = {"green":0.2, "amber":0.6}  # S<=0.2 green; <=0.6 amber; else red

# # === UTIL: load USDA JSON and build text list ===
# def load_usda_entries(path):
#     with open(path, "r", encoding="utf-8") as f:
#         usda = json.load(f)
#     entries = []
#     for rec in usda:
#         desc = rec.get("description") or rec.get("lowercaseDescription") or ""
#         fdcId = rec.get("fdcId", None)
#         # Normalize access to nutrients: some files use foodNutrients list
#         nutrients = {}
#         for n in rec.get("foodNutrients", []) or []:
#             # Some JSON nested differently in versions; attempt robust extraction
#             name = None
#             amt = None
#             unit = None
#             if isinstance(n, dict):
#                 # try different keys
#                 name = n.get("nutrient", {}).get("name") or n.get("name") or n.get("nutrientName")
#                 amt = n.get("amount")
#                 unit = n.get("nutrient", {}).get("unitName") or n.get("unitName")
#                 if name is None:
#                     # fallback
#                     name = n.get("name")
#             if name and amt is not None:
#                 nutrients[name] = {"amount": float(amt), "unit": unit}
#         entries.append({"fdcId": fdcId, "description": desc, "nutrients": nutrients, "raw": rec})
#     print(f"Loaded {len(entries)} USDA entries")
#     return entries

# # Build normalized SBERT indices for USDA descriptions
# def build_usda_sbert_index(entries, sbert_model):
#     texts = [e["description"] if e["description"] else "" for e in entries]
#     emb = sbert_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
#     # normalize rows
#     norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
#     emb = emb / norms
#     return emb

# # Simple health score using REF & WEIGHTS (map nutrient names heuristically)
# def compute_health_score(entry):
#     nut = entry["nutrients"]
#     # Map desired nutrient names by substring matching
#     def find_amount(key_substrs):
#         for k,v in nut.items():
#             kl = k.lower()
#             for sub in key_substrs:
#                 if sub in kl:
#                     return v.get("amount", 0.0)
#         return 0.0
#     energy = find_amount(["energy","kcal","calorie"])
#     satfat = find_amount(["saturated", "sat fat", "saturated fat"])
#     sugars = find_amount(["sugar","sugars"])
#     sodium = find_amount(["sodium","na"])
#     vals = {"Energy": energy, "Saturated fat": satfat, "Sugars": sugars, "Sodium": sodium}
#     score = 0.0
#     for k,w in WEIGHTS.items():
#         val = vals.get(k, 0.0)
#         ref = REF.get(k, 1.0)
#         score += w * (val / ref)
#     return score, vals

# def score_to_label(S):
#     if S <= THRESHOLDS["green"]:
#         return "GREEN"
#     if S <= THRESHOLDS["amber"]:
#         return "AMBER"
#     return "RED"

# === MAIN: prepare models ===
print("Device:", DEVICE)
print("Loading BLIP and SBERT models...")
processor = BlipProcessor.from_pretrained(BLIP_MODEL)
blip = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(DEVICE)
blip.eval()
sbert = SentenceTransformer(SBERT_MODEL)

# # === Load USDA entries & index ===
# usda_entries = load_usda_entries(USDA_JSON)  # :contentReference[oaicite:3]{index=3}
# usda_emb = build_usda_sbert_index(usda_entries, sbert)

# === Step: Generate captions for each image ===
image_paths = sorted([str(p) for p in Path(IMAGE_DIR).glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
print(f"Found {len(image_paths)} images in {IMAGE_DIR}")
if len(image_paths) == 0:
    raise SystemExit("No images found. Put 10-20 images in data/images_small/ and re-run.")

results = []  # will store dicts per image

for img_path in tqdm(image_paths):
    image = Image.open(img_path).convert("RGB")
    # prepare inputs
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    # generate captions (you can set num_return_sequences >1 for sampling/diversity)
    out = blip.generate(**inputs, max_length=30, num_beams=5, num_return_sequences=1, temperature=1.0)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Image: {img_path} Caption: {caption}")
    # embed caption
#     q = sbert.encode(caption, convert_to_numpy=True)
#     q = q / (np.linalg.norm(q) + 1e-12)
#     sims = usda_emb.dot(q)  # cosine since normalized
#     top_idx = np.argsort(-sims)[:TOP_K]
#     top_results = []
#     for idx in top_idx:
#         entry = usda_entries[idx]
#         sim = float(sims[idx])
#         score, nutvals = compute_health_score(entry)
#         label = score_to_label(score)
#         top_results.append({"fdcId": entry["fdcId"], "description": entry["description"],
#                             "sim": sim, "health_score": score, "health_label": label, "nutrients": nutvals})
#     results.append({"image": img_path, "caption": caption, "top_usda": top_results})

# # === Print results summary ===
# for r in results:
#     print("Image:", r["image"])
#     print(" Caption:", r["caption"])
#     for i,u in enumerate(r["top_usda"], start=1):
#         print(f"  Top{ i }: {u['description'][:80]} ... sim={u['sim']:.3f} score={u['health_score']:.3f} label={u['health_label']}")
#         print("    nutrients sample:", u["nutrients"])
#     print("-"*60)

# === Optional: create tiny calibration data interactively ===
# For calibration we need some human labels if possible: whether top-1 USDA is correct mapping.
# We will create an array of (sim, label_correct) for images you mark as correct/incorrect.
# cal_examples = []
# print("\n--- Calibration step (optional) ---")
# print("If you want to create a small calibration set, enter y to label each top-1 match as correct (1) or incorrect (0).")
# do_cal = input("Create calibration labels now? (y/n) ").strip().lower()
# if do_cal == "y":
#     for r in results:
#         top1 = r["top_usda"][0]
#         print(f"\nImage: {r['image']}\nCaption: {r['caption']}\nTop1 USDA: {top1['description']}\nSim: {top1['sim']:.4f}")
#         ans = input("Is this a correct mapping? (1=yes / 0=no / s=skip) ").strip().lower()
#         if ans in {"1","0"}:
#             cal_examples.append((top1["sim"], int(ans)))
#     if len(cal_examples) >= 10:
#         # fit Platt (logistic) on sims -> prob(correct)
#         X = np.array([s for s,_ in cal_examples]).reshape(-1,1)
#         y = np.array([l for _,l in cal_examples])
#         clf = LogisticRegression().fit(X, y)
#         print("Platt parameters:", clf.coef_, clf.intercept_)
#         # show calibrated probabilities
#         probs = clf.predict_proba(X)[:,1]
#         for (s,l),p in zip(cal_examples, probs):
#             print(f"sim={s:.3f} label={l} calibrated_p={p:.3f}")
#         # You can now use clf.predict_proba(sim.reshape(1, -1)) to get P(correct)
#     else:
#         print("Not enough calibration examples (need ~10). Save labels and collect more.")

# # Save results to disk
# out_path = "small_pipeline_results.json"
# with open(out_path, "w", encoding="utf-8") as fo:
#     json.dump(results, fo, indent=2)
# print(f"\nResults saved to {out_path}")
