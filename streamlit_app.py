# =============================================================================
# STREAMLIT APP -- ferning detector (frozen EfficientNet-B2), for free hosting
# =============================================================================
# Repo layout on GitHub (all three files in the repo root):
#     streamlit_app.py       <- this file
#     requirements.txt       <- written by export_for_deploy.py
#     ferning_b2_head.pt     <- written by export_for_deploy.py
#     sample_gate.npz        <- written by build_sample_gate.py
#
# Design notes
#   * Memory-conscious (free tier is ~1 GB): patches are scored in small
#     batches instead of being held in memory all at once, and the overlays
#     are drawn on a downscaled DISPLAY copy of the photo. Detection itself
#     still runs at full resolution, so accuracy is unaffected.
#   * The slow part (scoring patches) is cached, so moving the threshold
#     sliders re-draws instantly without re-running the model.
#   * Phone photos carry an EXIF rotation tag; it is applied on load so
#     the photo isn't analysed sideways.
# =============================================================================

import io
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
import streamlit as st
from PIL import Image, ImageOps

PATCH_SIZE = 256
TUNED_STRIDE = 128          # stride at which the minimum patch counts were tuned
BATCH = 16
DISPLAY_MAX = 1400          # longest side of the images shown on screen
HEAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ferning_b2_head.pt")
GATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_gate.npz")

DEFAULT_STRICT, DEFAULT_LOOSE = 0.99, 0.90
DEFAULT_MIN_STRICT, DEFAULT_MIN_LOOSE = 10, 20

st.set_page_config(page_title="Ferning Detector", page_icon="🔬", layout="wide")


# =============================================================================
# 1. PREPROCESSING -- identical to the training pipeline (expects BGR)
# =============================================================================

def clean_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    background_light = cv2.GaussianBlur(gray, (51, 51), 0)
    flattened = cv2.divide(gray, background_light, scale=255)
    soft_blended = cv2.addWeighted(flattened, 0.6, gray, 0.4, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(soft_blended)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)


def is_valid_patch(patch, black_thresh=0.3):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return np.mean(gray < 10) < black_thresh


# =============================================================================
# 2. MODEL (loaded once, shared across users/reruns)
# =============================================================================

@st.cache_resource(show_spinner="Loading model (first start takes a minute)...")
def load_model():
    ck = torch.load(HEAD_PATH, map_location="cpu")
    # Frozen backbone = the public pretrained weights; only the head is ours.
    model = timm.create_model(ck["timm_name"], pretrained=True,
                               num_classes=ck["num_classes"])
    model.get_classifier().load_state_dict(ck["head_state"])
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    return model, transform, float(ck["temperature"])


@torch.inference_mode()
def predict_proba(patches_rgb):
    model, transform, temperature = load_model()
    batch = torch.stack([transform(Image.fromarray(p)) for p in patches_rgb])
    logits = model(batch).float() / temperature
    return F.softmax(logits, dim=1)[:, 1].numpy()


# =============================================================================
# 3. IMAGE LOADING + SCORING
# =============================================================================

def decode_image(data: bytes):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE:                      # too small for even one patch
        s = PATCH_SIZE / min(h, w) * 1.1
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
    return bgr


@st.cache_data(show_spinner=False, max_entries=3)
def score_image(data: bytes, stride: int):
    """Slow step: slide a window over the full-resolution photo and score every
    valid patch. Cached on (image bytes, stride) so threshold changes are free."""
    bgr = decode_image(data)
    h, w = bgr.shape[:2]

    coords = []
    for y in range(0, max(1, h - PATCH_SIZE + 1), stride):
        for x in range(0, max(1, w - PATCH_SIZE + 1), stride):
            patch = bgr[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                continue
            if is_valid_patch(patch):
                coords.append((x, y))

    probas = np.empty(len(coords), dtype=np.float32)
    for i in range(0, len(coords), BATCH):                    # small batches -> low memory
        chunk = coords[i:i + BATCH]
        patches = [clean_patch(bgr[y:y + PATCH_SIZE, x:x + PATCH_SIZE]) for x, y in chunk]
        probas[i:i + len(chunk)] = predict_proba(patches)

    return coords, probas


# =============================================================================
# 3b. SAMPLE CHECK -- only analyse images that look like the training slides
# =============================================================================
# gate_sample() / the scoring below must stay identical to build_sample_gate.py,
# which calibrated the threshold using exactly this procedure.

@st.cache_resource(show_spinner=False)
def load_gate():
    if not os.path.exists(GATE_PATH):
        return None
    g = np.load(GATE_PATH)
    return {
        "ref": g["ref"].astype(np.float32),
        "mean": g["mean"].astype(np.float32),
        "threshold": float(g["threshold"]),
        "k": int(g["k"]),
        "n_patches": int(g["n_patches"]),
        "min_valid": int(g["min_valid"]),
    }


@torch.inference_mode()
def embed(patches_rgb):
    model, transform, _ = load_model()
    batch = torch.stack([transform(Image.fromarray(p)) for p in patches_rgb])
    feats = model.forward_head(model.forward_features(batch), pre_logits=True)
    return feats.float().numpy()


def gate_sample(bgr, n_patches):
    h, w = bgr.shape[:2]
    coords = []
    for y in range(0, max(1, h - PATCH_SIZE + 1), PATCH_SIZE):
        for x in range(0, max(1, w - PATCH_SIZE + 1), PATCH_SIZE):
            patch = bgr[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                continue
            if is_valid_patch(patch):
                coords.append((x, y))
    n_valid = len(coords)
    if n_valid == 0:
        return [], 0
    idx = np.unique(np.linspace(0, n_valid - 1, min(n_patches, n_valid)).round().astype(int))
    return [clean_patch(bgr[coords[i][1]:coords[i][1] + PATCH_SIZE,
                            coords[i][0]:coords[i][0] + PATCH_SIZE]) for i in idx], n_valid


@st.cache_data(show_spinner=False, max_entries=3)
def check_sample(data: bytes):
    """Does this image look like the slide photos the model was built for?"""
    gate = load_gate()
    bgr = decode_image(data)
    patches, n_valid = gate_sample(bgr, gate["n_patches"])
    result = {"passed": False, "score": None, "n_valid": n_valid,
              "threshold": gate["threshold"]}
    if n_valid < gate["min_valid"]:
        return result
    f = embed(patches) - gate["mean"]
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    sims = f @ gate["ref"].T
    k = min(gate["k"], sims.shape[1])
    score = float(np.median(np.sort(sims, axis=1)[:, -k:].mean(axis=1)))
    result.update(score=score, passed=score >= gate["threshold"])
    return result


# =============================================================================
# 4. VOTING + OVERLAYS (cheap; re-runs whenever a slider moves)
# =============================================================================

def render(bgr, coords, probas, t_strict, t_loose, m_strict, m_loose):
    h, w = bgr.shape[:2]
    s = min(1.0, DISPLAY_MAX / max(h, w))
    disp = (cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy())
    dh, dw = disp.shape[:2]

    strict, loose = probas > t_strict, probas > t_loose
    n_strict, n_loose = int(strict.sum()), int(loose.sum())
    diagnosis = "POSITIVE" if (n_strict >= m_strict and n_loose >= m_loose) else "NEGATIVE"

    # --- boxes: thick = strict, thin = loose ---
    ls = max(1.0, min(dh, dw) / 700)
    thin, thick = max(1, round(ls)), max(2, round(ls * 2.5))
    overlay = disp.copy()
    for (x, y), is_strict, is_loose in zip(coords, strict, loose):
        if not is_loose:
            continue
        p0 = (int(x * s), int(y * s))
        p1 = (int((x + PATCH_SIZE) * s), int((y + PATCH_SIZE) * s))
        cv2.rectangle(overlay, p0, p1, (0, 255, 0) if is_strict else (0, 200, 0),
                      thick if is_strict else thin)
    cv2.putText(overlay, f"Diagnosis: {diagnosis}", (int(10 * ls), int(40 * ls)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2 * ls, (0, 0, 255), thick)

    # --- heatmap (mean probability over overlapping patches) ---
    heat = np.zeros((dh, dw), np.float32)
    count = np.zeros((dh, dw), np.float32)
    for (x, y), p in zip(coords, probas):
        y0, y1 = int(y * s), int((y + PATCH_SIZE) * s)
        x0, x1 = int(x * s), int((x + PATCH_SIZE) * s)
        heat[y0:y1, x0:x1] += p
        count[y0:y1, x0:x1] += 1
    no_data = count == 0
    count[no_data] = 1
    heat /= count
    heat_color = cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET)
    alpha = np.where(no_data, 0.0, 0.45).astype(np.float32)[..., None]
    heat_overlay = (disp.astype(np.float32) * (1 - alpha)
                    + heat_color.astype(np.float32) * alpha).astype(np.uint8)

    return (diagnosis, n_strict, n_loose,
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(heat_overlay, cv2.COLOR_BGR2RGB))


# =============================================================================
# 5. UI
# =============================================================================

st.title("Ferning Pattern Detector")
st.caption("Not a certified medical diagnostic device. Results come from a "
           "research model and shouldn't replace clinical judgment or a confirmed test.")

file = st.file_uploader("Upload a microscope image of a dried amniotic fluid "
                         "sample (JPG or PNG)", type=["jpg", "jpeg", "png"])

with st.expander("About the sample check"):
    st.write("Before analysing anything, the app checks that the image looks like the "
             "microscope slide photos the model was built for. Images that don't "
             "(random photos, screenshots, other subjects) are rejected and not analysed. "
             "The check recognises how the image *looks*; it can't confirm what fluid "
             "is actually on the slide.")

with st.expander("Advanced settings"):
    stride = st.select_slider("Stride (higher = faster, coarser)", options=[128, 160, 192, 256], value=128)
    t_strict = st.slider("Strict threshold (thick boxes)", 0.50, 0.999, DEFAULT_STRICT, 0.005)
    t_loose = st.slider("Loose threshold (thin boxes)", 0.50, 0.999, DEFAULT_LOOSE, 0.005)
    m_strict = st.slider("Min patches above strict threshold", 1, 40, DEFAULT_MIN_STRICT)
    m_loose = st.slider("Min patches above loose threshold", 1, 60, DEFAULT_MIN_LOOSE)
    st.caption("The minimum counts were tuned at stride 128 and are scaled "
               "automatically if you change the stride.")

if file is None:
    st.info("Upload an image to begin.")
    st.stop()

if t_loose > t_strict:
    st.error("The loose threshold can't be higher than the strict one.")
    st.stop()

data = file.getvalue()

# --- sample check: the ferning model only runs on images that pass ---
if load_gate() is None:
    st.error("The sample-check file (sample_gate.npz) is missing from this app, "
             "so analysis is disabled.")
    st.stop()

with st.spinner("Checking the image..."):
    check = check_sample(data)

if not check["passed"]:
    st.error("This doesn't look like a microscope image of an amniotic fluid sample, "
             "so no analysis was run. Please upload a photo of the dried sample on a "
             "slide, taken through the microscope.")
    if check["score"] is None:
        st.caption(f"Reason: too little usable image area (found {check['n_valid']} regions).")
    else:
        st.caption(f"Match score {check['score']:.2f}; at least {check['threshold']:.2f} is required.")
    st.image(data, caption="Uploaded image", width=300)
    st.stop()

st.caption(f"Sample check passed (match score {check['score']:.2f}, "
           f"required {check['threshold']:.2f}).")

with st.spinner("Scanning the photo... this can take a minute on the free tier."):
    coords, probas = score_image(data, stride)

if len(coords) == 0:
    st.warning("No usable area found in this photo (too dark, or smaller than one patch).")
    st.stop()

count_scale = (TUNED_STRIDE / stride) ** 2
m_strict_eff = max(1, round(m_strict * count_scale))
m_loose_eff = max(1, round(m_loose * count_scale))

bgr = decode_image(data)
diagnosis, n_strict, n_loose, overlay_rgb, heat_rgb = render(
    bgr, coords, probas, t_strict, t_loose, m_strict_eff, m_loose_eff)

st.subheader(f"Diagnosis: {diagnosis}")
st.write(f"Patches evaluated: **{len(coords)}**  \n"
         f"Patches above {t_strict}: **{n_strict}** (need at least {m_strict_eff})  \n"
         f"Patches above {t_loose}: **{n_loose}** (need at least {m_loose_eff})")

c1, c2 = st.columns(2)
with c1:
    st.image(overlay_rgb, caption="Detected regions (thick = strict, thin = loose)")
with c2:
    st.image(heat_rgb, caption="Probability heatmap")

st.caption("Photos are processed on the hosting provider's servers.")
