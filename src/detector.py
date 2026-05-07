from transformers import pipeline
import numpy as np
import cv2
import torch
import open_clip

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
MODELS = {
    "sdxl": "Organika/sdxl-detector",
    "general": "umm-maybe/AI-image-detector",
}

pipes = {}

def load_pipelines():
    global pipes
    if pipes:
        return pipes

    for key, model_name in MODELS.items():
        try:
            pipes[key] = pipeline("image-classification", model=model_name)
        except:
            pass

    return pipes


# ─────────────────────────────────────────────
# CLIP
# ─────────────────────────────────────────────
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")


def analyze_clip(image):
    img = clip_preprocess(image).unsqueeze(0)

    text = tokenizer([
        "a real photograph",
        "an AI generated image"
    ])

    with torch.no_grad():
        image_features = clip_model.encode_image(img)
        text_features = clip_model.encode_text(text)
        probs = (image_features @ text_features.T).softmax(dim=-1)

    return probs[0][1].item()


# ─────────────────────────────────────────────
# MODEL ENSEMBLE
# ─────────────────────────────────────────────
def analyze_models(image):
    pipes = load_pipelines()
    probs = []

    # Labels used by the models
    AI_LABELS  = {"artificial", "ai", "fake", "ai-generated", "generated", "deepfake"}
    REAL_LABELS = {"human", "real", "genuine", "authentic"}

    for pipe in pipes.values():
        try:
            out = pipe(image)[0]
            label = out["label"].lower().strip()
            score = float(out["score"])

            if label in AI_LABELS:
                probs.append(score)
            elif label in REAL_LABELS:
                probs.append(1.0 - score)
            # Unknown label → skip (don't guess wrong direction)
        except:
            continue

    return np.mean(probs) if probs else 0.0


# ─────────────────────────────────────────────
# GEMINI / AI WATERMARK DETECTOR
# Gemini adds a sparkle ✦ watermark in corners.
# We detect it by looking for a bright, small,
# isolated high-contrast star/cross pattern.
# ─────────────────────────────────────────────
def detect_ai_watermark(image):
    """
    Returns a score 0-100. High score = watermark likely found.
    Checks all 4 corners for the Gemini sparkle pattern.
    """
    img = np.array(image)
    h, w = img.shape[:2]

    # Sample size for each corner (5% of image)
    ch = max(h // 12, 30)
    cw = max(w // 12, 30)

    corners = [
        img[:ch, :cw],             # top-left
        img[:ch, w-cw:],           # top-right
        img[h-ch:, :cw],           # bottom-left
        img[h-ch:, w-cw:],         # bottom-right
    ]

    for corner in corners:
        gray = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY)

        # The Gemini sparkle is bright white/near-white on a dark background
        # Look for a small isolated bright blob
        _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        bright_pixels = np.sum(bright > 0)
        total_pixels = gray.size

        # A sparkle watermark: small (1-5% of corner) but very bright
        ratio = bright_pixels / total_pixels
        if 0.005 < ratio < 0.12:
            # Check it's isolated (not just a bright uniform region)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
                if 8 < area < (total_pixels * 0.06):
                    # Small bright isolated blob = likely watermark
                    return 80

    return 0


# ─────────────────────────────────────────────
# ARTIFACT DETECTION
# ─────────────────────────────────────────────
def analyze_artifacts(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise = np.std(gray)

    score = 0

    if blur < 30:
        score += 25
    elif blur > 150:
        score += 15

    if noise < 20:
        score += 25

    if noise < 10:
        score += 20
    elif noise < 12:
        score += 5

    return min(score, 100)


# ─────────────────────────────────────────────
# SYMMETRY
# ─────────────────────────────────────────────
def analyze_symmetry(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    left = gray[:, :w//2]
    right = cv2.flip(gray[:, w//2:], 1)

    diff = np.mean(np.abs(left - right))

    if diff < 5:
        return 30
    elif diff < 10:
        return 15
    return 0


# ─────────────────────────────────────────────
# FREQUENCY ANALYSIS
# ─────────────────────────────────────────────
def analyze_frequency(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    center = magnitude[h//4:3*h//4, w//4:3*w//4]

    outer = magnitude.copy()
    outer[h//4:3*h//4, w//4:3*w//4] = 0

    center_energy = np.mean(center)
    outer_energy = np.mean(outer)

    ratio = outer_energy / (center_energy + 1e-5)

    if ratio < 0.8:
        return 30
    elif ratio < 1.0:
        return 15
    return 0


# ─────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────
def analyze_metadata(username, bio, followers, following, posts, age):
    score = 0
    flags = []

    if followers < 10 and age < 30:
        score += 20
        flags.append("Low followers + new account")

    if posts == 0:
        score += 15
        flags.append("No posts")

    if len(username) > 12 and any(char.isdigit() for char in username):
        score += 10
        flags.append("Suspicious username")

    return score, flags


# ─────────────────────────────────────────────
# FINAL ANALYSIS (V6)
# ─────────────────────────────────────────────
def full_profile_analysis(
    image_source,
    username="",
    bio="",
    followers=0,
    posts=0,
    following=0,
    account_age_days=0
):
    model_score     = analyze_models(image_source)
    clip_score      = analyze_clip(image_source)
    artifact_score  = analyze_artifacts(image_source)
    symmetry_score  = analyze_symmetry(image_source)
    frequency_score = analyze_frequency(image_source)
    watermark_score = detect_ai_watermark(image_source)   # NEW

    meta_score, flags = analyze_metadata(
        username, bio, followers, following, posts, account_age_days
    )

    # ── FINAL SCORE ──
    # If a watermark is detected it overrides everything — that's definitive.
    # Otherwise rely on model ensemble + CLIP as primary signals.
    final_score = (
        model_score     * 40 +
        clip_score      * 20 +
        artifact_score  * 0.15 +
        symmetry_score  * 0.1 +
        frequency_score * 0.2 +
        meta_score      * 0.15 +
        watermark_score * 0.5    # watermark = strong signal (up to 40 pts)
    )

    # Clamp
    final_score = max(0, min(100, final_score))

    if final_score > 60:
        verdict = "🚨 Likely AI"
        is_ai = True
    elif final_score > 42:
        verdict = "⚠️ Suspicious"
        is_ai = True
    else:
        verdict = "✅ Likely Real"
        is_ai = False

    return {
        "overall_suspicion_score": final_score,
        "overall_verdict": verdict,
        "image_analysis": {
            "ai_probability": final_score / 100,
            "is_ai_generated": is_ai,
            "confidence_level": "high" if final_score > 60 else "medium" if final_score > 42 else "low",
            "individual_results": {}
        },
        "metadata_analysis": {
            "metadata_suspicion_score": meta_score,
            "verdict": "Moderate" if meta_score > 20 else "Low",
            "red_flags": flags
        },
        "clip_score": clip_score * 100,
        "artifact_score": artifact_score,
        "symmetry_score": symmetry_score,
        "frequency_score": frequency_score,
        "watermark_score": watermark_score,
    }
