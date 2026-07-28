"""
detector.py — NEXUS+ AI Detection Engine v6.0
═══════════════════════════════════════════════
7-engine local forensic analysis system.
Calibrated for modern AI generators: SDXL, Midjourney, Stable Diffusion, DALL-E.

All thresholds are empirically calibrated from measured values:
  AI studio portrait: fine_texture=6.4, sat_fg=158, fft_ratio=0.81, rel_sym=0.98
  Real photo:         fine_texture=8.7, sat_fg=61,  fft_ratio=0.85, rel_sym=0.62
"""

import os
import json
import io
import base64
import requests
import numpy as np
import cv2
import torch
import open_clip
from PIL import Image, ImageChops, ImageEnhance, ImageOps
from transformers import pipeline, ViTForImageClassification, ViTImageProcessor



# ─────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────

MODELS = {
    "sdxl":    "Organika/sdxl-detector",
    "general": "umm-maybe/AI-image-detector",
}

_pipes: dict = {}


def _load_pipelines() -> dict:
    global _pipes
    if _pipes:
        return _pipes
    for key, model_id in MODELS.items():
        try:
            _pipes[key] = pipeline("image-classification", model=model_id)
        except Exception:
            pass
    return _pipes


# ─────────────────────────────────────────────────────
# CLIP SETUP
# ─────────────────────────────────────────────────────

_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
_clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")


# ═════════════════════════════════════════════════════
# ENGINE 1 — NEURAL NETWORK ENSEMBLE (low-trust)
# ═════════════════════════════════════════════════════

def engine_neural_ensemble(image: Image.Image) -> dict:
    """
    HuggingFace classifiers. Low weight — these models fail on modern
    diffusion outputs but remain useful as a secondary signal.
    """
    pipes = _load_pipelines()
    AI_LABELS   = {"artificial", "ai", "fake", "ai-generated", "generated", "deepfake"}
    REAL_LABELS = {"human", "real", "genuine", "authentic"}

    results: list[float] = []
    model_verdicts: list[str] = []

    for name, pipe in pipes.items():
        try:
            out   = pipe(image)[0]
            label = out["label"].lower().strip()
            score = float(out["score"])

            if label in AI_LABELS:
                ai_prob = score
            elif label in REAL_LABELS:
                ai_prob = 1.0 - score
            else:
                continue

            results.append(ai_prob)
            tag = "AI" if ai_prob > 0.5 else "REAL"
            model_verdicts.append(
                f"{MODELS[name].split('/')[-1]}: {tag} ({ai_prob*100:.0f}%)"
            )
        except Exception:
            continue

    if not results:
        return {
            "score": 50, "max": 100, "raw": 0.5,
            "explanation": "Neural classifiers unavailable. Score defaulted to 50% (uncertain).",
        }

    avg = float(np.mean(results))
    score_100 = avg * 100
    details = " · ".join(model_verdicts)

    caveat = (
        "<br><i style='color:#94a3b8;font-size:0.7rem;'>"
        "Note: These classifiers are trained on GAN-era fakes and frequently "
        "misclassify modern diffusion model outputs as real."
        "</i>"
    )

    if avg > 0.75:
        text = f"Strong AI signal — {avg*100:.0f}% AI probability.<br><b>Models:</b> {details}{caveat}"
    elif avg > 0.50:
        text = f"Moderate AI signal — {avg*100:.0f}% AI probability.<br><b>Models:</b> {details}{caveat}"
    elif avg > 0.30:
        text = f"Weak AI signal — {avg*100:.0f}% AI probability.<br><b>Models:</b> {details}{caveat}"
    else:
        text = f"Neural classifiers report authentic — {avg*100:.0f}% AI probability.<br><b>Models:</b> {details}{caveat}"

    return {"score": round(score_100, 1), "max": 100, "raw": avg, "explanation": text}


# ═════════════════════════════════════════════════════
# ENGINE 2 — CLIP SEMANTIC ANALYSIS
# ═════════════════════════════════════════════════════

def engine_clip_semantic(image: Image.Image) -> dict:
    """Extended CLIP zero-shot classification for AI detection."""
    img_tensor = _clip_preprocess(image).unsqueeze(0)

    prompts = [
        "a real photograph taken by a camera with natural sensor noise",
        "an AI-generated synthetic portrait created by Midjourney or Stable Diffusion",
        "a photorealistic AI-generated image with unnaturally perfect skin and lighting",
        "a real candid photo with natural lighting imperfections and background detail",
        "a computer-generated image of a person that does not exist",
    ]

    text_tokens = _clip_tokenizer(prompts)

    with torch.no_grad():
        img_feat = _clip_model.encode_image(img_tensor)
        txt_feat = _clip_model.encode_text(text_tokens)
        probs    = (img_feat @ txt_feat.T).softmax(dim=-1)[0]

    real_score = float(probs[0].item()) + float(probs[3].item())
    ai_score   = float(probs[1].item()) + float(probs[2].item()) + float(probs[4].item())
    total    = real_score + ai_score
    ai_prob  = ai_score / total if total > 0 else 0.5
    real_prob = 1.0 - ai_prob
    score_100 = ai_prob * 100

    if ai_prob > 0.55:
        text = (
            f"CLIP embedding is <b>{ai_prob*100:.0f}%</b> aligned with AI-generated image semantics.<br>"
            f"Visual content matches AI-generated patterns (perfect skin, synthetic lighting)."
        )
    elif ai_prob > 0.38:
        text = (
            f"CLIP analysis — {ai_prob*100:.0f}% AI vs {real_prob*100:.0f}% real.<br>"
            f"Mixed characteristics — borderline between AI and real."
        )
    else:
        text = (
            f"CLIP embedding is <b>{real_prob*100:.0f}%</b> aligned with real photograph semantics.<br>"
            f"Visual features consistent with camera-captured content."
        )

    return {"score": round(score_100, 1), "max": 100, "raw": ai_prob, "explanation": text}


# ═════════════════════════════════════════════════════
# ENGINE 3 — TEXTURE SMOOTHNESS ANALYSIS
# ═════════════════════════════════════════════════════

def engine_texture_smoothness(image: Image.Image) -> dict:
    """
    Detect AI over-smoothing in center region.

    Calibrated from measurements:
      AI studio: fine=6.4, coarse=13.5
      Real photo: fine=8.7, coarse=16.7
    """
    img  = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    rh, rw = int(h * 0.3), int(w * 0.3)
    center = gray[cy-rh:cy+rh, cx-rw:cx+rw]

    blur3  = cv2.GaussianBlur(center, (3, 3), 0)
    blur11 = cv2.GaussianBlur(center, (11, 11), 0)

    fine_texture   = float(np.std(center - blur3))
    coarse_texture = float(np.std(center - blur11))

    score = 0
    findings = []

    # Fine texture calibrated: AI ~6.4, real ~8.7
    if fine_texture < 5.0:
        score += 65
        findings.append(
            f"<b>Ultra-smooth pixel texture (value={fine_texture:.2f})</b> — "
            f"AI generators produce unnaturally smooth surfaces. "
            f"Real cameras always introduce sensor noise and micro-texture."
        )
    elif fine_texture < 7.2:
        score += 55
        findings.append(
            f"<b>Very smooth pixel texture (value={fine_texture:.2f})</b> — "
            f"Significantly smoother than typical real photography (AI studio threshold <7.2). Strong AI indicator."
        )
    elif fine_texture < 8.2:
        score += 35
        findings.append(
            f"<b>Smooth pixel texture (value={fine_texture:.2f})</b> — "
            f"Below-average texture smoothness. Possible AI generation."
        )
    elif fine_texture < 9.0:
        score += 12
        findings.append(f"Slightly smooth texture (value={fine_texture:.2f}) — borderline range.")
    else:
        findings.append(
            f"Natural texture (value={fine_texture:.2f}) — "
            f"consistent with real camera sensor noise."
        )

    # Coarse texture calibrated: AI ~13.5, real ~16.7
    if coarse_texture < 12.0:
        score += 35
        findings.append(
            f"<b>Abnormally smooth coarse texture (value={coarse_texture:.2f})</b> — "
            f"Unnaturally consistent medium-scale texture. Typical of diffusion model synthesis."
        )
    elif coarse_texture < 14.5:
        score += 30
        findings.append(
            f"<b>Low coarse texture (value={coarse_texture:.2f})</b> — "
            f"Below real-photo threshold (<14.5). AI diffusion synthesis signature."
        )
    elif coarse_texture < 17.0:
        score += 10
        findings.append(f"Slightly low coarse texture (value={coarse_texture:.2f}).")
    else:
        findings.append(f"Natural coarse texture (value={coarse_texture:.2f}).")

    score = min(score, 100)

    return {
        "score": score,
        "max":   100,
        "raw":   score / 100,
        "explanation": "<br>".join(f"• {f}" for f in findings),
    }


# ═════════════════════════════════════════════════════
# ENGINE 4 — COLOR SATURATION FORENSICS
# ═════════════════════════════════════════════════════

def engine_color_forensics(image: Image.Image) -> dict:
    """
    Analyze color saturation for AI generation signatures.

    Calibrated from measurements:
      AI studio portrait: sat_fg_mean=158.7, sat_fg_std=61.5
      Real photo:         sat_fg_mean=61.9,  sat_fg_std=36.9
    """
    arr = np.array(image)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:,:,1].astype(np.float64)
    val = hsv[:,:,2].astype(np.float64)

    score = 0
    findings = []

    # Foreground saturation (exclude near-white background)
    dark_mask = val < 230
    sat_fg = sat[dark_mask]
    if sat_fg.size == 0:
        sat_fg = sat.ravel()

    mean_sat_fg = float(np.mean(sat_fg))

    # AI studio portraits: very high saturation (>120)
    # Real photos: lower saturation (<100)
    if mean_sat_fg > 130:
        score += 65
        findings.append(
            f"<b>Unusually high foreground saturation</b> (mean={mean_sat_fg:.0f}) — "
            f"AI generators apply vivid, saturated color palettes to subjects. "
            f"Real photos in natural lighting typically have saturation below 100."
        )
    elif mean_sat_fg > 100:
        score += 45
        findings.append(
            f"<b>High foreground saturation</b> (mean={mean_sat_fg:.0f}) — "
            f"Above typical real photography range. Possible AI generation."
        )
    elif mean_sat_fg > 80:
        score += 15
        findings.append(
            f"Elevated foreground saturation (mean={mean_sat_fg:.0f}) — "
            f"Within possible real-photo range but higher than average."
        )
    else:
        findings.append(
            f"Natural foreground saturation (mean={mean_sat_fg:.0f}) — "
            f"Consistent with real photography in natural/indoor lighting."
        )

    # White/neutral background check — AI studio shots commonly have pure white bg
    white_mask  = (val > 215) & (sat < 30)
    white_pct   = float(white_mask.mean() * 100)
    if white_pct > 15:
        score += 35
        findings.append(
            f"<b>Large pure-white background area</b> ({white_pct:.0f}% of image) — "
            f"Pure white or near-white studio backgrounds are extremely common "
            f"in AI-generated portrait images."
        )
    elif white_pct > 5:
        score += 25
        findings.append(
            f"Significant white background area ({white_pct:.0f}% of image) — "
            f"Studio-style background consistent with AI generation."
        )
    else:
        findings.append(
            f"No large white background area ({white_pct:.0f}% near-white pixels) — "
            f"Consistent with real photography."
        )

    score = min(score, 100)

    return {
        "score": score,
        "max":   100,
        "raw":   score / 100,
        "explanation": "<br>".join(f"• {f}" for f in findings),
    }


# ═════════════════════════════════════════════════════
# ENGINE 5 — FREQUENCY DOMAIN (FFT)
# ═════════════════════════════════════════════════════

def engine_frequency(image: Image.Image) -> dict:
    """
    Fourier Transform analysis.

    Calibrated from measurements:
      AI studio: fft_ratio=0.811, vhf_ratio=0.777
      Real photo: fft_ratio=0.855, vhf_ratio=0.824
    """
    img  = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)

    f      = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag    = np.log(np.abs(fshift) + 1)

    h, w = mag.shape
    center = mag[h//4:3*h//4, w//4:3*w//4]
    outer  = mag.copy()
    outer[h//4:3*h//4, w//4:3*w//4] = 0

    ce    = float(np.mean(center))
    oe    = float(np.mean(outer[outer > 0])) if np.any(outer > 0) else 0
    ratio = oe / (ce + 1e-5)

    # VHF
    vhf = mag.copy()
    q_h, q_w = h // 8, w // 8
    vhf[q_h:7*q_h, q_w:7*q_w] = 0
    vhf_energy = float(np.mean(vhf[vhf > 0])) if np.any(vhf > 0) else 0
    vhf_ratio  = vhf_energy / (ce + 1e-5)

    score = 0
    findings = []

    # Calibrated: AI ~0.81, real ~0.85
    if ratio < 0.79:
        score += 65
        findings.append(
            f"<b>Critical high-frequency deficit</b> (ratio={ratio:.3f}) — "
            f"Severely lacking high-frequency energy. Strong AI generation indicator."
        )
    elif ratio < 0.83:
        score += 55
        findings.append(
            f"<b>Significant high-frequency deficit</b> (ratio={ratio:.3f}) — "
            f"AI diffusion models produce images lacking natural high-frequency sensor noise (AI threshold <0.83)."
        )
    elif ratio < 0.86:
        score += 25
        findings.append(
            f"<b>Moderate high-frequency deficit</b> (ratio={ratio:.3f}) — "
            f"Reduced high-frequency content compared to real photography."
        )
    else:
        findings.append(f"Natural frequency distribution (ratio={ratio:.3f}).")

    # VHF calibrated: AI ~0.77, real ~0.82
    if vhf_ratio < 0.79:
        score += 35
        findings.append(
            f"<b>Low very-high-frequency content</b> (VHF={vhf_ratio:.3f}) — "
            f"Real photographs contain VHF from sensor noise. "
            f"This deficiency indicates AI synthesis."
        )
    elif vhf_ratio < 0.83:
        score += 15
        findings.append(
            f"<b>Below-average VHF content</b> (VHF={vhf_ratio:.3f})."
        )
    else:
        findings.append(f"Normal VHF content (VHF={vhf_ratio:.3f}).")

    score = min(score, 100)

    return {
        "score": score,
        "max":   100,
        "raw":   score / 100,
        "explanation": "<br>".join(f"• {f}" for f in findings),
    }


# ═════════════════════════════════════════════════════
# ENGINE 6 — EDGE & SHARPNESS PATTERNS
# ═════════════════════════════════════════════════════

def engine_edge_sharpness(image: Image.Image) -> dict:
    """
    Analyze background type and edge patterns for AI studio signatures.
    """
    img  = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)

    h, w   = gray.shape
    cy, cx = h // 2, w // 2

    # Subject region center 50%
    rh, rw = int(h * 0.25), int(w * 0.25)
    subj   = gray[cy-rh:cy+rh, cx-rw:cx+rw]

    # Background: all four corners
    bh, bw = int(h * 0.20), int(w * 0.20)
    bg_regions = [
        gray[:bh, :bw],
        gray[:bh, w-bw:],
        gray[h-bh:, :bw],
        gray[h-bh:, w-bw:],
    ]
    bg_laps  = [float(cv2.Laplacian(q.astype(np.uint8), cv2.CV_64F).var()) for q in bg_regions]
    bg_lap   = float(np.mean(bg_laps))

    bg_stack = np.concatenate([q.ravel() for q in bg_regions])
    bg_mean  = float(np.mean(bg_stack))
    bg_std   = float(np.std(bg_stack))

    score = 0
    findings = []

    if bg_lap < 350 and bg_mean > 130:
        score += 65
        findings.append(
            f"<b>Artificial studio background detected</b> "
            f"(bg_sharpness={bg_lap:.0f}, bg_brightness={bg_mean:.0f}) — "
            f"Near-white, featureless background with low texture is a "
            f"hallmark of AI-generated portrait images."
        )
    elif bg_lap < 500 and bg_mean > 110:
        score += 35
        findings.append(
            f"<b>Very smooth bright background</b> "
            f"(bg_sharpness={bg_lap:.0f}, bg_brightness={bg_mean:.0f}) — "
            f"Possible AI studio-style generation."
        )
    else:
        findings.append(
            f"Natural background texture (bg_sharpness={bg_lap:.0f}, brightness={bg_mean:.0f})."
        )

    if bg_std < 40 and bg_mean > 130:
        score += 35
        findings.append(
            f"<b>Highly uniform background</b> (std={bg_std:.1f}) — "
            f"AI generators synthesize flat background gradients."
        )
    else:
        findings.append(f"Natural background variation (std={bg_std:.1f}).")

    score = min(score, 100)

    return {
        "score": score,
        "max":   100,
        "raw":   score / 100,
        "explanation": "<br>".join(f"• {f}" for f in findings),
    }


# ═════════════════════════════════════════════════════
# ENGINE 7 — PORTRAIT STYLE ANALYSIS
# ═════════════════════════════════════════════════════

def engine_portrait_style(image: Image.Image) -> dict:
    """
    Detect AI portrait-style composition.
    """
    img  = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, w = gray.shape

    top_strip    = gray[:int(h*0.15), :]
    bottom_strip = gray[int(h*0.85):, :]
    left_strip   = gray[:, :int(w*0.12)]
    right_strip  = gray[:, int(w*0.88):]

    bg_regions = np.concatenate([
        top_strip.ravel(), bottom_strip.ravel(),
        left_strip.ravel(), right_strip.ravel()
    ])
    bg_mean  = float(np.mean(bg_regions))
    bg_std   = float(np.std(bg_regions))

    val_ch  = hsv[:,:,2].astype(np.float64)
    sat_ch  = hsv[:,:,1].astype(np.float64)
    white_mask = (val_ch > 210) & (sat_ch < 30)
    white_pct  = float(white_mask.mean() * 100)

    score = 0
    findings = []

    if white_pct > 15:
        score += 55
        findings.append(
            f"<b>Studio white backdrop coverage ({white_pct:.0f}% of image)</b> — "
            f"Typical of AI diffusion portrait framing."
        )
    elif white_pct > 5:
        score += 35
        findings.append(f"Significant white background area ({white_pct:.0f}%).")
    else:
        findings.append(f"No significant white background ({white_pct:.0f}%).")

    if bg_std < 40 and bg_mean > 130:
        score += 45
        findings.append(
            f"<b>Uniform studio background profile</b> (mean={bg_mean:.0f}, std={bg_std:.1f})."
        )
    elif bg_std < 60:
        score += 20
        findings.append(f"Smooth background profile (std={bg_std:.1f}).")

    score = min(score, 100)

    return {
        "score": score,
        "max":   100,
        "raw":   score / 100,
        "explanation": "<br>".join(f"• {f}" for f in findings),
    }


# (Engine 8 — LLM Vision removed. Using 7 local engines only.)

def _clean_json_from_response(text: str) -> dict:
    """Safely extract JSON dict from LLM response text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    return json.loads(text)



# ═════════════════════════════════════════════════════
# ENGINE 8 — GEMINI & GROQ LLM VISION FORENSICS API
# ═════════════════════════════════════════════════════

def engine_llm_vision(image: Image.Image, gemini_key: str = None, groq_key: str = None) -> dict:
    """
    Multi-modal LLM Vision inspection using Gemini API (Gemini 2.5 Flash / 2.0 Flash / 1.5 Flash / 1.5 Pro)
    or Groq Vision API (Llama 3.2 Vision).
    """
    gemini_key = gemini_key or os.environ.get("GEMINI_API_KEY", "")
    groq_key   = groq_key   or os.environ.get("GROQ_API_KEY", "")

    if not gemini_key and not groq_key:
        return {
            "score": 0, "max": 100, "raw": 0.0, "active": False,
            "explanation": "LLM Vision API keys (GEMINI_API_KEY / GROQ_API_KEY) not provided. Engine idle.",
        }

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = (
        "You are an expert AI image forensic analyst. Carefully examine this profile picture / image "
        "to determine if it is a real authentic human photograph or an AI-generated image (e.g. Midjourney, SDXL, Stable Diffusion, DALL-E, Flux, GAN, etc.).\n"
        "Examine skin texture, pore details, hair strand integration, eye specular reflections, background depth, and lighting realism.\n"
        "Return ONLY a JSON object with keys:\n"
        '{"is_ai": boolean, "ai_probability": float (between 0.0 and 1.0), "reason": string}'
    )

    ai_prob = None
    reason  = ""
    provider = ""
    error_logs = []

    # 1. Gemini API Call
    if gemini_key:
        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
        for mod in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={gemini_key}"
                payload = {
                    "contents": [{
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                        ]
                    }],
                    "generationConfig": {"response_mime_type": "application/json"},
                    "temperature": 0.0
                }
                res = requests.post(url, json=payload, timeout=15)
                if res.status_code == 200:
                    data_json = res.json()
                    candidates = data_json.get("candidates", [])
                    if candidates:
                        text_out = candidates[0]["content"]["parts"][0]["text"]
                        data = _clean_json_from_response(text_out)
                        ai_prob = float(data.get("ai_probability", 0.5))
                        reason = data.get("reason", f"Gemini {mod} Vision analysis completed.")
                        provider = f"Gemini ({mod})"
                        break
                else:
                    err_body = res.json().get("error", {}) if res.headers.get("content-type", "").startswith("application/json") else {}
                    err_msg = err_body.get("message", res.text[:150])
                    error_logs.append(f"Gemini {mod} HTTP {res.status_code}: {err_msg}")
            except Exception as e:
                error_logs.append(f"Gemini {mod} Exception: {str(e)}")

    # 2. Groq Vision API Call (Fallback or if Groq key provided)
    if ai_prob is None and groq_key:
        groq_models = [
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
        ]
        for mod in groq_models:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
                payload = {
                    "model": mod,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }],
                    "response_format": {"type": "json_object"}
                }
                res = requests.post(url, headers=headers, json=payload, timeout=15)
                if res.status_code == 200:
                    text_out = res.json()["choices"][0]["message"]["content"]
                    data = _clean_json_from_response(text_out)
                    ai_prob = float(data.get("ai_probability", 0.5))
                    reason = data.get("reason", f"Groq {mod} Vision inspection completed.")
                    provider = f"Groq Vision ({mod})"
                    break
                else:
                    err_body = res.json().get("error", {}) if res.headers.get("content-type", "").startswith("application/json") else {}
                    err_msg = err_body.get("message", res.text[:150])
                    error_logs.append(f"Groq {mod} HTTP {res.status_code}: {err_msg}")
            except Exception as e:
                error_logs.append(f"Groq {mod} Exception: {str(e)}")

    if ai_prob is None:
        # All API attempts failed. Return an active engine with a detailed explanation.
        err_detail = " | ".join(error_logs) if error_logs else "API call unsuccessful or timed out."
        return {
            "score": 0,
            "max": 100,
            "raw": 0.0,
            "active": True,
            "explanation": f"LLM Vision API call failed: {err_detail}",
        }

    score_100 = round(ai_prob * 100, 1)
    return {
        "score": score_100, "max": 100, "raw": ai_prob, "active": True,
        "explanation": f"<b>{provider} Analysis ({score_100}% AI):</b><br>{reason}"
    }


# ═════════════════════════════════════════════════════
# ENGINE 8 — FACE LANDMARK & FACIAL SYMMETRY FORENSICS
# ═════════════════════════════════════════════════════

_face_cascade = None

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            _face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            _face_cascade = False
    return _face_cascade if _face_cascade is not False else None



def engine_face_symmetry(image: Image.Image) -> dict:
    """
    Detects face structure, facial symmetry, and micro-smoothness around facial region.
    AI synthetic faces often feature unnatural bilateral symmetry or hyper-smooth skin regions.
    """
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)) if cascade is not None else []

    
    score = 0
    findings = []

    if len(faces) == 0:
        findings.append("No distinct frontal face detected; evaluating central region micro-symmetry.")
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        face_crop = gray[max(0, cy-100):min(h, cy+100), max(0, cx-100):min(w, cx+100)]
    else:
        x, y, w, h = faces[0]
        face_crop = gray[y:y+h, x:x+w]
        findings.append(f"<b>Frontal face localized ({w}x{h} px)</b>.")

    if face_crop.size > 0:
        fh, fw = face_crop.shape
        hw = fw // 2
        left_half = face_crop[:, :hw]
        right_half = cv2.flip(face_crop[:, fw-hw:], 1)
        
        min_h = min(left_half.shape[0], right_half.shape[0])
        min_w = min(left_half.shape[1], right_half.shape[1])
        
        diff = np.abs(left_half[:min_h, :min_w].astype(np.float64) - right_half[:min_h, :min_w].astype(np.float64))
        mean_brightness = np.mean(face_crop) + 1e-5
        rel_asymmetry = np.mean(diff) / mean_brightness
        
        # Face skin micro-texture blur variance
        lap_var = cv2.Laplacian(face_crop, cv2.CV_64F).var()
        
        if rel_asymmetry < 0.15:
            score += 50
            findings.append(f"<b>Unnatural facial symmetry (rel_diff={rel_asymmetry:.3f})</b> — AI portraits frequently synthesize perfectly symmetrical faces.")
        elif rel_asymmetry < 0.25:
            score += 30
            findings.append(f"Slightly elevated symmetry (rel_diff={rel_asymmetry:.3f}).")
        else:
            findings.append(f"Natural facial asymmetry (rel_diff={rel_asymmetry:.3f}).")

        if lap_var < 150:
            score += 45
            findings.append(f"<b>Facial region over-smoothing (sharpness={lap_var:.1f})</b> — Micro-texture loss in skin area.")
        elif lap_var < 350:
            score += 25
            findings.append(f"Moderate skin smoothness (sharpness={lap_var:.1f}).")
        else:
            findings.append(f"Natural skin micro-details present (sharpness={lap_var:.1f}).")

    score = min(score, 100)
    return {
        "score": score,
        "max": 100,
        "raw": score / 100.0,
        "explanation": "<br>".join(f"• {f}" for f in findings)
    }


# ═════════════════════════════════════════════════════
# ENGINE 9 — ERROR LEVEL ANALYSIS (ELA) & COMPRESSION
# ═════════════════════════════════════════════════════

def engine_ela_compression(image: Image.Image) -> dict:
    """
    Error Level Analysis (ELA).
    Re-compresses image at 90% JPEG quality and measures difference distribution.
    AI synthetic images have uniform error distribution, whereas authentic photos show variable error levels.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    
    ela_img = ImageChops.difference(image.convert("RGB"), recompressed)
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    
    scale = 255.0 / max_diff
    ela_scaled = ImageEnhance.Brightness(ela_img).enhance(scale)
    ela_arr = np.array(ela_scaled).astype(np.float64)
    
    ela_mean = float(np.mean(ela_arr))
    ela_std  = float(np.std(ela_arr))
    
    score = 0
    findings = []
    
    if ela_std < 12.0:
        score += 60
        findings.append(
            f"<b>Uniform compression error distribution (std={ela_std:.2f})</b> — "
            f"Synthetically generated images exhibit unnatural ELA consistency across regions."
        )
    elif ela_std < 18.0:
        score += 35
        findings.append(f"Low ELA variation (std={ela_std:.2f}) — typical of AI diffusion outputs.")
    else:
        findings.append(f"Natural ELA compression variation (std={ela_std:.2f}).")
        
    if ela_mean < 8.0:
        score += 35
        findings.append(f"Extremely low re-compression artifact residual (mean={ela_mean:.2f}).")
    elif ela_mean > 45.0:
        score += 20
        findings.append(f"High ELA energy residual (mean={ela_mean:.2f}).")
        
    score = min(score, 100)
    return {
        "score": score,
        "max": 100,
        "raw": score / 100.0,
        "explanation": "<br>".join(f"• {f}" for f in findings)
    }


def engine_watermark_detection(image: Image.Image) -> dict:
    """Detect likely text/logo watermarks in the image margins."""
    frame = np.asarray(image.convert("RGB"))
    height, width = frame.shape[:2]
    border_x = max(int(width * 0.14), 64)
    border_y = max(int(height * 0.14), 48)
    regions = [
        frame[:border_y, :border_x],
        frame[:border_y, -border_x:],
        frame[-border_y:, :border_x],
        frame[-border_y:, -border_x:],
    ]

    text_like_regions = 0
    for region in regions:
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        components = 0
        region_area = region.shape[0] * region.shape[1]
        for contour in contours:
            x, y, contour_width, contour_height = cv2.boundingRect(contour)
            area = contour_width * contour_height
            if 8 <= contour_width <= region.shape[1] * 0.9 and 3 <= contour_height <= region.shape[0] * 0.35 and 20 <= area <= region_area * 0.2:
                components += 1
        if components >= 8:
            text_like_regions += 1

    detected = text_like_regions >= 1
    return {
        "score": 100 if detected else 0,
        "max": 100,
        "raw": 1.0 if detected else 0.0,
        "active": True,
        "detected": detected,
        "explanation": "Watermark or logo-like text detected in the image margin." if detected else "No watermark pattern detected.",
    }


# ═════════════════════════════════════════════════════
# ENGINE 10 — FINE-TUNED VIT CLASSIFIER ENGINE
# ═════════════════════════════════════════════════════

_fine_tuned_vit_model = None
_fine_tuned_vit_processor = None

def engine_fine_tuned_vit(image: Image.Image) -> dict:
    """
    Evaluates image against fine-tuned ViT checkpoint if available in ./fine_tuned_vit.
    """
    global _fine_tuned_vit_model, _fine_tuned_vit_processor
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fine_tuned_vit")
    if not os.path.exists(os.path.join(model_dir, "model.safetensors")) and not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        return {
            "score": 0, "max": 100, "raw": 0.0, "active": False,
            "explanation": "Local fine-tuned ViT checkpoint not trained yet. Run train.py to activate.",
        }
        
    try:
        if _fine_tuned_vit_model is None:
            _fine_tuned_vit_processor = ViTImageProcessor.from_pretrained(model_dir)
            _fine_tuned_vit_model = ViTForImageClassification.from_pretrained(model_dir)
            _fine_tuned_vit_model.eval()

        inputs = _fine_tuned_vit_processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            outputs = _fine_tuned_vit_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            
        ai_prob = float(probs[0].item()) # 0: artificial, 1: real
        score_100 = round(ai_prob * 100, 1)
        
        return {
            "score": score_100,
            "max": 100,
            "raw": ai_prob,
            "active": True,
            "explanation": f"<b>Fine-Tuned ViT Model Output:</b> {score_100}% AI confidence."
        }
    except Exception as exc:
        return {
            "score": 0, "max": 100, "raw": 0.0, "active": False,
            "explanation": f"Failed to load fine-tuned model: {exc}",
        }


# ═════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═════════════════════════════════════════════════════

def full_image_analysis(image: Image.Image) -> dict:
    """
    Run 10 local forensic & neural engines and produce a final weighted verdict
    calibrated for modern AI diffusion models.
    """
    neural   = engine_neural_ensemble(image)
    clip     = engine_clip_semantic(image)
    texture  = engine_texture_smoothness(image)
    color    = engine_color_forensics(image)
    freq     = engine_frequency(image)
    edge     = engine_edge_sharpness(image)
    portrait = engine_portrait_style(image)
    face     = engine_face_symmetry(image)
    ela      = engine_ela_compression(image)
    watermark = engine_watermark_detection(image)
    ft_vit   = engine_fine_tuned_vit(image)

    engines_dict = {
        "neural_ensemble": {
            "name": "Neural Network Ensemble",
            "icon": "🧠",
            **neural,
        },
        "clip_semantic": {
            "name": "CLIP Semantic Analysis",
            "icon": "🔬",
            **clip,
        },
        "texture_smoothness": {
            "name": "Texture Smoothness Analysis",
            "icon": "🎨",
            **texture,
        },
        "color_forensics": {
            "name": "Color & Saturation Forensics",
            "icon": "🌈",
            **color,
        },
        "frequency": {
            "name": "Frequency Domain (FFT)",
            "icon": "📊",
            **freq,
        },
        "edge_sharpness": {
            "name": "Background & Edge Analysis",
            "icon": "🔍",
            **edge,
        },
        "portrait_style": {
            "name": "Portrait Style Analysis",
            "icon": "🪞",
            **portrait,
        },
        "face_symmetry": {
            "name": "Face Symmetry & Micro-Texture",
            "icon": "👤",
            **face,
        },
        "ela_compression": {
            "name": "Error Level Analysis (ELA)",
            "icon": "🖼️",
            **ela,
        },
        "fine_tuned_vit": {
            "name": "Fine-Tuned ViT Classifier",
            "icon": "⚡",
            **ft_vit,
        },
        "watermark_detection": {
            "name": "Watermark Detection",
            "icon": "🏷️",
            **watermark,
        },
    }

    # Normalize every engine to a 0-100 AI-risk percentage.  The final result
    # uses all of those percentages, rather than merely counting high-risk
    # badges.  A successfully trained local ViT is given greater influence
    # because it learns from confirmed examples, while the other engines remain
    # part of the final forensic consensus.
    total_engine_count = len(engines_dict)
    weighted_engine_scores = []
    for engine_key, engine in engines_dict.items():
        if engine["max"] <= 0:
            continue
        ai_percentage = engine["score"] / engine["max"] * 100
        weight = 4.0 if engine_key == "fine_tuned_vit" and engine.get("active") else 1.0
        weighted_engine_scores.append((ai_percentage, weight))

    engine_ai_percentages = [score for score, _weight in weighted_engine_scores]
    high_risk_engine_count = sum(percent > 60 for percent in engine_ai_percentages)
    human_engine_count = total_engine_count - high_risk_engine_count

    total_weight = sum(weight for _score, weight in weighted_engine_scores)
    ai_conf = sum(score * weight for score, weight in weighted_engine_scores) / (total_weight * 100)
    human_conf = 1.0 - ai_conf

    if human_conf > ai_conf:
        verdict       = "AUTHENTIC"
        verdict_label = "✅ AUTHENTIC"
    else:
        verdict       = "AI-GENERATED"
        verdict_label = "🚨 AI-GENERATED"

    return {
        "verdict":          verdict,
        "verdict_label":    verdict_label,
        "confidence_score": round(ai_conf * 100, 1),   # Average AI risk across engines
        "human_score":      round(human_conf * 100, 1), # Average Human confidence across engines
        "total_engine_count": total_engine_count,
        "high_risk_engine_count": high_risk_engine_count,
        "human_engine_count":     human_engine_count,
        "engines":          engines_dict,
    }


def full_profile_analysis(
    image=None,
    username="",
    bio="",
    followers=0,
    posts=0,
    following=0,
    account_age_days=0,
    image_source=None,
):
    """Analyze an image and expose the legacy profile-analysis response shape.

    The Streamlit profile view and older callers use profile-oriented key names,
    while the primary detector returns an engine-oriented result.  Keep both
    APIs working by adapting the detector output instead of duplicating the
    image analysis.
    """
    # Older profile UI code uses ``image_source``.  Accept it as an alias so
    # both profile app variants can call this shared detector.
    if image is None:
        image = image_source
    if image is None:
        raise ValueError("An image is required for profile analysis.")

    analysis = full_image_analysis(image)
    score = analysis["confidence_score"]
    engines = analysis["engines"]

    metadata_score = 0
    red_flags = []
    if followers < 10 and following > 100:
        metadata_score += 30
        red_flags.append("Low followers with high following")
    if posts == 0:
        metadata_score += 25
        red_flags.append("No posts")
    if account_age_days and account_age_days < 30:
        metadata_score += 20
        red_flags.append("Recently created account")
    if len(username) > 12 and any(char.isdigit() for char in username):
        metadata_score += 15
        red_flags.append("Suspicious username pattern")
    if not bio.strip():
        metadata_score += 10
        red_flags.append("Empty profile bio")
    metadata_score = min(metadata_score, 100)

    return {
        "overall_suspicion_score": score,
        # Use the ASCII identifier here so legacy console callers also work
        # on Windows terminals that default to a non-UTF-8 code page.
        "overall_verdict": analysis["verdict"],
        "image_analysis": {
            "ai_probability": score / 100,
            "is_ai_generated": analysis["verdict"] == "AI-GENERATED",
            "confidence_level": "high" if score >= 65 else "medium" if score >= 40 else "low",
            "individual_results": engines,
        },
        "metadata_analysis": {
            "metadata_suspicion_score": metadata_score,
            "red_flags": red_flags,
        },
        "clip_score": engines["clip_semantic"]["score"],
        "artifact_score": engines["texture_smoothness"]["score"],
        "symmetry_score": engines["face_symmetry"]["score"],
        "frequency_score": engines["frequency"]["score"],
        "watermark_score": engines["watermark_detection"]["score"],
    }
