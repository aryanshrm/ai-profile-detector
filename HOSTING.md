# Hosting NEXUS+ for a Portfolio Live Link

## Recommended first option: Streamlit Community Cloud

Use this first because the project is already a Streamlit app.

### 1. Push the project to GitHub
Make sure your repository contains:

- `app.py`
- `src/detector.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `runtime.txt`

Do not commit `.venv/`, logs, caches, API keys, or large local model files.

### 2. Deploy
1. Go to Streamlit Community Cloud.
2. Sign in with GitHub.
3. Click **New app**.
4. Select repository: `aryanshrm/ai-profile-detector`.
5. Branch: `main`.
6. Main file path: `app.py`.
7. Deploy.

### 3. Add secrets only if needed
If you use optional API engines, add secrets in the Streamlit Cloud app settings, not in GitHub:

```toml
GEMINI_API_KEY = "your-key"
GROQ_API_KEY = "your-key"
```

### 4. Portfolio text
Use something like:

> NEXUS+ — AI image forensics dashboard using Streamlit, PyTorch, CLIP, OpenCV, FFT, ELA, and explainable multi-engine scoring.

## Important note about the ViT model
The app works without `fine_tuned_vit/model.safetensors`; that engine simply stays inactive.
If you want the hosted app to use the fine-tuned model, upload the model correctly with Git LFS or host it separately. Very large model files can make free hosting slower.

## If Streamlit Cloud is too slow
Use Hugging Face Spaces with Streamlit SDK. It is often better for ML demos with larger models.
