import os, requests, base64, io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Load test image
img_path = 'data/sample_images/_temp_upload.jpg'
if not os.path.isfile(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")
img = Image.open(img_path)
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
b64 = base64.b64encode(buffer.getvalue()).decode()

# Gemini API key
gemini_key = os.getenv('GEMINI_API_KEY')
if not gemini_key:
    raise EnvironmentError('GEMINI_API_KEY not set')

url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}'
payload = {
    "contents": [{
        "role": "user",
        "parts": [
            {"text": "Test prompt"},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
        ]
    }],
    "generationConfig": {"response_mime_type": "application/json"},
    "temperature": 0.0
}

r = requests.post(url, json=payload, timeout=30)
print('Status code:', r.status_code)
print('Response snippet:', r.text[:500])

if r.status_code != 200:
    print('Error details:', r.text)
else:
    try:
        data = r.json()
        print('Received JSON:', data)
    except Exception as e:
        print('Failed to parse JSON:', e)
