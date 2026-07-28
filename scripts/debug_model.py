from transformers import pipeline
from PIL import Image

# Load model
print("Loading model...")
classifier = pipeline("image-classification", model="umm-maybe/AI-image-detector")
print("Model loaded!")

# Test with your image - change this path to your actual image
img = Image.open("data/sample_images/_temp_upload.jpg").convert("RGB")

results = classifier(img)
print("\nRaw model output:")
for r in results:
    print(r)