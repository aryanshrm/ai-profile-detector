"""Quick test to verify scoring."""
import sys
from PIL import Image

sys.path.insert(0, 'files')
from src.detector import full_profile_analysis

img = Image.open("akshra.jpeg").convert("RGB")

result = full_profile_analysis(img, username="cristiano", followers=500, posts=300, following=499, account_age_days=34)

print("=" * 50)
print(f"Verdict:        {result['overall_verdict']}")
print(f"Score:          {result['overall_suspicion_score']:.2f} / 100")
print(f"AI Generated?   {result['image_analysis']['is_ai_generated']}")
print(f"Confidence:     {result['image_analysis']['confidence_level']}")
print(f"CLIP Score:     {result['clip_score']:.2f}")
print(f"Artifact Score: {result['artifact_score']}")
print(f"Symmetry Score: {result['symmetry_score']}")
print(f"Frequency Score:{result['frequency_score']}")
print(f"Metadata Score: {result['metadata_analysis']['metadata_suspicion_score']}")
print("=" * 50)
