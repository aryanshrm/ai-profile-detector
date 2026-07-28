import sys
sys.path.insert(0, 'files')
# pyrefly: ignore [missing-import]
from detector import full_image_analysis
from PIL import Image

img_ai   = Image.open('data/sample_images/_temp_upload.jpg')
img_real = Image.open('akshra.jpeg')

print("=== AI MAN IN SUIT ===")
res = full_image_analysis(img_ai)
v   = res["verdict"]
s   = res["confidence_score"]
print(f"Verdict: {v} | Score: {s}")
for k, e in res["engines"].items():
    print(f"  {e['name']}: {e['score']}/{e['max']} (raw={e['raw']:.2f})")

print()
print("=== REAL HUMAN (akshra.jpeg) ===")
res2 = full_image_analysis(img_real)
v2   = res2["verdict"]
s2   = res2["confidence_score"]
print(f"Verdict: {v2} | Score: {s2}")
for k, e in res2["engines"].items():
    print(f"  {e['name']}: {e['score']}/{e['max']} (raw={e['raw']:.2f})")
