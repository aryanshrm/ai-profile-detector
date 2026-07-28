"""Compare raw metric values for both AI and real images side by side."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2
from PIL import Image

def analyze(path, label):
    img  = Image.open(path).convert("RGB")
    arr  = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    print(f"\n{'='*60}")
    print(f"  {label}  ({img.size[0]}x{img.size[1]})")
    print(f"{'='*60}")

    # TEXTURE
    rh, rw = int(h * 0.3), int(w * 0.3)
    center = gray[cy-rh:cy+rh, cx-rw:cx+rw]
    blur3  = cv2.GaussianBlur(center, (3, 3), 0)
    blur11 = cv2.GaussianBlur(center, (11, 11), 0)
    fine   = float(np.std(center - blur3))
    coarse = float(np.std(center - blur11))
    print(f"[TEXTURE] fine={fine:.3f}  coarse={coarse:.3f}")

    # COLOR
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:,:,1].astype(np.float64)
    val = hsv[:,:,2].astype(np.float64)
    dark_mask = val < 230
    sat_fg = sat[dark_mask]
    mean_sat_fg = float(np.mean(sat_fg)) if sat_fg.size > 0 else 0
    sat_std_fg  = float(np.std(sat_fg)) if sat_fg.size > 0 else 0
    # Histogram roughness
    img_arr = arr.astype(np.float64)
    roughness_vals = []
    for ch in [img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]]:
        hist, _ = np.histogram(ch.ravel(), bins=64, range=(0,256))
        hist = hist.astype(np.float64) / (hist.sum() + 1e-8)
        roughness_vals.append(float(np.mean(np.abs(np.diff(hist)))))
    hist_rough = float(np.mean(roughness_vals))
    rg_corr = float(np.corrcoef(img_arr[:,:,0].ravel(), img_arr[:,:,1].ravel())[0,1])
    rb_corr = float(np.corrcoef(img_arr[:,:,0].ravel(), img_arr[:,:,2].ravel())[0,1])
    avg_corr = (abs(rg_corr) + abs(rb_corr)) / 2
    print(f"[COLOR]   sat_fg_mean={mean_sat_fg:.2f}  sat_fg_std={sat_std_fg:.2f}")
    print(f"[COLOR]   hist_roughness={hist_rough:.6f}  channel_corr={avg_corr:.4f}")
    print(f"[COLOR]   dark_pixels={dark_mask.sum()} / {dark_mask.size} ({dark_mask.mean()*100:.1f}%)")

    # FFT
    f_fft  = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f_fft)
    mag    = np.log(np.abs(fshift) + 1)
    center_mag = mag[h//4:3*h//4, w//4:3*w//4]
    outer  = mag.copy()
    outer[h//4:3*h//4, w//4:3*w//4] = 0
    ce    = float(np.mean(center_mag))
    oe    = float(np.mean(outer[outer > 0])) if np.any(outer > 0) else 0
    ratio = oe / (ce + 1e-5)
    vhf   = mag.copy()
    q_h, q_w = h//8, w//8
    vhf[q_h:7*q_h, q_w:7*q_w] = 0
    vhf_energy = float(np.mean(vhf[vhf > 0])) if np.any(vhf > 0) else 0
    vhf_ratio  = vhf_energy / (ce + 1e-5)
    print(f"[FFT]     outer/center={ratio:.4f}  vhf_ratio={vhf_ratio:.4f}")

    # EDGE
    rh2, rw2 = int(h*0.25), int(w*0.25)
    subj = gray[cy-rh2:cy+rh2, cx-rw2:cx+rw2]
    bh, bw = int(h*0.20), int(w*0.20)
    bg_laps = [cv2.Laplacian(gray[:bh,:bw].astype(np.uint8), cv2.CV_64F).var(),
               cv2.Laplacian(gray[:bh,w-bw:].astype(np.uint8), cv2.CV_64F).var(),
               cv2.Laplacian(gray[h-bh:,:bw].astype(np.uint8), cv2.CV_64F).var(),
               cv2.Laplacian(gray[h-bh:,w-bw:].astype(np.uint8), cv2.CV_64F).var()]
    subj_lap = float(cv2.Laplacian(subj.astype(np.uint8), cv2.CV_64F).var())
    bg_lap   = float(np.mean(bg_laps))
    ratio_sharp = subj_lap / (bg_lap + 1e-5)
    sobx = cv2.Sobel(subj.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
    soby = cv2.Sobel(subj.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobx**2 + soby**2)
    strong = grad[grad > np.percentile(grad, 90)]
    grad_cv = float(np.std(strong) / (np.mean(strong) + 1e-8))
    print(f"[EDGE]    subj_lap={subj_lap:.1f}  bg_lap={bg_lap:.1f}  ratio={ratio_sharp:.2f}")
    print(f"[EDGE]    grad_uniformity_CV={grad_cv:.4f}")

    # SYMMETRY
    hw = w // 2
    left  = gray[:, :hw]
    right = cv2.flip(gray[:, w-hw:], 1)
    mh2   = min(left.shape[0], right.shape[0])
    mw2   = min(left.shape[1], right.shape[1])
    sym_diff = float(np.mean(np.abs(left[:mh2,:mw2] - right[:mh2,:mw2])))
    mean_brightness = float(np.mean(gray))
    rel_diff = sym_diff / (mean_brightness + 1e-5)
    print(f"[SYMM]    abs_diff={sym_diff:.3f}  brightness={mean_brightness:.1f}  rel_diff={rel_diff:.4f}")

analyze('data/sample_images/_temp_upload.jpg', 'AI IMAGE: Man in Suit')
analyze('akshra.jpeg', 'REAL IMAGE: akshra.jpeg')
