"""
Photo Express Cary — Passport Photo Studio
3-column layout: Original | Crop Tools | Result
"""

import io
import warnings
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

HAS_CROPPER = False  # using native slider crop instead

st.set_page_config(page_title="Photo Express Cary", page_icon="📸",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; font-size: 15px; }
.stApp { background: #f0f2f5; color: #1a1c22; }

[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e0e2e8; }
[data-testid="stSidebar"] * { color: #111827 !important; }
[data-testid="stSidebar"] h3 { font-size: 14px !important; font-weight: 700 !important; }
[data-testid="stSidebar"] label { font-size: 13px !important; font-weight: 500 !important; }

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
     font-size: 1.8rem !important; color: #1a1c22 !important; margin-bottom: 0 !important; }

.stDownloadButton > button, .stButton > button {
    background: #2563eb !important; color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 0.5rem 1rem !important; width: 100%;
}
.stDownloadButton > button:hover, .stButton > button:hover { opacity: 0.88 !important; }
[data-testid="stSlider"] > div > div > div { background: #2563eb !important; }
[data-testid="stSlider"] label { font-size: 12px !important; font-weight: 600 !important; color: #374151 !important; }
[data-testid="stSlider"] { margin-bottom: 4px !important; }

/* Column panels */
.col-panel {
    background: #ffffff;
    border: 1px solid #e0e2e8;
    border-radius: 14px;
    padding: 14px 14px 18px 14px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    height: 100%;
}
.col-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 0.82rem;
    color: #6b7280; text-transform: uppercase;
    letter-spacing: .08em; margin-bottom: 10px;
    padding-bottom: 8px; border-bottom: 1px solid #f0f0f0;
}
.col-num {
    display: inline-block; background: #2563eb; color: white;
    border-radius: 50%; width: 22px; height: 22px; line-height: 22px;
    text-align: center; font-weight: 700; font-size: 12px; margin-right: 6px;
}
.ctrl-label {
    font-size: 10px; font-weight: 700; color: #9ca3af;
    text-transform: uppercase; letter-spacing: .07em;
    margin: 10px 0 3px 0; display: block;
}
.section-rule { border: none; border-top: 1px solid #e0e2e8; margin: 12px 0; }
.img-label { text-align: center; font-size: 0.7rem; color: #9ca3af;
             letter-spacing: 0.1em; text-transform: uppercase;
             margin-top: 5px; font-weight: 500; }
.result-placeholder {
    border: 2px dashed #e0e2e8; border-radius: 10px;
    padding: 50px 14px; text-align: center;
    background: #f9fafb; margin-top: 4px;
}
.stat-row { display:flex; justify-content:space-between; font-size:0.78rem;
            color:#6b7280; margin-top:8px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def pil_to_bytes(img, fmt="JPEG", quality=95):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()

def strip_exif(img):
    data = list(img.getdata())
    clean = Image.new(img.mode, img.size)
    clean.putdata(data)
    return clean

def load_image_file(uploaded_file):
    """Load JPG, PNG, or ARW (Sony RAW) into a PIL RGB image."""
    fname = uploaded_file.name.lower()
    raw_exts = (".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2", ".dng")

    if any(fname.endswith(ext) for ext in raw_exts):
        if not HAS_RAWPY:
            st.error("rawpy is not installed. Run: pip install rawpy")
            st.stop()
        # Write to a temp file — rawpy needs a real file path
        import tempfile, os
        suffix = os.path.splitext(fname)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with rawpy.imread(tmp_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=False,
                output_bps=8,
            )
        os.unlink(tmp_path)
        return Image.fromarray(rgb)
    else:
        return Image.open(uploaded_file).convert("RGB")


def fix_shadows(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def remove_background(img):
    if not HAS_REMBG:
        return img
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = Image.open(io.BytesIO(rembg_remove(buf.getvalue()))).convert("RGBA")
    white = Image.new("RGBA", result.size, (255,255,255,255))
    white.paste(result, mask=result.split()[3])
    return white.convert("RGB")

def detect_face_bbox(img_np):
    if not HAS_MEDIAPIPE:
        return None
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as det:
        res = det.process(img_np)
    if not res.detections:
        return None
    bbox = res.detections[0].location_data.relative_bounding_box
    ih, iw = img_np.shape[:2]
    return (int(bbox.xmin*iw), int(bbox.ymin*ih), int(bbox.width*iw), int(bbox.height*ih))

def suggest_crop(face_bbox, img_shape):
    x, y, w, h = face_bbox
    ih, iw = img_shape[:2]
    crop_size   = int(h / 0.60)
    top_offset  = int(crop_size * 0.34)
    side_margin = int(crop_size * 0.16)
    face_cx = x + w // 2
    x1 = max(int(face_cx - crop_size/2 - side_margin), 0)
    y1 = max(int(y - top_offset), 0)
    cs = int(crop_size + 2*side_margin)
    cs = min(cs, iw-x1, ih-y1)
    return x1, y1, cs

def draw_preview(img_pil, x1, y1, crop_sz, max_w=360):
    iw, ih = img_pil.size
    scale = min(max_w/iw, 380/ih, 1.0)
    dw, dh = int(iw*scale), int(ih*scale)
    disp = img_pil.copy().resize((dw, dh), Image.LANCZOS)
    dx1 = max(0, min(int(x1*scale), dw-2))
    dy1 = max(0, min(int(y1*scale), dh-2))
    dsz = max(10, min(int(crop_sz*scale), dw-dx1, dh-dy1))
    overlay = Image.new("RGBA", (dw, dh), (0,0,0,110))
    blended = Image.alpha_composite(disp.convert("RGBA"), overlay)
    blended.paste(disp.crop((dx1, dy1, dx1+dsz, dy1+dsz)), (dx1, dy1))
    disp = blended.convert("RGB")
    draw = ImageDraw.Draw(disp)
    draw.rectangle([dx1, dy1, dx1+dsz-1, dy1+dsz-1], outline=(37,99,235), width=3)
    hs = 11
    for cx2, cy2 in [(dx1,dy1),(dx1+dsz,dy1),(dx1,dy1+dsz),(dx1+dsz,dy1+dsz)]:
        draw.rectangle([cx2-hs//2, cy2-hs//2, cx2+hs//2, cy2+hs//2], fill=(37,99,235))
    for cx2, cy2 in [(dx1+dsz//2,dy1),(dx1+dsz//2,dy1+dsz),(dx1,dy1+dsz//2),(dx1+dsz,dy1+dsz//2)]:
        draw.rectangle([cx2-hs//2, cy2-hs//2, cx2+hs//2, cy2+hs//2], fill=(96,165,250))
    for i in range(1,3):
        draw.line([(dx1+dsz*i//3,dy1),(dx1+dsz*i//3,dy1+dsz)], fill=(255,255,255), width=1)
        draw.line([(dx1,dy1+dsz*i//3),(dx1+dsz,dy1+dsz*i//3)], fill=(255,255,255), width=1)
    return disp

def apply_adjustments(img_np, exposure, shadows, contrast):
    """
    Apply exposure, shadow lift, and contrast to an RGB uint8 image.

    exposure : stops, -2.0 to +2.0  (multiplies luminance by 2^stops)
    shadows  : -100 to +100  (lifts/crushes pixels in the shadow range 0-128)
    contrast : -100 to +100  (S-curve style contrast around mid-grey)
    """
    img = img_np.astype(np.float32)

    # ── Exposure (in stops, like a real camera) ──────────────────────────────
    if abs(exposure) > 0.01:
        img *= (2.0 ** exposure)

    # ── Shadow lift / crush (only affects dark tones 0-128) ──────────────────
    if abs(shadows) > 0:
        # Mask that is 1.0 for pure black and 0.0 at mid-grey (128)
        shadow_mask = np.clip(1.0 - img / 128.0, 0.0, 1.0)
        img += shadow_mask * shadows

    # ── Contrast (S-curve around 128) ────────────────────────────────────────
    if abs(contrast) > 0:
        # Normalise to 0-1, apply scaled S-curve, back to 0-255
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        img = factor * (img - 128) + 128

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_crop_and_rotate(img_np, x1, y1, crop_sz, rotation, final_size):
    ih, iw = img_np.shape[:2]
    pil_img = Image.fromarray(img_np)
    if abs(rotation) > 0.1:
        pil_img = pil_img.rotate(-rotation, resample=Image.BICUBIC, expand=False)
    img_rot = np.array(pil_img)
    x2c = min(int(x1+crop_sz), iw)
    y2c = min(int(y1+crop_sz), ih)
    cropped = img_rot[int(y1):y2c, int(x1):x2c]
    if cropped.size == 0:
        cropped = img_rot
    return cv2.resize(cropped, (final_size, final_size))


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for k, v in [("crop_applied", False), ("last_file", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📸 Photo Express Cary")
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Output Size")
    # 1 = 200px, 10 = 1800px, evenly spaced
    size_levels = {1:200, 2:400, 3:600, 4:700, 5:800, 6:1000, 7:1200, 8:1400, 9:1600, 10:1800}
    qs1, qs2 = st.columns([3, 1])
    with qs1:
        size_level = st.slider("Quality Level", 1, 10, 8,
                               label_visibility="collapsed")
    with qs2:
        size_level = st.number_input("ql_num", 1, 10, int(size_level), 1,
                                     label_visibility="collapsed", key="ql_num")
    final_size = size_levels[int(size_level)]
    st.markdown(
        f'<div style="text-align:center;background:#f0f6ff;border:1px solid #bfdbfe;' +
        f'border-radius:6px;padding:5px;margin-top:4px;">' +
        f'<span style="font-weight:700;color:#2563eb;font-size:1rem;">Level {size_level}</span>' +
        f'<span style="color:#6b7280;font-size:0.8rem;"> · {final_size} × {final_size} px</span></div>',
        unsafe_allow_html=True)
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Enhancements")
    do_strip_exif = st.toggle("Strip EXIF metadata", value=True,
                               help="Removes GPS, camera info and hidden data from output")

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Photo Metadata")
    st.markdown(
        '<p style="font-size:11px;color:#9ca3af;margin:-4px 0 8px 0;">'
        'Optional — useful for Canadian passport and other formats</p>',
        unsafe_allow_html=True)
    meta_enabled = st.toggle("Embed metadata in output", value=False,
                              help="Adds studio info into the image EXIF fields")
    if meta_enabled:
        meta_date         = st.text_input("Photo taken date",
                                          value=datetime.now().strftime("%Y-%m-%d"))
        meta_studio       = st.text_input("Studio name", value="Photo Express Cary")
        meta_photographer = st.text_input("Photographer", value="", placeholder="Optional")
        meta_notes        = st.text_area("Notes", value="",
                                         placeholder="Any additional notes...", height=80)
    else:
        meta_date = meta_studio = meta_photographer = meta_notes = ""

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Photo Adjustments")

    st.markdown('<span style="font-size:12px;font-weight:600;color:#6b7280;">EXPOSURE (stops)</span>', unsafe_allow_html=True)
    exposure_val = st.slider("Exposure", -2.0, 2.0, 0.0, 0.05,
                             key="exposure_val", label_visibility="collapsed",
                             help="Brighten or darken the whole image")

    st.markdown('<span style="font-size:12px;font-weight:600;color:#6b7280;">SHADOWS</span>', unsafe_allow_html=True)
    shadow_val = st.slider("Shadows", -100, 100, 0, 1,
                           key="shadow_val", label_visibility="collapsed",
                           help="Lift or crush dark areas")

    st.markdown('<span style="font-size:12px;font-weight:600;color:#6b7280;">CONTRAST</span>', unsafe_allow_html=True)
    contrast_val = st.slider("Contrast", -100, 100, 0, 1,
                             key="contrast_val", label_visibility="collapsed",
                             help="Increase or reduce tonal contrast")
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    if not HAS_MEDIAPIPE:
        st.warning("MediaPipe not installed.")
    if not HAS_REMBG:
        st.caption("Install rembg[cpu] for background removal.")
    if not HAS_RAWPY:
        st.caption("Install rawpy for RAW/ARW support.")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER + UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 📸 Photo Express Cary")
st.markdown(
    '<p style="color:#6b7280;font-size:0.88rem;margin-top:-8px;margin-bottom:14px;">' +
    "US Passport Photo Studio · 2×2 inch · Cary, NC</p>",
    unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a photo (JPG / PNG / ARW / RAW)",
    type=["jpg","jpeg","png","arw","cr2","cr3","nef","orf","raf","rw2","dng"],
    label_visibility="collapsed")

if not uploaded_file:
    st.markdown(
        '<div style="border:2px dashed #d1d5db;border-radius:12px;padding:44px;' +
        'text-align:center;background:#ffffff;margin-top:8px;">' +
        '<div style="font-size:2rem;margin-bottom:8px;">📷</div>' +
        '<div style="font-size:1.05rem;color:#6b7280;font-weight:600;">Drop a photo here to begin</div>' +
        '<div style="font-size:0.8rem;color:#9ca3af;margin-top:4px;">JPG · PNG · ARW · CR2 · NEF · DNG and more</div>' +
        '<div style="font-size:0.8rem;color:#d1d5db;margin-top:4px;">JPG or PNG · any size</div></div>',
        unsafe_allow_html=True)
    st.stop()

if st.session_state.last_file != uploaded_file.name:
    st.session_state.crop_applied = False
    st.session_state.last_file = uploaded_file.name

# Load image — handles JPG, PNG, and RAW formats (ARW, CR2, NEF, etc.)
is_raw = any(uploaded_file.name.lower().endswith(e)
             for e in (".arw",".cr2",".cr3",".nef",".orf",".raf",".rw2",".dng"))
if is_raw:
    with st.spinner("Processing RAW file — this may take a moment..."):
        original_pil = load_image_file(uploaded_file)
else:
    original_pil = Image.open(uploaded_file).convert("RGB")
original_np  = np.array(original_pil)
ih, iw = original_np.shape[:2]

do_bg_remove  = False
do_shadow_fix = False

face_bbox = detect_face_bbox(original_np)
if face_bbox:
    sx1, sy1, ssize = suggest_crop(face_bbox, original_np.shape)
else:
    ssize = min(iw, ih)
    sx1   = (iw - ssize) // 2
    sy1   = (ih - ssize) // 2

# Current crop values
cur_x   = max(0, min(st.session_state.get("crop_x",   int(sx1)), iw-50))
cur_y   = max(0, min(st.session_state.get("crop_y",   int(sy1)), ih-50))
cur_sz  = max(50, min(st.session_state.get("crop_sz",  int(ssize)), iw-cur_x, ih-cur_y))
cur_rot = st.session_state.get("rotation", 0.0)

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3 COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

# ── COLUMN 1: Original + live preview ────────────────────────────────────────
with col1:
    st.markdown(
        '<div class="col-header"><span class="col-num">1</span>Original Photo</div>',
        unsafe_allow_html=True)

    if face_bbox:
        st.success("Face detected", icon="✅")
    else:
        st.warning("No face found — adjust manually", icon="⚠️")

    # Read current crop values
    cur_x   = max(0, min(st.session_state.get("crop_x",   int(sx1)), iw - 50))
    cur_y   = max(0, min(st.session_state.get("crop_y",   int(sy1)), ih - 50))
    cur_sz  = max(50, min(st.session_state.get("crop_sz",  int(ssize)), iw - cur_x, ih - cur_y))
    cur_rot = st.session_state.get("rotation", 0.0)

    # Draw live preview with crop box overlay
    preview = draw_preview(original_pil, cur_x, cur_y, cur_sz, max_w=400)
    st.image(preview, use_container_width=True)
    st.markdown(
        f'<p class="img-label">Blue box = crop · {cur_sz}×{cur_sz} px</p>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="stat-row"><span>{iw}×{ih} px</span>'
        f'<span>{uploaded_file.name[:22]}</span></div>',
        unsafe_allow_html=True)


# ── COLUMN 2: Crop controls ───────────────────────────────────────────────────
with col2:
    st.markdown(
        '<div class="col-header"><span class="col-num">2</span>Crop & Rotation Tools</div>',
        unsafe_allow_html=True)

    st.markdown('<span class="ctrl-label">📍 Horizontal Position</span>', unsafe_allow_html=True)
    new_x = st.slider("Horizontal", 0, max(iw - 50, 1), cur_x,
                      key="crop_x", label_visibility="collapsed")

    st.markdown('<span class="ctrl-label">📍 Vertical Position</span>', unsafe_allow_html=True)
    new_y = st.slider("Vertical", 0, max(ih - 50, 1), cur_y,
                      key="crop_y", label_visibility="collapsed")

    st.markdown('<span class="ctrl-label">⬛ Crop Size (square)</span>', unsafe_allow_html=True)
    max_sz = max(min(iw - new_x, ih - new_y), 51)
    new_sz = st.slider("Size", 50, max_sz, min(cur_sz, max_sz),
                       key="crop_sz", label_visibility="collapsed")

    st.markdown('<span class="ctrl-label">🔄 Rotation</span>', unsafe_allow_html=True)
    new_rot = st.slider("Rotation", -45.0, 45.0, cur_rot, 0.5,
                        key="rotation", label_visibility="collapsed")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

    # Info box
    st.markdown(
        f'<div style="text-align:center;background:#f0f6ff;border:1px solid #bfdbfe;'
        f'border-radius:8px;padding:10px;margin-bottom:12px;">'
        f'<span style="font-size:1rem;font-weight:700;color:#2563eb;">'
        f'{new_sz} × {new_sz} px</span>'
        f'<span style="font-size:0.8rem;color:#6b7280;"> · {new_rot:+.1f}°</span>'
        f'<br><span style="font-size:0.75rem;color:#9ca3af;">'
        f'Position: X={new_x} Y={new_y}</span></div>',
        unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("↺  Reset to AI", use_container_width=True):
            st.session_state.crop_x       = int(sx1)
            st.session_state.crop_y       = int(sy1)
            st.session_state.crop_sz      = int(ssize)
            st.session_state.rotation     = 0.0
            st.session_state.crop_applied = False
            st.session_state.pop("confirmed_crop", None)
            st.rerun()
    with b2:
        if st.button("✓  Apply Crop", use_container_width=True, type="primary"):
            st.session_state["confirmed_x"]   = new_x
            st.session_state["confirmed_y"]   = new_y
            st.session_state["confirmed_sz"]  = new_sz
            st.session_state["confirmed_rot"] = new_rot
            st.session_state.crop_applied     = True
            st.rerun()


# ── COLUMN 3: Final result + download ────────────────────────────────────────
with col3:
    st.markdown(
        '<div class="col-header"><span class="col-num">3</span>Passport Photo · Download</div>',
        unsafe_allow_html=True)

    # Crop must be confirmed at least once before showing result
    if "confirmed_sz" not in st.session_state:
        st.markdown(
            '<div class="result-placeholder">'
            '<div style="font-size:2.8rem;margin-bottom:10px;">🪪</div>'
            '<div style="font-size:0.95rem;font-weight:600;color:#6b7280;">Result appears here</div>'
            '<div style="font-size:0.8rem;color:#d1d5db;margin-top:6px;">'
            'Adjust sliders in column 2<br>then click <strong>✓ Apply Crop</strong></div></div>',
            unsafe_allow_html=True)
    else:
        # Crop is locked to confirmed values — adjustments apply live
        fx  = st.session_state["confirmed_x"]
        fy  = st.session_state["confirmed_y"]
        fsz = st.session_state["confirmed_sz"]
        frt = st.session_state["confirmed_rot"]

        passport_np = apply_crop_and_rotate(original_np, fx, fy, fsz, frt, final_size)

        if do_bg_remove:
            passport_np = np.array(remove_background(Image.fromarray(passport_np)))

        # Adjustments (exposure, shadows, contrast) update live on every slider move
        adjusted = apply_adjustments(passport_np, exposure_val, shadow_val, contrast_val)

        passport_np  = adjusted.astype(np.uint8)
        passport_pil = Image.fromarray(passport_np)
        if do_strip_exif:
            passport_pil = strip_exif(passport_pil)

        # Embed optional metadata into correct EXIF/XMP fields
        custom_exif_bytes = None
        if meta_enabled and any([meta_date, meta_studio, meta_photographer, meta_notes]):
            try:
                import piexif
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

                # Date Taken → DateTime (tag 306) + DateTimeOriginal (Exif tag 36867)
                # Windows "Date taken" reads DateTimeOriginal
                if meta_date:
                    date_exif = (meta_date.replace("-", ":") + " 00:00:00").encode("utf-8")
                    exif_dict["0th"][piexif.ImageIFD.DateTime] = date_exif
                    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = date_exif
                    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = date_exif

                # Authors → Artist (315) — Windows "Authors" field
                if meta_photographer:
                    artist_str = "Photographer: " + meta_photographer
                    exif_dict["0th"][piexif.ImageIFD.Artist] = artist_str.encode("utf-8")

                # Subject → XPSubject (40095) — Windows "Subject" field
                if meta_notes:
                    subject_bytes = ("Address: " + meta_notes).encode("utf-16-le") + bytes([0, 0])
                    exif_dict["0th"][40095] = subject_bytes

                # Studio → XPComment (40092) — Windows "Comments" field
                if meta_studio:
                    comment_bytes = ("Studio: " + meta_studio).encode("utf-16-le") + bytes([0, 0])
                    exif_dict["0th"][40092] = comment_bytes

                # Software tag — always write
                exif_dict["0th"][piexif.ImageIFD.Software] = b"Photo Express Cary"

                custom_exif_bytes = piexif.dump(exif_dict)
            except Exception as e:
                st.caption(f"Metadata embedding skipped: {e}")

        st.image(passport_pil, use_container_width=True)
        st.markdown('<p class="img-label">2 × 2 inch · 51 × 51 mm</p>', unsafe_allow_html=True)

        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

        file_stem = st.text_input(
            "File name",
            value=f"passport-{datetime.now().strftime('%Y%m%d')}",
            label_visibility="visible")

        if custom_exif_bytes:
            buf = io.BytesIO()
            passport_pil.save(buf, format="JPEG", quality=95, exif=custom_exif_bytes)
            photo_bytes = buf.getvalue()
        else:
            photo_bytes = pil_to_bytes(passport_pil, fmt="JPEG", quality=95)
        file_kb     = len(photo_bytes) / 1024
        size_str    = f"{'%.2f' % (file_kb/1024)} MB" if file_kb > 1024 else f"{'%.1f' % file_kb} KB"

        st.download_button(
            "⬇  Download Passport Photo",
            data=photo_bytes,
            file_name=f"{file_stem}.jpg",
            mime="image/jpeg",
            use_container_width=True)

        st.markdown(
            f'<div class="stat-row">'
            f'<span>{final_size}×{final_size} px</span>'
            f'<span>{size_str}</span>'
            f'<span>EXIF {"stripped" if do_strip_exif else "kept"}</span></div>',
            unsafe_allow_html=True)

        st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
        if st.button("← Re-adjust Crop", use_container_width=True):
            st.session_state.pop("confirmed_sz", None)
            st.session_state.crop_applied = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)