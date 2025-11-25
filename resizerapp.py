"""
Streamlit Image Resizer - lightweight single-file app
Features:
- Upload multiple images or a zip containing a folder of images
- Resize by target file size (KB/MB) or by resolution (pixels / cm / inches)
- Controls for DPI when converting between physical units and pixels
- Option to preserve aspect ratio
- Preview before & after (small thumbnails)
- Download all converted images as a ZIP with a user-provided filename

Usage:
- Run locally: `pip install -r requirements.txt` then `streamlit run streamlit_image_resizer.py`
- Deploy: push this file + requirements.txt to a GitHub repo and use Streamlit Cloud or any other host

Dependencies: streamlit, pillow
"""

import streamlit as st
from PIL import Image, ImageOps
import io
import zipfile
import os
import math
import tempfile
from typing import List, Tuple

# ----------------------- Helpers -----------------------

SUPPORTED_FORMATS = ("PNG", "JPEG", "JPG", "WEBP", "BMP")

def read_images_from_zip(uploaded_zip) -> List[Tuple[str, bytes]]:
    images = []
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
        for name in z.namelist():
            if name.endswith('/'):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
                images.append((os.path.basename(name), z.read(name)))
    return images


def load_uploaded_files(uploaded_files) -> List[Tuple[str, bytes]]:
    images = []
    for f in uploaded_files:
        try:
            images.append((f.name, f.read()))
        except Exception:
            continue
    return images


def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert('RGB')


def image_to_bytes(img: Image.Image, fmt: str, quality: int = 85, dpi: Tuple[int,int]=None) -> bytes:
    buf = io.BytesIO()
    save_args = {}
    if fmt.upper() in ('JPEG', 'JPG'):
        save_args['format'] = 'JPEG'
        save_args['quality'] = quality
        save_args['optimize'] = True
    elif fmt.upper() == 'PNG':
        save_args['format'] = 'PNG'
        # Pillow ignores quality for PNG; we can try optimize
        save_args['optimize'] = True
    elif fmt.upper() == 'WEBP':
        save_args['format'] = 'WEBP'
        save_args['quality'] = quality
    else:
        save_args['format'] = fmt

    if dpi:
        save_args['dpi'] = dpi

    img.save(buf, **save_args)
    return buf.getvalue()


def resize_by_pixels(img: Image.Image, target_w: int, target_h: int, keep_aspect: bool=True) -> Image.Image:
    if keep_aspect:
        img = ImageOps.contain(img, (target_w, target_h))
    else:
        img = img.resize((target_w, target_h), Image.LANCZOS)
    return img


def cm_to_pixels(cm: float, dpi: int) -> int:
    inches = cm / 2.54
    return max(1, int(round(inches * dpi)))


def inches_to_pixels(inches: float, dpi: int) -> int:
    return max(1, int(round(inches * dpi)))


def target_size_binary_search(img: Image.Image, fmt: str, target_bytes: int, max_iter=12, dpi=None) -> Tuple[bytes,int]:
    """Binary search on JPEG/WEBP quality to approach target file size. Returns bytes and quality used.
    If fmt is PNG, will first attempt PNG; if too big, convert to JPEG.
    """
    # For PNGs, try saving optimized first
    if fmt.upper() == 'PNG':
        b = image_to_bytes(img, 'PNG', quality=100, dpi=dpi)
        if len(b) <= target_bytes:
            return b, None
        # fallback to JPEG
        fmt = 'JPEG'

    # clamp target
    target_bytes = max(1, int(target_bytes))

    lo, hi = 5, 95
    best = None
    while max_iter > 0 and lo <= hi:
        mid = (lo + hi) // 2
        b = image_to_bytes(img, fmt, quality=mid, dpi=dpi)
        size = len(b)
        # print('try', mid, size)
        if size > target_bytes:
            hi = mid - 1
        else:
            best = (b, mid)
            lo = mid + 1
        max_iter -= 1
    if best:
        return best
    else:
        # if no quality yields <= target, return lowest quality attempt
        b = image_to_bytes(img, fmt, quality=lo, dpi=dpi)
        return b, lo

# ----------------------- Streamlit App -----------------------

st.set_page_config(page_title="Lightweight Image Resizer", layout="wide")
st.title("📦 Lightweight Image Resizer")
st.caption("Upload images (or a ZIP folder). Resize by target file size (KB/MB) or by resolution (pixels / cm / inches).")

# Input selection
with st.expander('Input selection (upload files or upload a zip of a folder)', expanded=True):
    col1, col2 = st.columns([2,1])
    with col1:
        files = st.file_uploader("Choose image files", type=['png','jpg','jpeg','webp','bmp'], accept_multiple_files=True)
    with col2:
        zip_file = st.file_uploader("Or upload a ZIP containing a folder of images", type=['zip'], accept_multiple_files=False)

# Collect images
images = []  # list of (filename, bytes)
if zip_file is not None:
    try:
        images = read_images_from_zip(zip_file)
        st.success(f"Found {len(images)} image(s) inside ZIP")
    except Exception as e:
        st.error(f"Failed to read ZIP: {e}")

if files:
    uploaded_imgs = load_uploaded_files(files)
    images.extend(uploaded_imgs)

if not images:
    st.info("No images uploaded yet. Upload files or a zip to start.")

# Settings
st.markdown("---")
st.header("Resize Settings")
mode = st.radio("Resize mode", ['Target file size', 'Target resolution'], index=0)

col1, col2, col3 = st.columns(3)
with col1:
    output_format = st.selectbox("Output format", options=['KEEP (original)', 'JPEG', 'PNG', 'WEBP'], index=0)
    keep_aspect = st.checkbox('Preserve aspect ratio', value=True)

with col2:
    if mode == 'Target file size':
        size_input = st.number_input('Target size', min_value=1.0, value=200.0, step=1.0, help='Enter value in chosen unit')
        size_unit = st.selectbox('Unit', ['KB','MB'], index=0)
        # convert to bytes later
    else:
        # Target resolution
        res_mode = st.selectbox('Resolution type', ['Pixels', 'Centimeters', 'Inches'], index=0)
        w_val = st.number_input('Width', min_value=1.0, value=1024.0, step=1.0)
        h_val = st.number_input('Height', min_value=1.0, value=1024.0, step=1.0)
        dpi = st.number_input('DPI (for cm/inches conversion)', min_value=72, value=300)

with col3:
    if mode == 'Target file size':
        prefer_format_when_size = st.selectbox('Prefer format if size target', ['JPEG', 'WEBP', 'PNG'], index=0)
        size_tolerance = st.number_input('Tolerance (%)', min_value=0.1, value=10.0, step=0.1)
    else:
        res_unit_note = st.write('Enter desired width/height in the chosen unit.')

st.markdown("---")

# Conversion controls
st.header('Conversion')
output_name = st.text_input('Base name for download file (zip will use this as prefix)', value='resized_images')
run = st.button('Start Conversion')

# Perform conversion
if run and images:
    st.info('Starting conversion — processing images now...')
    tmpdir = tempfile.mkdtemp()
    output_files = []  # list of (name, bytes)
    progress = st.progress(0)
    total = len(images)
    i = 0

    for fname, fbytes in images:
        i += 1
        try:
            img = pil_from_bytes(fbytes)
        except Exception as e:
            st.warning(f"Skipping {fname}: cannot open as image ({e})")
            continue

        orig_size = len(fbytes)
        orig_w, orig_h = img.size

        # Decide output format
        fmt = output_format
        if fmt == 'KEEP (original)':
            fmt = os.path.splitext(fname)[1].replace('.', '').upper() or 'JPEG'
            if fmt == 'JPG':
                fmt = 'JPEG'
        fmt = fmt.upper()

        converted = None
        used_quality = None

        if mode == 'Target file size':
            # compute target bytes
            if size_unit == 'KB':
                target_bytes = int(size_input * 1024)
            else:
                target_bytes = int(size_input * 1024 * 1024)

            # If user wants KEEP original format and original is already under target, just keep
            if fmt in ('PNG','JPEG','WEBP','BMP'):
                # attempt binary search on quality for JPEG/WEBP
                # If format is PNG and we want to reduce, convert to prefer_format
                preferred = prefer_format_when_size
                if fmt == 'PNG' and preferred in ('JPEG','WEBP'):
                    # convert to preferred before searching
                    tmp_img = img.copy()
                    b, q = target_size_binary_search(tmp_img, preferred, target_bytes, dpi=(dpi,dpi) if 'dpi' in locals() else None)
                    converted = b
                    used_quality = q
                    out_ext = preferred.lower()
                else:
                    # try current format first
                    b, q = target_size_binary_search(img, fmt, target_bytes, dpi=(dpi,dpi) if 'dpi' in locals() else None)
                    converted = b
                    used_quality = q
                    out_ext = fmt.lower()

        else:
            # Target resolution
            if res_mode == 'Pixels':
                target_w = int(round(w_val))
                target_h = int(round(h_val))
            elif res_mode == 'Centimeters':
                target_w = cm_to_pixels(w_val, dpi)
                target_h = cm_to_pixels(h_val, dpi)
            else:
                target_w = inches_to_pixels(w_val, dpi)
                target_h = inches_to_pixels(h_val, dpi)

            resized = resize_by_pixels(img, target_w, target_h, keep_aspect)
            # Save with default quality
            save_fmt = fmt if fmt != 'BMP' else 'PNG'  # BMP no compression -> avoid large
            b = image_to_bytes(resized, save_fmt, quality=85, dpi=(dpi,dpi) if 'dpi' in locals() else None)
            converted = b
            out_ext = save_fmt.lower()

        if converted is None:
            # fallback: save as jpeg at moderate quality
            converted = image_to_bytes(img, 'JPEG', quality=85)
            out_ext = 'jpg'

        safe_name = os.path.splitext(fname)[0]
        out_name = f"{safe_name}_resized.{out_ext}"
        output_files.append((out_name, converted))

        progress.progress(int(i/total * 100))

    # Package into zip
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, b in output_files:
            zout.writestr(name, b)
    zip_bytes = zip_buf.getvalue()

    # Show preview - show up to 4 thumbnails (before/after)
    st.success(f'Conversion done: {len(output_files)} file(s) converted')
    preview_count = min(4, len(output_files))
    if preview_count > 0:
        st.subheader('Preview (first few converted images)')
        cols = st.columns(preview_count)
        for idx in range(preview_count):
            name, b = output_files[idx]
            with cols[idx]:
                st.image(b, caption=name, use_column_width=True)

    st.download_button('Download ZIP', data=zip_bytes, file_name=f"{output_name}.zip", mime='application/zip')

    st.markdown('---')
    st.write('🙏 Thank you — conversion complete!')
    if st.button('Start a new conversion'):
        st.experimental_rerun()

else:
    if run and not images:
        st.error('No images were provided. Upload images or a zip file first.')

# End of app


# Minimal requirements that should be included in requirements.txt
# streamlit
# pillow
