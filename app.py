from __future__ import annotations

import html
import hashlib
import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from accounts import authenticate_account, initialize_accounts, register_account
from caption_model import generate_caption, load_caption_engine, train_captioner


st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
APP_DATA = ROOT / ".app_data"
MODEL_DIR = APP_DATA / "caption_model"
MODEL_PATH = MODEL_DIR / "caption_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
CAPTION_CANDIDATES = (
    ROOT / "Flickr8k" / "Flickr8k.token.txt",
    ROOT / "Flickr8k.token.txt",
)
IMAGE_CANDIDATES = (
    ROOT / "Flickr8k_Images" / "Flicker8k_Dataset",
    ROOT / "Flickr8k_Dataset",
    ROOT / "Flickr8k" / "Flicker8k_Dataset",
)
PAGE_SIZE = 12


st.markdown(
    """
    <style>
    :root {
        --paper: #07111f;
        --ink: #e7efff;
        --muted: #a5b4c8;
        --line: #26364b;
        --green: #67a9ff;
        --lime: #b4d5ff;
        --coral: #80bdff;
        --blue: #60a5fa;
    }
    .stApp { background: linear-gradient(145deg, #0a1729 0%, #07111f 48%, #050b14 100%); color: var(--ink); }
    html, body, [class*="css"] { font-family: 'Segoe UI', 'Trebuchet MS', sans-serif; }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: var(--ink); letter-spacing: 0; }
    h1, h2 { font-family: 'Book Antiqua', 'Palatino Linotype', Georgia, serif; font-weight: 500; }
    [data-testid="stSidebar"] { background: #080f1a; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1 { font-size: 1.35rem; }
    [data-testid="stMetric"] { background: #0d1a2a; border: 1px solid var(--line); padding: 15px 18px; border-radius: 6px; }
    [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: #8bbdff; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); border-radius: 6px; background: rgba(13,26,42,.76); }
    .eyebrow { color: #8bbdff; font-size: .73rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .hero-title { font: 500 3.6rem/1.02 'Book Antiqua', 'Palatino Linotype', Georgia, serif; max-width: 640px; margin: .45rem 0 1rem; }
    .hero-copy { color: var(--muted); font-size: 1rem; line-height: 1.65; max-width: 650px; }
    .status-pill { display: inline-block; padding: 6px 10px; border-radius: 4px; background: #15345a; color: #b8d9ff; font-size: .75rem; font-weight: 700; }
    .caption-quote { border-left: 3px solid var(--coral); padding: 12px 0 12px 15px; margin: 13px 0; color: var(--ink); line-height: 1.6; background: #0d1a2a; }
    .small-note { color: var(--muted); font-size: .84rem; line-height: 1.55; }
    .section-kicker { color: var(--muted); font-size: .82rem; }
    div.stButton > button { border-radius: 4px; border-color: #4b8fe8; color: #c5ddff; background: #10243d; font-weight: 600; }
    div.stButton > button[kind="primary"] { background: #2875d4; border-color: #438ce8; color: #fff; }
    div.stButton > button:hover { border-color: #8bbdff; color: #fff; }
    [data-baseweb="input"] > div, [data-baseweb="select"] > div { background: #0b1726; border-color: #30445e; }
    [data-baseweb="input"] input, [data-baseweb="select"] input { color: #e7efff; }
    [data-testid="stFileUploader"] { background: #0d1a2a; border: 1px dashed #477dbd; border-radius: 6px; }
    [data-testid="stAlert"] { background: #10243d; border-color: #315d91; }
    footer { visibility: hidden; }
    @media (max-width: 640px) { .hero-title { font-size: 2.7rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading Flickr8k captions and image index...")
def load_project_data():
    caption_file = next((path for path in CAPTION_CANDIDATES if path.is_file()), None)
    image_root = next((path for path in IMAGE_CANDIDATES if path.is_dir()), None)

    if caption_file is None:
        return {}, {}, ""

    captions: dict[str, list[str]] = {}
    with caption_file.open("r", encoding="utf-8") as source:
        for line in source:
            image_and_index, separator, text = line.rstrip("\n").partition("\t")
            if not separator:
                continue
            image_name = image_and_index.rsplit("#", 1)[0]
            cleaned = text.strip()
            if cleaned:
                captions.setdefault(image_name, []).append(cleaned)

    if image_root is None:
        return captions, {}, str(caption_file)

    image_paths = {
        path.name: str(path)
        for path in image_root.glob("*.jpg")
        if path.is_file()
    }
    return captions, image_paths, str(caption_file)


def show_image(source: str | bytes, *, width: int | None = None):
    image_source = io.BytesIO(source) if isinstance(source, bytes) else source
    image = Image.open(image_source)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st.image(image, width="stretch" if width is None else width)


def caption_text(captions: dict[str, list[str]], image_name: str) -> str:
    return " ".join(captions.get(image_name, [])).casefold()


def open_caption_lab(image_name: str):
    st.session_state.selected_image = image_name
    st.session_state.active_page = "Dataset"


def open_training_page():
    st.session_state.active_page = "Train model"


initialize_accounts()
if not st.session_state.get("current_user"):
    spacer, login_column, _ = st.columns([0.7, 1.15, 0.7])
    with login_column:
        st.markdown('<div class="eyebrow">FLICKR8K / LOCAL IMAGE MODEL</div>', unsafe_allow_html=True)
        st.title("Image Caption Generator")
        st.markdown("Create an account or sign in to train a caption model from this project’s Flickr8k folder.")
        sign_in_tab, register_tab = st.tabs(["Sign in", "Register"])
        with sign_in_tab:
            with st.form("sign_in_form"):
                username = st.text_input("Username", key="sign_in_username", autocomplete="username")
                password = st.text_input("Password", type="password", key="sign_in_password", autocomplete="current-password")
                sign_in = st.form_submit_button("Sign in", type="primary", width="stretch")
            if sign_in:
                if authenticate_account(username, password):
                    st.session_state.current_user = username.strip()
                    st.rerun()
                st.error("Username or password is incorrect.")
        with register_tab:
            with st.form("register_form"):
                new_username = st.text_input("Create username", key="register_username", autocomplete="username")
                new_password = st.text_input("Create password", type="password", key="register_password", autocomplete="new-password")
                confirm_password = st.text_input("Confirm password", type="password", key="confirm_password", autocomplete="new-password")
                register = st.form_submit_button("Create account", type="primary", width="stretch")
            if register:
                if new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    try:
                        register_account(new_username, new_password)
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        st.session_state.current_user = new_username.strip()
                        st.rerun()
    st.stop()


captions, image_paths, caption_source = load_project_data()
available_names = sorted(set(captions).intersection(image_paths))
captions = {name: captions[name] for name in available_names}
caption_total = sum(len(captions[name]) for name in available_names)
pages = ["Generate caption", "Train model", "Dataset"]
if st.session_state.get("active_page") not in pages:
    st.session_state.active_page = pages[0]


@st.cache_resource
def cached_caption_engine():
    return load_caption_engine(MODEL_DIR)


model_ready = MODEL_PATH.is_file() and TOKENIZER_PATH.is_file()
model_metadata = {}
if model_ready:
    try:
        import json

        model_metadata = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        model_ready = False

with st.sidebar:
    st.markdown("## Image Caption Generator")
    st.caption(f"Signed in as **{st.session_state.current_user}**")
    page = st.radio(
        "Workspace",
        pages,
        key="active_page",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Local Flickr8k data**")
    if available_names:
        st.markdown(f'<span class="status-pill">{len(available_names):,} images indexed</span>', unsafe_allow_html=True)
        st.caption(f"{caption_total:,} matching captions")
    else:
        st.error("Flickr8k data was not found next to app.py.")
        st.caption("Keep the Flickr8k and Flickr8k_Images folders beside this app.")
    st.markdown("---")
    st.markdown("**Caption model**")
    st.caption("Trained and saved locally" if model_ready else "Train the model to enable captions")
    if st.button("Sign out", width="stretch"):
        st.session_state.pop("current_user", None)
        st.rerun()


if not available_names:
    st.title("Flickr8k is not ready")
    st.info("Place the supplied Flickr8k caption file and image folder beside app.py, then refresh this page.")
    st.stop()


if page == "Generate caption":
    st.markdown('<div class="eyebrow">LOCAL FLICKR8K MODEL</div>', unsafe_allow_html=True)
    st.title("Image Caption Generator")
    st.markdown(
        '<div class="hero-copy">Train on the image-caption pairs in your project folder, then upload a picture and generate a caption with your saved model.</div>',
        unsafe_allow_html=True,
    )
    if model_ready:
        st.markdown(
            f'<span class="status-pill">READY · trained with {model_metadata.get("image_count", 0):,} local images</span>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("No trained caption model yet. Open Train model to train one from the Flickr8k folder.")

    upload_column, action_column = st.columns([1.35, 1], gap="large")
    with upload_column:
        uploaded = st.file_uploader("Choose a picture", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            uploaded_bytes = uploaded.getvalue()
            show_image(uploaded_bytes)
            upload_digest = hashlib.sha256(uploaded_bytes).hexdigest()
        else:
            uploaded_bytes = None
            upload_digest = ""
            st.info("Upload a picture to preview it and generate a caption.")
    with action_column:
        st.markdown("### Generate a caption")
        st.markdown("Upload a photo, then caption it with the model trained on your Flickr8k folder.")
        if not model_ready:
            st.warning("Train the model once on your folder to enable photo captions.")
            if st.button("Train my Flickr8k model", type="primary", width="stretch", on_click=open_training_page):
                st.rerun()
            st.caption("A quick starter run uses 500 folder images and one training epoch. You can choose the full dataset on the training page.")
        generate = st.button(
            "Generate caption",
            type="primary",
            width="stretch",
            disabled=not model_ready or uploaded_bytes is None,
        )
        if generate and uploaded_bytes is not None:
            with st.spinner("Reading the image and generating a caption..."):
                try:
                    decoder, encoder, tokenizer = cached_caption_engine()
                    result = generate_caption(decoder, encoder, tokenizer, uploaded_bytes)
                except Exception as error:
                    st.error(f"Caption generation failed: {error}")
                else:
                    st.session_state.generated_caption = (upload_digest, result)

        saved_caption = st.session_state.get("generated_caption")
        if uploaded_bytes is not None and saved_caption and saved_caption[0] == upload_digest:
            st.markdown(" ")
            st.markdown("### Generated caption")
            st.markdown(
                f'<div class="caption-quote">{html.escape(saved_caption[1])}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Generated by the model trained on this folder’s Flickr8k data.")

elif page == "Train model":
    st.markdown('<div class="eyebrow">LEARN FROM YOUR FOLDER</div>', unsafe_allow_html=True)
    st.title("Train the caption model")
    st.markdown(
        '<div class="hero-copy">The image encoder extracts visual features; the caption decoder learns word sequences from the human captions paired with images in this project.</div>',
        unsafe_allow_html=True,
    )
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Local training images", f"{len(available_names):,}")
    metric_b.metric("Matching captions", f"{caption_total:,}")
    metric_c.metric("Model status", "Ready" if model_ready else "Not trained")
    if model_ready:
        st.success(
            f"Saved model trained on {model_metadata.get('image_count', 0):,} images for "
            f"{model_metadata.get('epochs', 0)} epoch(s). You can upload a picture on Generate caption."
        )
    if len(available_names) < 2:
        st.error("At least two local images with matching captions are needed to train.")
    else:
        image_options = sorted({value for value in (500, 1000, 2000, 4000, 8000, len(available_names)) if 2 <= value <= len(available_names)})
        default_images = min(500, len(available_names))
        selected_default = min(image_options, key=lambda value: abs(value - default_images))
        train_images = st.select_slider(
            "Images to train on",
            options=image_options,
            value=selected_default,
            format_func=lambda value: f"All {value:,}" if value == len(available_names) else f"{value:,} images",
        )
        epochs = st.slider("Training epochs", min_value=1, max_value=10, value=1)
        if train_images == len(available_names):
            st.caption("Using the full local collection takes longer on this CPU-only Windows runtime.")
        if st.button("Train on folder", type="primary", width="stretch"):
            progress_bar = st.progress(0.0, text="Starting local training...")
            progress_text = st.empty()

            def update_training_progress(message: str, fraction: float) -> None:
                progress_bar.progress(min(max(fraction, 0.0), 1.0), text=message)
                progress_text.caption(message)

            try:
                training_summary = train_captioner(
                    captions,
                    image_paths,
                    MODEL_DIR,
                    image_limit=train_images,
                    epochs=epochs,
                    progress_callback=update_training_progress,
                )
            except Exception as error:
                st.error(f"Training failed: {error}")
            else:
                cached_caption_engine.clear()
                st.session_state.training_result = training_summary
                st.rerun()
    if st.session_state.get("training_result"):
        result = st.session_state.pop("training_result")
        st.success(
            f"Training complete. Saved a model from {result['image_count']:,} images and "
            f"{result['caption_count']:,} captions. Open Generate caption to try it."
        )

elif page == "Dataset":
    st.markdown('<div class="eyebrow">LOCAL FLICKR8K FILES</div>', unsafe_allow_html=True)
    st.title("Training data")
    st.markdown(
        '<div class="hero-copy">Inspect the exact local image and human reference captions used to train your caption generator.</div>',
        unsafe_allow_html=True,
    )
    if available_names:
        default_name = st.session_state.get("selected_image", available_names[0])
        if default_name not in available_names:
            default_name = available_names[0]
        selected_name = st.selectbox(
            "Choose an image from the folder",
            available_names,
            index=available_names.index(default_name),
            format_func=lambda name: f"{name} · {len(captions.get(name, []))} captions",
        )
        image_column, caption_column = st.columns([1.25, 1], gap="large")
        with image_column:
            show_image(image_paths[selected_name])
            st.caption(selected_name)
        with caption_column:
            st.markdown("### Human training captions")
            for number, text in enumerate(captions.get(selected_name, []), start=1):
                st.markdown(
                    f'<div class="caption-quote"><span class="eyebrow">{number:02d}</span><br>{html.escape(text)}</div>',
                    unsafe_allow_html=True,
                )
            st.caption("These are training references, not generated predictions.")