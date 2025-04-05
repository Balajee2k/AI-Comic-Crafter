import streamlit as st
import os
import base64
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer
import traceback
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# --- Environment Setup ---
ROOT_DIR = Path(__file__).parent
ENV_PATH = ROOT_DIR / '.env'

load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables")
    st.stop()

genai.configure(api_key=api_key)

# --- Backend Import ---
try:
    from BACKEND import generate_panels, generate_image, process_comic
    BACKEND_AVAILABLE = True
except ImportError:
    st.error("🚨 Failed to import BACKEND modules. Ensure BACKEND.py exists with required functions.")
    BACKEND_AVAILABLE = False
    def generate_panels(prompt, style): return [{"Description": f"Mock Desc {i+1}", "Text": f"Mock Panel {i+1}"} for i in range(6)]
    def generate_image(panel_data, style): return []
    def process_comic(image_paths, panel_texts, output_path): pass

# --- Constants and Setup ---
PANEL_FOLDER = "PANEL_IMAGES"
OUTPUT_FOLDER = "OUTPUT"
BACKGROUND_IMAGE_FILE = "background.png"

os.makedirs(PANEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

STYLE_DESCRIPTIONS = {
    "Manga": "🇯🇵 High-contrast B&W sketch, sharp lines, exaggerated expressions, dramatic shading.",
    "Anime": "🌸 Vibrant colors, cel shading, large expressive eyes, detailed hair, dynamic poses.",
    "American": "💥 Bold outlines, heavy inking, bright saturated colors, exaggerated muscular features.",
    "Belgian": " Tintin-style. Clean lines, soft flat shading, rich detailed backgrounds, semi-realistic.",
}
# --- Function to Add Futuristic Styles ---
def set_futuristic_styles():
    futuristic_styles = '''
    <style>
    /* Import futuristic and comic fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Anime+Ace&display=swap');

    /* Neon gradient background with subtle comic elements */
    .stApp {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
        color: #E0E0E0;
        font-family: 'Anime Ace', sans-serif;
    }

    /* Main content area with a holographic card effect */
    [data-testid="stMain"] {
        background: rgba(26, 26, 46, 0.9);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 0 20px rgba(0, 221, 235, 0.5); /* Electric blue glow */
    }

    /* Futuristic headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00DDEB; /* Electric blue */
        text-shadow: 0 0 10px #00DDEB, 0 0 20px #00DDEB;
    }

    /* Style buttons with neon glow */
    .stButton>button {
        background: linear-gradient(45deg, #FF00FF, #00DDEB); /* Neon pink to electric blue */
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-family: 'Orbitron', sans-serif;
        font-size: 16px;
        box-shadow: 0 0 15px #FF00FF;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #00DDEB, #FF00FF);
        box-shadow: 0 0 25px #00DDEB;
        transform: scale(1.05);
    }

    /* Style text inputs and selectboxes with neon borders */
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background: #2A2A4A;
        color: #E0E0E0;
        border: 2px solid #39FF14; /* Lime green */
        border-radius: 8px;
        box-shadow: 0 0 10px #39FF14;
    }

    /* Style the comic strip image with a glowing frame */
    img {
        border: 3px solid #FF00FF; /* Neon pink */
        border-radius: 10px;
        box-shadow: 0 0 15px #FF00FF, 0 0 30px #FF00FF;
    }

    /* Style captions and info boxes */
    .stCaption, .stInfo {
        font-family: 'Anime Ace', sans-serif;
        color: #39FF14; /* Lime green */
        text-shadow: 0 0 5px #39FF14;
    }

    /* Add a subtle pulse animation to cards */
    @keyframes pulse {
        0% { box-shadow: 0 0 10px #00DDEB; }
        50% { box-shadow: 0 0 20px #FF00FF; }
        100% { box-shadow: 0 0 10px #00DDEB; }
    }

    /* Style for cards (used for input and output sections) */
    .card {
        background: rgba(42, 42, 74, 0.8);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #00DDEB;
        box-shadow: 0 0 15px #00DDEB;
        animation: pulse 3s infinite;
    }
    </style>
    '''
    st.markdown(futuristic_styles, unsafe_allow_html=True)

# Apply the futuristic styles
set_futuristic_styles()

# --- Streamlit UI ---
st.title("ComicCrafter AI: Transform Your Stories into Visual Adventures 🎨")
st.caption("Where AI Meets Artistry: Your Stories, Our Panels, Infinite Possibilities!")
st.divider()

# Input section with card styling
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("1. Enter Your Story Idea")
    st.write("📝 **Story Prompt:**")
    user_prompt = st.text_area("", "", height=150, placeholder="e.g., A forgetful wizard accidentally turns himself into different animals while trying to find his lost spellbook in his messy cottage.")

with col2:
    st.subheader("2. Choose Style")
    st.write("🎨 **Art Style:**")
    art_style = st.selectbox("", list(STYLE_DESCRIPTIONS.keys()))
    st.markdown(f"**Style Preview:**")
    st.info(f"{STYLE_DESCRIPTIONS[art_style]}", icon="🖌️")
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.subheader("3. Generate!")
generate_button = st.button("🚀 Generate Comic Strip", use_container_width=True)

# --- Processing Logic ---
if generate_button:
    if not user_prompt:
        st.warning("⚠️ Please enter a story prompt before generating.", icon="✍️")
        st.stop()

    if not BACKEND_AVAILABLE:
        st.error("🚨 Backend functions are unavailable. Cannot generate comic.")
        st.stop()

    panel_data = None
    image_paths = []
    panel_texts = []
    story_data = None
    success = False

    try:
        # --- Step 1: Generate Story ---
        with st.spinner("⏳ Step 1/4: Generating story..."):
            story_data = generate_panels.generate_story(user_prompt, art_style)

        # --- Step 2: Generate Panels ---
        with st.spinner("⏳ Step 2/4: Generating panel descriptions & dialogues..."):
            panel_data = generate_panels.generate_panels(user_prompt, art_style)

        if not isinstance(panel_data, list) or len(panel_data) != 6:
            st.error(f"❌ Error: Expected 6 panel descriptions, but received {len(panel_data) if isinstance(panel_data, list) else 'invalid data'} from `generate_panels`.")
            st.write("Received Data:", panel_data)
            st.stop()
        if not all(isinstance(p, dict) and "Text" in p and "Description" in p for p in panel_data):
            st.error(f"❌ Error: Panel data structure from `generate_panels` is incorrect.")
            st.write("Received Data:", panel_data)
            st.stop()

        panel_texts = [panel["Text"] for panel in panel_data]

        # --- Step 3: Generate Images ---
        with st.spinner("⏳ Step 3/4: Generating images for comic panels... (This may take a while!)"):
            image_paths_iterable = generate_image.generate_images(panel_data, art_style)
            image_paths = list(image_paths_iterable)

        paths_ok = False
        if len(image_paths) == 6:
            all_valid = True
            for i, img_path in enumerate(image_paths):
                is_str = isinstance(img_path, str)
                exists = os.path.exists(img_path) if is_str else False
                if not is_str or not exists:
                    all_valid = False
                    st.warning(f"⚠️ Problem with generated path/file for panel {i+1}: '{img_path}'")
                    break
            if all_valid:
                paths_ok = True
        else:
            st.warning(f"⚠️ Expected 6 image paths, but received {len(image_paths)}.")

        # --- Step 4: Assemble Comic and Create PDF ---
        if paths_ok:
            output_image_path = os.path.join(OUTPUT_FOLDER, "comic_strip_with_text.png")
            pdf_output_path = os.path.join(OUTPUT_FOLDER, "comic_strip.pdf")

            with st.spinner("⏳ Step 4/4: Assembling comic strip and PDF..."):
                process_comic.create_comic_strip_with_text(image_paths, panel_texts, output_image_path)

                def create_pdf(image_path, pdf_output_path):
                    if not os.path.exists(image_path):
                        st.error(f"❌ Cannot create PDF: Final comic image '{image_path}' not found.")
                        return False
                    try:
                        doc = SimpleDocTemplate(pdf_output_path, pagesize=A4, leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=30)
                        with Image.open(image_path) as img_pil:
                            img_width, img_height = img_pil.size
                        aspect = img_height / float(img_width)
                        page_width, page_height = A4
                        display_width = page_width * 0.9
                        display_height = display_width * aspect
                        if display_height > page_height * 0.9:
                            display_height = page_height * 0.9
                            display_width = display_height / aspect
                        rl_img = RLImage(image_path, width=display_width, height=display_height)
                        rl_img.hAlign = 'CENTER'
                        spacer = Spacer(1, 20)
                        story = [rl_img, spacer]
                        doc.build(story)
                        return True
                    except Exception as pdf_err:
                        st.error(f"❌ Failed to create PDF: {pdf_err}")
                        print(traceback.format_exc())
                        return False

                pdf_created = create_pdf(output_image_path, pdf_output_path)

            # --- Display Results ---
            if os.path.exists(output_image_path):
                st.subheader("🎉 Your Story and Comic Strip!")
                col1, col2 = st.columns([1, 1])
                with col1:
                    # Style the story section with a holographic card look
                    if 'title' in story_data:
                        st.markdown(
                            f"""
                            <div class="card">
                                <h2 style='font-family: "Orbitron", sans-serif; color: #FF00FF; text-shadow: 0 0 10px #FF00FF;'>{story_data['title']}</h2>
                                <p style='color: #E0E0E0;'><strong style='color: #39FF14;'>Introduction:</strong> {story_data['introduction']}</p>
                                <p style='color: #E0E0E0;'><strong style='color: #39FF14;'>Storyline:</strong> {story_data['storyline']}</p>
                                <p style='color: #E0E0E0;'><strong style='color: #39FF14;'>Climax:</strong> {story_data['climax']}</p>
                                <p style='color: #E0E0E0;'><strong style='color: #39FF14;'>Moral:</strong> {story_data['moral']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("**Title:** N/A")
                    for section in ["introduction", "storyline", "climax", "moral"]:
                        if section in story_data:
                            st.write(f"**{section.capitalize()}:** {story_data[section]}")
                        else:
                            st.write(f"**{section.capitalize()}:** N/A")
                with col2:
                    st.image(output_image_path, caption="Your Generated Comic Strip", use_container_width=True)
                    st.success("✅ Comic generated successfully!")
                    success = True

                    if pdf_created and os.path.exists(pdf_output_path):
                        with open(pdf_output_path, "rb") as pdf_file:
                            st.download_button(
                                label="⬇️ Download Comic as PDF",
                                data=pdf_file,
                                file_name="comic_strip.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
            else:
                st.error("❌ Error: Final comic image file was not found.")
        else:
            st.error("❌ Something went wrong! Image generation failed or produced invalid/missing files.")

    except Exception as e:
        st.error(f"💥 An unexpected error occurred during the generation process!")
        with st.expander("Show Error Details"):
            st.error(f"{e}")
            st.code(traceback.format_exc())
        print(traceback.format_exc())

st.divider()
st.caption("ComicCrafter AI - Powered by Clipdrop & ReportLab")