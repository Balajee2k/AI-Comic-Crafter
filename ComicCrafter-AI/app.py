"""
ComicCrafter AI - Enhanced Edition
===================================
Research-grade AI Comic Generation System

Features:
- Character upload and reference image support
- Curated multi-style comic tradition selection (stable core styles)
- Emotion-driven dynamic panel layouts
- Character consistency tracking
- Quality evaluation metrics
- Professional PDF export

Research Contributions:
1. Character Consistency Engine with embedding-based tracking
2. Emotion-Driven Layout System with comic grammar
3. Multi-Cultural Style Adaptation across 15+ traditions
3. Style Stability Optimization with a curated high-performing style set
4. Quantitative Comic Quality Evaluation Framework
"""

import streamlit as st
import os
import base64
import json
from copy import deepcopy
from datetime import datetime
from PIL import Image
import traceback
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from io import BytesIO

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
    from BACKEND.consistency import CharacterConsistencyEngine
    from BACKEND.emotion import EmotionAnalyzer, DynamicLayoutEngine
    from BACKEND.evaluation import ComicEvaluator
    from BACKEND.styles import StyleAdapter, ComicTradition
    BACKEND_AVAILABLE = True
    ENHANCED_BACKEND = True
except ImportError as e:
    print(f"Import warning: {e}")
    try:
        from BACKEND import generate_panels, generate_image, process_comic
        BACKEND_AVAILABLE = True
        ENHANCED_BACKEND = False
    except ImportError:
        st.error("🚨 Failed to import BACKEND modules. Ensure BACKEND folder exists with required files.")
        BACKEND_AVAILABLE = False
        ENHANCED_BACKEND = False

# Import Regional LoRA module (v2.0 feature)
try:
    from BACKEND.regional_lora import (
        RegionalLoRAController, 
        RegionalConfig,
        BoundingBoxExtractor,
        LoRAManager,
        PIPELINE_AVAILABLE
    )
    REGIONAL_LORA_AVAILABLE = True
    print("✅ Regional LoRA module loaded (v2.0)")
except ImportError as e:
    print(f"Regional LoRA not available: {e}")
    REGIONAL_LORA_AVAILABLE = False
    PIPELINE_AVAILABLE = False

# --- Constants and Setup ---
PANEL_FOLDER = "PANEL_IMAGES"
OUTPUT_FOLDER = "OUTPUT"
REFERENCE_FOLDER = "REFERENCE_IMAGES"
BACKGROUND_IMAGE_FILE = "background.png"

os.makedirs(PANEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REFERENCE_FOLDER, exist_ok=True)

# Enhanced style descriptions with traditions
STYLE_TRADITIONS = {
    "Manga (Japanese)": {
        "tradition": "MANGA",
        "icon": "🇯🇵",
        "description": "High-contrast line art with screentones, expressive framing, and strong panel readability"
    },
    "Anime Style": {
        "tradition": "ANIME",
        "icon": "🌸",
        "description": "Vibrant cel shading, expressive characters, and stable visual continuity across panels"
    },
    "American Superhero": {
        "tradition": "AMERICAN_SUPERHERO",
        "icon": "💥",
        "description": "Bold inking, dynamic poses, dramatic perspective, and high-impact action composition"
    },
    "Franco-Belgian (Tintin)": {
        "tradition": "FRANCO_BELGIAN",
        "icon": "🎨",
        "description": "Clear ligne-claire linework, detailed backgrounds, and balanced storytelling compositions"
    },
    "Webtoon (Vertical)": {
        "tradition": "WEBTOON",
        "icon": "📱",
        "description": "Modern digital comic rendering with soft shading and clean scene-to-scene continuity"
    },
    "Film Noir": {
        "tradition": "NOIR",
        "icon": "🎬",
        "description": "Cinematic high-contrast mood, shadow-driven storytelling, and dramatic visual tension"
    }
}

# Panel count options
PANEL_COUNT_OPTIONS = {
    "4 Panels (Quick Story)": 4,
    "6 Panels (Standard)": 6,
    "8 Panels (Extended)": 8,
    "9 Panels (3x3 Grid)": 9,
    "12 Panels (Full Page)": 12
}


def set_enhanced_styles():
    """Apply enhanced futuristic comic-themed CSS styles."""
    styles = '''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Bangers&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0a2a4a 100%);
        color: #E0E0E0;
    }

    [data-testid="stMain"] {
        background: rgba(10, 10, 30, 0.95);
        border-radius: 15px;
        padding: 30px;
    }

    h1 {
        font-family: 'Bangers', cursive;
        color: #FFD700;
        text-shadow: 3px 3px 0 #FF4500, 6px 6px 0 #000;
        font-size: 3em;
        letter-spacing: 3px;
    }

    h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00DDEB;
        text-shadow: 0 0 10px #00DDEB;
    }

    .stButton>button {
        background: linear-gradient(45deg, #FF00FF, #00DDEB);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-family: 'Orbitron', sans-serif;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 0 20px #FF00FF;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(45deg, #00DDEB, #FF00FF);
        box-shadow: 0 0 30px #00DDEB;
        transform: scale(1.05);
    }

    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background: #1a1a3a;
        color: #E0E0E0;
        border: 2px solid #39FF14;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(57, 255, 20, 0.3);
    }

    .stSlider > div > div {
        background: linear-gradient(90deg, #FF00FF, #00DDEB);
    }

    img {
        border: 3px solid #FF00FF;
        border-radius: 10px;
        box-shadow: 0 0 15px #FF00FF, 0 0 30px rgba(255, 0, 255, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 0, 255, 0.2);
        border-radius: 8px;
        color: #E0E0E0;
        font-family: 'Orbitron', sans-serif;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #FF00FF, #00DDEB);
        color: white;
    }

    .feature-card {
        background: rgba(42, 42, 74, 0.8);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #00DDEB;
        box-shadow: 0 0 15px rgba(0, 221, 235, 0.3);
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(255, 0, 255, 0.1), rgba(0, 221, 235, 0.1));
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid #39FF14;
    }

    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2em;
        color: #39FF14;
        text-shadow: 0 0 10px #39FF14;
    }

    .metric-label {
        font-family: 'Comic Neue', cursive;
        color: #E0E0E0;
        font-size: 0.9em;
    }

    .story-card {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.9), rgba(42, 42, 74, 0.9));
        border-radius: 15px;
        padding: 25px;
        border: 2px solid #FF00FF;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
    }

    .upload-zone {
        border: 2px dashed #00DDEB;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        background: rgba(0, 221, 235, 0.05);
        transition: all 0.3s ease;
    }

    .upload-zone:hover {
        border-color: #FF00FF;
        background: rgba(255, 0, 255, 0.05);
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 15px #00DDEB; }
        50% { box-shadow: 0 0 30px #FF00FF; }
    }

    .generating {
        animation: pulse-glow 1.5s infinite;
    }

    .grade-a { color: #39FF14; }
    .grade-b { color: #00DDEB; }
    .grade-c { color: #FFD700; }
    .grade-d { color: #FF4500; }
    .grade-f { color: #FF0000; }
    </style>
    '''
    st.markdown(styles, unsafe_allow_html=True)


def save_uploaded_image(uploaded_file, folder: str, filename: str) -> str:
    """Save uploaded image and return the path."""
    if uploaded_file is None:
        return None
    
    filepath = os.path.join(folder, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath


def display_metrics(metrics: dict):
    """Display evaluation metrics in a visually appealing way."""
    if not metrics:
        return
    
    st.markdown("### 📊 Quality Evaluation Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    overall = metrics.get('overall_score', 0)
    grade = metrics.get('grade', 'N/A')
    
    # Determine grade color
    grade_class = "grade-c"
    if grade.startswith('A'):
        grade_class = "grade-a"
    elif grade.startswith('B'):
        grade_class = "grade-b"
    elif grade.startswith('D'):
        grade_class = "grade-d"
    elif grade == 'F':
        grade_class = "grade-f"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{overall:.1%}</div>
            <div class="metric-label">Overall Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {grade_class}">{grade}</div>
            <div class="metric-label">Grade</div>
        </div>
        """, unsafe_allow_html=True)
    
    detailed = metrics.get('detailed_metrics', {})
    
    with col3:
        narrative = detailed.get('narrative_flow_score', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{narrative:.1%}</div>
            <div class="metric-label">Narrative Flow</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        emotion = detailed.get('emotion_conveyance_score', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{emotion:.1%}</div>
            <div class="metric-label">Emotion</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show improvement suggestions
    suggestions = metrics.get('suggestions', [])
    if suggestions:
        with st.expander("💡 Improvement Suggestions"):
            for suggestion in suggestions[:5]:
                st.write(f"• {suggestion}")


def display_emotional_arc(panel_data: list):
    """Display emotional arc visualization."""
    if not panel_data:
        return
    
    st.markdown("### 🎭 Emotional Arc")
    
    emotions = [p.get('emotion', 'neutral') for p in panel_data]
    intensities = [p.get('emotion_intensity', p.get('intensity', 0.5)) for p in panel_data]
    
    # Simple text-based visualization
    arc_display = " → ".join([f"{e.upper()}" for e in emotions])
    st.markdown(f"**Emotion Flow:** {arc_display}")
    
    # Intensity bar
    intensity_avg = sum(intensities) / len(intensities) if intensities else 0
    st.progress(intensity_avg, text=f"Average Intensity: {intensity_avg:.0%}")


def init_session_state():
    """Initialize app-level session state used for iterative workflows."""
    defaults = {
        "comic_memory": None,
        "user_prompt_input": "",
        "story_genre_input": "Auto-detect",
        "story_mood_input": "Auto-detect",
        "target_audience_input": "General",
        "surprise_characters": "",
        "edit_request_input": "",
        "pending_surprise_seed": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_pending_surprise_seed():
    """Apply surprise story values before widgets are instantiated in this run."""
    seed = st.session_state.get("pending_surprise_seed")
    if not seed:
        return

    st.session_state.user_prompt_input = seed.get("prompt", "")
    st.session_state.story_genre_input = seed.get("genre", "Adventure")
    st.session_state.story_mood_input = seed.get("mood", "Exciting")
    st.session_state.target_audience_input = seed.get("audience", "General")
    st.session_state.surprise_characters = seed.get("characters", "")
    st.session_state.pending_surprise_seed = None


def build_story_prompt(base_prompt: str, genre: str, mood: str, audience: str, extra_characters: str) -> str:
    """Construct a richer story prompt while keeping the base user input intact."""
    details = []
    if genre and genre != "Auto-detect":
        details.append(f"Genre: {genre}")
    if mood and mood != "Auto-detect":
        details.append(f"Mood: {mood}")
    if audience:
        details.append(f"Target audience: {audience}")
    if extra_characters.strip():
        details.append(f"Character notes: {extra_characters.strip()}")

    if not details:
        return base_prompt

    return f"{base_prompt}\n\nAdditional constraints:\n- " + "\n- ".join(details)


def create_image_generator(char_image_path: str,
                           ref_image_path: str,
                           character_description: str,
                           character_name: str):
    """Create and preconfigure the enhanced image generator."""
    img_generator = generate_image.EnhancedImageGenerator()

    if char_image_path:
        img_generator.set_reference_image(char_image_path, "main_character")
    if ref_image_path:
        img_generator.set_style_reference(ref_image_path)
    if character_description:
        img_generator.set_character_description(character_description)
    if character_name and character_description:
        img_generator.set_character_description(character_description, character_name)

    return img_generator


def ensure_panel_images(img_generator,
                        panel_data: list,
                        image_paths: list,
                        art_style: str,
                        num_panels: int) -> list:
    """
    Ensure we end with exactly num_panels valid image files.

    Strategy:
    1) Keep valid generated paths.
    2) Regenerate only missing panels.
    3) If regeneration still fails, create placeholder image for that panel.
    """
    normalized_paths = [None] * num_panels

    for idx in range(min(len(image_paths), num_panels)):
        candidate = image_paths[idx]
        if isinstance(candidate, str) and os.path.exists(candidate):
            normalized_paths[idx] = candidate

    for idx in range(num_panels):
        if normalized_paths[idx]:
            continue

        try:
            panel = panel_data[idx] if idx < len(panel_data) else {
                "Description": "Story continues.",
                "Text": "...",
                "emotion": "neutral"
            }
            target_path = os.path.join(PANEL_FOLDER, f"panel_{idx+1:02d}.png")
            regenerated = img_generator.generate_single_panel(
                panel_data=panel,
                art_style=art_style,
                panel_number=idx + 1,
                output_path=target_path,
                use_enhanced=True
            )
            if regenerated and os.path.exists(regenerated):
                normalized_paths[idx] = regenerated
                continue
        except Exception as e:
            print(f"Panel {idx + 1} targeted regeneration failed: {e}")

        # Final fallback: create placeholder so comic assembly can continue.
        placeholder_path = os.path.join(PANEL_FOLDER, f"panel_{idx+1:02d}_placeholder.png")
        try:
            Image.new("RGB", (1024, 1024), color=(245, 245, 245)).save(placeholder_path, format="PNG")
            normalized_paths[idx] = placeholder_path
        except Exception as e:
            print(f"Failed to create placeholder for panel {idx + 1}: {e}")

    return normalized_paths


def save_comic_memory(payload: dict):
    """Persist the latest generated comic in session memory for iterative edits."""
    data = deepcopy(payload)
    data["updated_at"] = datetime.utcnow().isoformat()
    st.session_state.comic_memory = data


def render_session_comic(memory: dict):
    """Render current in-session comic snapshot."""
    if not memory:
        return

    st.subheader("🧠 Current Session Comic")
    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.caption(f"Title: {memory.get('story_data', {}).get('title', 'My Comic')}")
    with meta_cols[1]:
        st.caption(f"Style: {memory.get('art_style', 'Anime Style')}")
    with meta_cols[2]:
        st.caption(f"Panels: {len(memory.get('panel_data', []))}")

    output_image_path = memory.get("output_image_path")
    if output_image_path and os.path.exists(output_image_path):
        st.image(output_image_path, caption="Latest Comic (Session Memory)", use_container_width=True)


def regenerate_comic_outputs(memory: dict, use_dynamic_layout: bool, evaluate: bool = True) -> dict:
    """Rebuild combined comic outputs (PNG/PDF) from stored session memory."""
    panel_data = memory.get("panel_data", [])
    image_paths = memory.get("image_paths", [])
    panel_texts = [panel.get("Text", "...") for panel in panel_data]

    output_image_path = memory.get("output_image_path", os.path.join(OUTPUT_FOLDER, "comic_strip_with_text.png"))
    pdf_output_path = memory.get("pdf_output_path", os.path.join(OUTPUT_FOLDER, "comic_strip.pdf"))

    process_comic.create_comic_strip_with_text(
        image_paths,
        panel_texts,
        output_image_path,
        panel_data=panel_data if use_dynamic_layout else None,
        title=memory.get("story_data", {}).get("title", "My Comic"),
        use_dynamic_layout=use_dynamic_layout
    )

    pdf_created = process_comic.create_comic_pdf(
        output_image_path,
        pdf_output_path,
        story_data=memory.get("story_data", {}),
        panel_data=panel_data,
        evaluation_metrics=None
    )

    evaluation_metrics = None
    if evaluate and ENHANCED_BACKEND:
        try:
            evaluator = ComicEvaluator()
            evaluation_metrics = evaluator.evaluate(
                panel_data=panel_data,
                image_paths=image_paths,
                story_data=memory.get("story_data", {})
            )
        except Exception as e:
            print(f"Evaluation after edit failed: {e}")

    memory["panel_texts"] = panel_texts
    memory["output_image_path"] = output_image_path
    memory["pdf_output_path"] = pdf_output_path
    memory["pdf_created"] = pdf_created
    memory["evaluation_metrics"] = evaluation_metrics
    return memory


# Apply styles
set_enhanced_styles()
init_session_state()
apply_pending_surprise_seed()

# --- Main UI ---
st.title("🎨 ComicCrafter AI")
st.caption("Research-Grade AI Comic Generation • Character Consistency • Emotion-Driven Layouts • Multi-Cultural Styles")
st.divider()

# --- Sidebar for Advanced Options ---
with st.sidebar:
    st.header("⚙️ Advanced Options")
    
    # Panel count
    st.subheader("📐 Layout Settings")
    panel_count_option = st.selectbox(
        "Number of Panels",
        list(PANEL_COUNT_OPTIONS.keys()),
        index=1  # Default to 6 panels
    )
    num_panels = PANEL_COUNT_OPTIONS[panel_count_option]
    
    # Dynamic layout toggle
    use_dynamic_layout = st.checkbox(
        "Use Emotion-Based Layout",
        value=True,
        help="Adjusts panel sizes based on emotional intensity"
    )
    
    st.divider()
    
    # Reference images section
    st.subheader("🖼️ Reference Images")
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    
    reference_image = st.file_uploader(
        "Upload Reference Style/Theme",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to guide the art style"
    )
    
    character_image = st.file_uploader(
        "Upload Character Reference",
        type=['png', 'jpg', 'jpeg'],
        help="Upload character design for consistency"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if reference_image:
        st.image(reference_image, caption="Style Reference", width=150)
    
    if character_image:
        st.image(character_image, caption="Character Reference", width=150)
    
    st.divider()
    
    # Character description
    st.subheader("👤 Character Details")
    character_name = st.text_input(
        "Main Character Name",
        placeholder="e.g., Luna the Wizard"
    )
    
    character_description = st.text_area(
        "Character Description",
        placeholder="e.g., Young woman with silver hair, blue robes, carries a wooden staff with a glowing crystal",
        height=100
    )
    
    # Secondary character support
    with st.expander("➕ Add More Characters"):
        character2_name = st.text_input("Character 2 Name", placeholder="e.g., Tom")
        character2_desc = st.text_area("Character 2 Description", placeholder="e.g., Tall man with brown hair, glasses, wearing a lab coat", height=80)
        character2_image = st.file_uploader("Character 2 Reference", type=['png', 'jpg', 'jpeg'], key="char2_img")
        
        character3_name = st.text_input("Character 3 Name", placeholder="e.g., Mia")
        character3_desc = st.text_area("Character 3 Description", placeholder="e.g., Young girl with pigtails, cheerful smile", height=80)
        character3_image = st.file_uploader("Character 3 Reference", type=['png', 'jpg', 'jpeg'], key="char3_img")
    
    st.divider()
    
    # Regional LoRA v2.0 Controls
    st.subheader("🧬 v2.0: Regional LoRA")
    
    if REGIONAL_LORA_AVAILABLE:
        use_regional_lora = st.checkbox(
            "Enable Regional LoRA",
            value=True,
            help="Apply character-specific LoRAs to spatial regions. Solves multi-character interference."
        )
        
        if use_regional_lora:
            lora_backend = st.selectbox(
                "LoRA Backend",
                ["replicate", "local", "clipdrop_fallback"],
                help="Replicate: Cloud (no GPU needed). Local: Requires GPU + diffusers."
            )
            
            lora_method = st.selectbox(
                "Character LoRA Method",
                ["instant (PuLID/PhotoMaker)", "trained (Full LoRA)"],
                help="Instant: Fast, single image. Trained: 15-30 min, better quality."
            )
            
            show_layout_preview = st.checkbox(
                "Show Bounding Box Preview",
                value=False,
                help="Visualize character regions before generation"
            )
            
            if st.button("📊 View LoRA Status"):
                if 'regional_controller' in st.session_state:
                    metrics = st.session_state.regional_controller.get_metrics()
                    st.json(metrics)
                else:
                    st.info("No generation session yet")
    else:
        st.warning("⚠️ Regional LoRA not available")
        st.caption("Install: `pip install replicate torch diffusers`")
        use_regional_lora = False
        lora_backend = "clipdrop_fallback"
        lora_method = "instant (PuLID/PhotoMaker)"
        show_layout_preview = False
    
    st.divider()
    
    # Show features status
    st.subheader("📦 Features Status")
    if ENHANCED_BACKEND:
        st.success("✅ All Enhanced Features Active")
        st.markdown("""
        - Character Consistency Engine
        - Emotion Analysis
        - Quality Evaluation
        - Multi-Cultural Styles
        """)
        if REGIONAL_LORA_AVAILABLE:
            st.success("✅ Regional LoRA v2.0 Ready")
            st.markdown("""
            - Dynamic Regional Injection
            - Multi-Character LoRA
            - Bounding Box Extraction
            """)
    else:
        st.warning("⚠️ Basic Mode Only")
        st.markdown("Enhanced features unavailable")

# --- Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["📝 Create Comic", "📖 Gallery", "📊 About"])

with tab1:
    # Story input section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("1. 📝 Your Story Idea")

        surprise_col1, surprise_col2 = st.columns([1.3, 1])
        with surprise_col1:
            st.caption("Need inspiration? Auto-generate a complete story and edit before rendering.")
        with surprise_col2:
            surprise_clicked = st.button("🎲 Surprise Me", use_container_width=True)

        # Optional story details
        with st.expander("📌 Additional Story Details (Optional)"):
            story_genre = st.selectbox(
                "Genre",
                ["Auto-detect", "Comedy", "Adventure", "Drama", "Horror", "Romance", "Sci-Fi", "Fantasy", "Slice of Life"],
                key="story_genre_input"
            )
            story_mood = st.selectbox(
                "Overall Mood",
                ["Auto-detect", "Light-hearted", "Serious", "Dramatic", "Mysterious", "Exciting", "Emotional"],
                key="story_mood_input"
            )
            target_audience = st.selectbox(
                "Target Audience",
                ["General", "Children", "Young Adult", "Adult", "All Ages"],
                key="target_audience_input"
            )
            st.text_area(
                "Auto-generated Character Notes",
                height=80,
                key="surprise_characters",
                placeholder="Character notes appear here when using Surprise Me"
            )

        user_prompt = st.text_area(
            "Story Prompt",
            height=150,
            key="user_prompt_input",
            placeholder="Enter your story idea here...\n\ne.g., A forgetful wizard accidentally turns himself into different animals while trying to find his lost spellbook in his messy cottage. Each transformation leads to a humorous situation until he finally finds the book.",
            label_visibility="collapsed"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("2. 🎨 Art Style")

        selected_style = st.selectbox(
            "Art Style",
            list(STYLE_TRADITIONS.keys()),
            label_visibility="collapsed"
        )

        style_info = STYLE_TRADITIONS[selected_style]
        st.markdown(f"""
        **{style_info['icon']} {selected_style}**

        {style_info['description']}
        """)

        art_style = selected_style
        st.markdown('</div>', unsafe_allow_html=True)

    if surprise_clicked:
        if not BACKEND_AVAILABLE:
            st.error("🚨 Backend functions are unavailable. Cannot generate a surprise story.")
        else:
            try:
                with st.spinner("Creating a surprise story seed..."):
                    seed = generate_panels.generate_surprise_story_seed(art_style, num_panels)
                st.session_state.pending_surprise_seed = seed
                st.success("✨ Surprise story generated. Review or edit it, then click Generate Comic Strip.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to generate surprise story: {e}")

    st.divider()

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        generate_button = st.button(
            "🚀 Generate Comic Strip",
            use_container_width=True,
            type="primary"
        )

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
        evaluation_metrics = None

        ref_image_path = None
        char_image_path = None

        if reference_image:
            ref_image_path = save_uploaded_image(reference_image, REFERENCE_FOLDER, "style_reference.png")

        if character_image:
            char_image_path = save_uploaded_image(character_image, REFERENCE_FOLDER, "character_reference.png")

        try:
            progress_bar = st.progress(0, text="Starting comic generation...")

            full_story_prompt = build_story_prompt(
                user_prompt,
                st.session_state.story_genre_input,
                st.session_state.story_mood_input,
                st.session_state.target_audience_input,
                st.session_state.surprise_characters
            )

            progress_bar.progress(10, text="⏳ Step 1/5: Generating story...")
            with st.spinner(""):
                story_data = generate_panels.generate_story(full_story_prompt, art_style)
            progress_bar.progress(20, text="✅ Story generated!")

            progress_bar.progress(25, text=f"⏳ Step 2/5: Generating {num_panels} panel descriptions...")
            with st.spinner(""):
                panel_data = generate_panels.generate_panels(full_story_prompt, art_style, num_panels)

            if not isinstance(panel_data, list):
                st.error("❌ Error: Panel data is invalid.")
                st.stop()

            if len(panel_data) > num_panels:
                panel_data = panel_data[:num_panels]
            elif len(panel_data) < num_panels:
                while len(panel_data) < num_panels:
                    panel_data.append({
                        "Description": "Story continues with the characters.",
                        "Text": "...",
                        "emotion": "neutral",
                        "relative_size": "medium",
                        "emotion_intensity": 0.5
                    })

            panel_texts = [panel.get("Text", "...") for panel in panel_data]
            progress_bar.progress(40, text=f"✅ {len(panel_data)} panels generated!")

            progress_bar.progress(45, text="⏳ Step 3/5: Generating panel images... (This may take a while!)")
            img_generator = None
            with st.spinner(""):
                use_regional = (
                    REGIONAL_LORA_AVAILABLE and
                    use_regional_lora and
                    character_image is not None
                )

                if use_regional:
                    st.info("🧬 Using Regional LoRA v2.0 for character consistency...")
                    try:
                        regional_config = RegionalConfig(
                            image_width=1024,
                            image_height=1024,
                            backend=lora_backend.split()[0] if ' ' in lora_backend else lora_backend,
                            num_inference_steps=28,
                            guidance_scale=7.5
                        )

                        controller = RegionalLoRAController(config=regional_config, lora_cache_dir="LORA_CACHE")
                        st.session_state.regional_controller = controller

                        if character_name and character_description and char_image_path:
                            lora_method_clean = "instant" if "instant" in lora_method else "trained"
                            controller.register_character(
                                name=character_name or "MainCharacter",
                                description=character_description or "The main character",
                                reference_image=char_image_path,
                                train_lora=True,
                                lora_method=lora_method_clean
                            )

                        all_characters = [c for c in [character_name, character2_name, character3_name] if c]

                        if character2_name and character2_desc:
                            char2_path = save_uploaded_image(character2_image, REFERENCE_FOLDER, "character2_reference.png") if character2_image else None
                            controller.register_character(
                                name=character2_name,
                                description=character2_desc,
                                reference_image=char2_path,
                                train_lora=char2_path is not None,
                                lora_method="instant"
                            )

                        if character3_name and character3_desc:
                            char3_path = save_uploaded_image(character3_image, REFERENCE_FOLDER, "character3_reference.png") if character3_image else None
                            controller.register_character(
                                name=character3_name,
                                description=character3_desc,
                                reference_image=char3_path,
                                train_lora=char3_path is not None,
                                lora_method="instant"
                            )

                        image_paths = []
                        for i, panel in enumerate(panel_data):
                            progress_pct = 45 + int((i / len(panel_data)) * 30)
                            progress_bar.progress(progress_pct, text=f"⏳ Generating panel {i+1}/{len(panel_data)}...")

                            panel_chars = panel.get("characters", all_characters) or all_characters
                            result = controller.generate_panel(
                                description=panel.get("Description", ""),
                                characters=panel_chars,
                                art_style=art_style,
                                panel_number=i + 1,
                                emotion=panel.get("emotion", "neutral")
                            )

                            output_path = os.path.join(PANEL_FOLDER, f"panel_{i+1:02d}.png")
                            result.image.save(output_path)
                            image_paths.append(output_path)

                            if show_layout_preview and i == 0:
                                layout_preview = controller.visualize_layout(result, show_masks=True)
                                st.image(layout_preview, caption=f"Panel {i+1} Layout + Masks", width=400)

                    except Exception as e:
                        st.warning(f"⚠️ Regional LoRA failed: {e}. Falling back to standard generation.")
                        print(f"Regional LoRA error: {e}")
                        use_regional = False

                if not use_regional:
                    img_generator = create_image_generator(
                        char_image_path=char_image_path,
                        ref_image_path=ref_image_path,
                        character_description=character_description,
                        character_name=character_name
                    )
                    image_paths = list(img_generator.generate_images(panel_data, art_style, use_enhanced=True))

            # If some panels fail from API errors/timeouts, recover them individually.
            if len(image_paths) < num_panels or any(
                not isinstance(path, str) or not os.path.exists(path)
                for path in image_paths[:num_panels]
            ):
                if img_generator is None:
                    img_generator = create_image_generator(
                        char_image_path=char_image_path,
                        ref_image_path=ref_image_path,
                        character_description=character_description,
                        character_name=character_name
                    )
                image_paths = ensure_panel_images(
                    img_generator=img_generator,
                    panel_data=panel_data,
                    image_paths=image_paths,
                    art_style=art_style,
                    num_panels=num_panels
                )

            progress_bar.progress(75, text="✅ Images generated!")

            image_paths = image_paths[:num_panels]
            for i, img_path in enumerate(image_paths):
                if not isinstance(img_path, str) or not os.path.exists(img_path):
                    st.error(f"❌ Invalid generated image for panel {i+1}: {img_path}")
                    st.stop()

            output_image_path = os.path.join(OUTPUT_FOLDER, "comic_strip_with_text.png")
            pdf_output_path = os.path.join(OUTPUT_FOLDER, "comic_strip.pdf")

            progress_bar.progress(80, text="⏳ Step 4/5: Assembling comic strip...")
            with st.spinner(""):
                process_comic.create_comic_strip_with_text(
                    image_paths,
                    panel_texts,
                    output_image_path,
                    panel_data=panel_data if use_dynamic_layout else None,
                    title=story_data.get('title', 'My Comic') if story_data else 'My Comic',
                    use_dynamic_layout=use_dynamic_layout
                )

            progress_bar.progress(92, text="⏳ Step 5/5: Creating PDF and evaluating quality...")
            with st.spinner(""):
                pdf_created = process_comic.create_comic_pdf(
                    output_image_path,
                    pdf_output_path,
                    story_data=story_data,
                    panel_data=panel_data,
                    evaluation_metrics=None
                )

                if ENHANCED_BACKEND:
                    try:
                        evaluator = ComicEvaluator()
                        evaluation_metrics = evaluator.evaluate(
                            panel_data=panel_data,
                            image_paths=image_paths,
                            story_data=story_data
                        )
                    except Exception as e:
                        print(f"Evaluation failed: {e}")

            progress_bar.progress(100, text="✅ Complete!")

            memory = {
                "story_prompt": full_story_prompt,
                "story_data": story_data,
                "panel_data": panel_data,
                "panel_texts": panel_texts,
                "image_paths": image_paths,
                "art_style": art_style,
                "num_panels": num_panels,
                "reference_image_path": ref_image_path,
                "character_image_path": char_image_path,
                "character_name": character_name,
                "character_description": character_description,
                "use_dynamic_layout": use_dynamic_layout,
                "output_image_path": output_image_path,
                "pdf_output_path": pdf_output_path,
                "pdf_created": pdf_created,
                "evaluation_metrics": evaluation_metrics,
                "used_regional_lora": bool(REGIONAL_LORA_AVAILABLE and use_regional_lora and character_image is not None)
            }
            save_comic_memory(memory)

            st.balloons()
            st.success("🎉 Comic Generated Successfully! You can now iteratively edit individual panels below.")

        except Exception as e:
            st.error("💥 An unexpected error occurred during the generation process!")
            with st.expander("Show Error Details"):
                st.error(f"{e}")
                st.code(traceback.format_exc())
            print(traceback.format_exc())

    memory = st.session_state.comic_memory
    if memory:
        st.divider()
        render_session_comic(memory)

        if memory.get("panel_data"):
            display_emotional_arc(memory["panel_data"])

        st.markdown('<div class="story-card">', unsafe_allow_html=True)
        st.subheader("📖 Story Snapshot")
        story_data = memory.get("story_data", {})
        if story_data:
            st.markdown(f"### {story_data.get('title', 'My Comic')}")
            for section in ["introduction", "storyline", "climax", "resolution", "moral"]:
                if story_data.get(section):
                    st.markdown(f"**{section.capitalize()}:** {story_data[section]}")
        st.markdown('</div>', unsafe_allow_html=True)

        if memory.get("evaluation_metrics"):
            display_metrics(memory["evaluation_metrics"])

        st.divider()
        st.subheader("✏️ Edit Existing Comic (Session Memory)")

        edit_mode_col1, edit_mode_col2 = st.columns([1.2, 1.8])
        with edit_mode_col1:
            auto_detect_panels = st.checkbox(
                "Auto-detect affected panel(s)",
                value=True,
                help="If enabled, the app detects which panel(s) should change from your request."
            )
            selected_panel = st.selectbox(
                "Manual panel target",
                options=[i + 1 for i in range(len(memory.get("panel_data", [])))],
                index=0,
                disabled=auto_detect_panels
            )
        with edit_mode_col2:
            edit_request = st.text_area(
                "Follow-up instruction",
                key="edit_request_input",
                height=110,
                placeholder="e.g., Change panel 2 so the hero wears a red jacket and update the dialogue to sound more confident."
            )

        apply_edit_btn = st.button("🔧 Apply Edit", type="secondary")

        if apply_edit_btn:
            if not edit_request.strip():
                st.warning("Please provide an edit instruction.")
            else:
                try:
                    working_memory = deepcopy(memory)
                    target_panels = None if auto_detect_panels else [selected_panel - 1]

                    with st.spinner("Analyzing and applying panel edit..."):
                        edit_result = generate_panels.apply_edit_to_panels(
                            edit_request=edit_request,
                            panel_data=working_memory.get("panel_data", []),
                            art_style=working_memory.get("art_style", "Anime Style"),
                            story_data=working_memory.get("story_data", {}),
                            target_panels=target_panels
                        )

                    updated_panels = edit_result.get("panel_data", working_memory.get("panel_data", []))
                    affected_panels = edit_result.get("affected_panels", [])
                    visual_panels = edit_result.get("visual_panels", [])

                    if not affected_panels:
                        st.info("No panel changes were required for this instruction.")
                    else:
                        working_memory["panel_data"] = updated_panels

                        # Regenerate images only for visually affected panels.
                        if visual_panels:
                            img_generator = create_image_generator(
                                char_image_path=working_memory.get("character_image_path"),
                                ref_image_path=working_memory.get("reference_image_path"),
                                character_description=working_memory.get("character_description", ""),
                                character_name=working_memory.get("character_name", "")
                            )

                            for idx in visual_panels:
                                current_paths = working_memory.get("image_paths", [])
                                current_output = current_paths[idx] if idx < len(current_paths) else os.path.join(PANEL_FOLDER, f"panel_{idx+1:02d}.png")
                                new_path = img_generator.generate_single_panel(
                                    panel_data=updated_panels[idx],
                                    art_style=working_memory.get("art_style", "Anime Style"),
                                    panel_number=idx + 1,
                                    output_path=current_output,
                                    use_enhanced=True
                                )
                                if new_path:
                                    working_memory["image_paths"][idx] = new_path

                        working_memory = regenerate_comic_outputs(
                            memory=working_memory,
                            use_dynamic_layout=working_memory.get("use_dynamic_layout", True),
                            evaluate=True
                        )

                        save_comic_memory(working_memory)

                        affected_human = ", ".join([str(i + 1) for i in affected_panels])
                        visual_human = ", ".join([str(i + 1) for i in visual_panels]) if visual_panels else "none"
                        st.success(
                            f"✅ Edit applied. Updated panel(s): {affected_human}. "
                            f"Image regeneration called for panel(s): {visual_human}."
                        )
                        st.rerun()

                except Exception as e:
                    st.error(f"Failed to apply edit: {e}")
                    with st.expander("Show Error Details"):
                        st.code(traceback.format_exc())

        st.divider()
        st.subheader("⬇️ Download Current Comic")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        output_image_path = memory.get("output_image_path")
        pdf_output_path = memory.get("pdf_output_path")

        with dl_col1:
            if output_image_path and os.path.exists(output_image_path):
                with open(output_image_path, "rb") as img_file:
                    st.download_button(
                        label="📷 Download as PNG",
                        data=img_file,
                        file_name="comic_strip.png",
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.button("📷 PNG Unavailable", disabled=True, use_container_width=True)

        with dl_col2:
            if memory.get("pdf_created") and pdf_output_path and os.path.exists(pdf_output_path):
                with open(pdf_output_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Download as PDF",
                        data=pdf_file,
                        file_name="comic_strip.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.button("📄 PDF Unavailable", disabled=True, use_container_width=True)

        with dl_col3:
            if st.button("🖼️ Export Panels", use_container_width=True):
                panels_folder = os.path.join(OUTPUT_FOLDER, "individual_panels")
                exported = process_comic.export_individual_panels(
                    memory.get("image_paths", []),
                    panels_folder,
                    memory.get("panel_texts", [])
                )
                st.success(f"✅ Exported {len(exported)} panels to {panels_folder}")

with tab2:
    st.subheader("📖 Comic Gallery")
    st.markdown("*Your previously generated comics will appear here.*")
    
    # Check for existing comics
    output_files = list(Path(OUTPUT_FOLDER).glob("*.png"))
    
    if output_files:
        gallery_cols = st.columns(3)
        for i, img_path in enumerate(output_files[:9]):
            with gallery_cols[i % 3]:
                st.image(str(img_path), caption=img_path.stem, use_container_width=True)
    else:
        st.info("No comics generated yet. Create your first comic in the 'Create Comic' tab!")

with tab3:
    st.subheader("📊 About ComicCrafter AI v2.0")
    
    st.markdown("""
    ### 🔬 Research Features
    
    **ComicCrafter AI v2.0** is a research-grade AI comic generation system with novel contributions for IEEE publication:
    
    ---
    
    #### 🧬 NEW IN v2.0: Dynamic Regional LoRA Injection
    
    **The Research Problem:** Standard multi-LoRA approaches apply character adapters globally, 
    causing "character bleeding" when two characters appear in the same panel.
    
    **Our Solution:** Apply character-specific LoRA weights ONLY to their spatial regions during 
    cross-attention computation in the diffusion process.
    
    **Algorithm:**
    1. Gemini Vision extracts character bounding boxes from panel description
    2. Attention masks are generated with Gaussian edge blending
    3. During diffusion, cross-attention is modified per-region
    4. Result: Multiple distinct characters without interference
    
    **Key Innovation:** `A_final = A_base + Σ(M_i ⊙ A_lora_i)` instead of global `A_final = A_base + Σ A_lora_i`
    
    ---
    
    #### 1. 🎭 Character Consistency Engine
    Uses embedding-based identity tracking to maintain visual consistency of characters across all panels.
    - Extracts character attributes from story
    - Generates consistent visual descriptors
    - Tracks character appearances across panels
    
    #### 2. 💫 Emotion-Driven Layout System
    Dynamically adjusts panel compositions based on emotional content:
    - **Fear**: Smaller, cramped panels with dutch angles
    - **Joy**: Open, bright panels with warm colors
    - **Action**: Large dynamic panels with motion lines
    - **Dramatic**: Splash pages for impact moments
    
    #### 3. 🌍 Curated Style Adaptation
    Focused on stable, high-performing comic traditions:
    - 🇯🇵 Japanese Manga
    - 🌸 Anime Style
    - 🇺🇸 American Superhero
    - 🇧🇪 Franco-Belgian
    - 🎬 Film Noir
    - 📱 Webtoon (Vertical)
    
    #### 4. 📏 Quality Evaluation Framework
    Novel metrics for assessing AI-generated comics:
    - Narrative coherence scoring
    - Visual consistency evaluation
    - Emotion conveyance assessment
    - Overall grade with improvement suggestions
    
    ---
    
    ### 🛠️ Technical Stack
    - **LLM**: Google Gemini 2.0 Flash
    - **Image Generation**: Stability AI / ClipDrop / Replicate (FLUX/SDXL)
    - **Regional LoRA**: Custom cross-attention manipulation via diffusers
    - **Character LoRA**: PuLID, PhotoMaker, or trained LoRA via Replicate
    - **UI**: Streamlit
    - **Processing**: PIL/Pillow, ReportLab
    
    ### 📚 Citation
    If you use ComicCrafter AI in your research, please cite:
    ```bibtex
    @article{comiccrafter2024,
      title={Dynamic Regional LoRA Injection for Consistent Multi-Character Comic Generation},
      author={Your Name},
      journal={IEEE Transactions on Multimedia},
      year={2024},
      note={Addresses multi-character LoRA interference via spatial attention masking}
    }
    ```
    """)

st.divider()
st.caption("ComicCrafter AI - Research Edition • Powered by Gemini, Stability AI & Streamlit")