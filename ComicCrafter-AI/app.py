"""
ComicCrafter AI - Enhanced Edition
===================================
Research-grade AI Comic Generation System

Features:
- Character upload and reference image support
- Multi-cultural comic tradition selection (15+ styles)
- Emotion-driven dynamic panel layouts
- Character consistency tracking
- Quality evaluation metrics
- Professional PDF export

Research Contributions:
1. Character Consistency Engine with embedding-based tracking
2. Emotion-Driven Layout System with comic grammar
3. Multi-Cultural Style Adaptation across 15+ traditions
4. Quantitative Comic Quality Evaluation Framework
"""

import streamlit as st
import os
import base64
import json
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
        "description": "Black & white with screentones, right-to-left reading, expressive eyes, speed lines, dramatic close-ups"
    },
    "Anime Style": {
        "tradition": "ANIME",
        "icon": "🌸",
        "description": "Vibrant colors, cel shading, large expressive eyes, detailed hair, dynamic action poses"
    },
    "American Superhero": {
        "tradition": "AMERICAN_SUPERHERO",
        "icon": "💥",
        "description": "Bold outlines, heavy inking, saturated colors, exaggerated muscular features, dynamic splash pages"
    },
    "Franco-Belgian (Tintin)": {
        "tradition": "FRANCO_BELGIAN",
        "icon": "🎨",
        "description": "Clear line style, soft flat shading, rich detailed backgrounds, semi-realistic proportions"
    },
    "Manhwa (Korean)": {
        "tradition": "MANHWA",
        "icon": "🇰🇷",
        "description": "Full color, vertical scroll format, realistic proportions, manhwa-style eyes, soft shading"
    },
    "Manhua (Chinese)": {
        "tradition": "MANHUA",
        "icon": "🇨🇳",
        "description": "Full color, detailed backgrounds, martial arts focused, flowing action lines"
    },
    "Webcomic Modern": {
        "tradition": "WEBCOMIC",
        "icon": "💻",
        "description": "Clean digital art, consistent colors, simplified backgrounds, meme-friendly expressions"
    },
    "Webtoon (Vertical)": {
        "tradition": "WEBTOON",
        "icon": "📱",
        "description": "Vertical scroll format, full color, dramatic pacing, mobile-optimized layouts"
    },
    "Horror Comics": {
        "tradition": "HORROR",
        "icon": "👻",
        "description": "Heavy shadows, unsettling angles, muted colors with red accents, detailed gore"
    },
    "Film Noir": {
        "tradition": "NOIR",
        "icon": "🎬",
        "description": "High contrast B&W, heavy shadows, cynical atmosphere, dramatic lighting"
    },
    "Newspaper Strip": {
        "tradition": "NEWSPAPER",
        "icon": "📰",
        "description": "Simple clear art, 3-4 panel format, punchy dialogue, daily comic style"
    },
    "Golden Age": {
        "tradition": "GOLDEN_AGE",
        "icon": "✨",
        "description": "Vintage aesthetic, limited color palette, bold primary colors, classic heroic poses"
    },
    "Underground Comix": {
        "tradition": "UNDERGROUND",
        "icon": "🎸",
        "description": "Counter-culture aesthetic, exaggerated grotesque, psychedelic elements, adult themes"
    },
    "Children's Illustrated": {
        "tradition": "CHILDRENS",
        "icon": "🧸",
        "description": "Soft rounded shapes, bright primary colors, simple expressions, educational focus"
    },
    "Indie Graphic Novel": {
        "tradition": "INDIE",
        "icon": "📖",
        "description": "Artistic experimental style, muted tones, literary focus, unique personal aesthetic"
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
    intensities = [p.get('intensity', 0.5) for p in panel_data]
    
    # Simple text-based visualization
    arc_display = " → ".join([f"{e.upper()}" for e in emotions])
    st.markdown(f"**Emotion Flow:** {arc_display}")
    
    # Intensity bar
    intensity_avg = sum(intensities) / len(intensities) if intensities else 0
    st.progress(intensity_avg, text=f"Average Intensity: {intensity_avg:.0%}")


# Apply styles
set_enhanced_styles()

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
        user_prompt = st.text_area(
            "",
            "",
            height=150,
            placeholder="Enter your story idea here...\n\ne.g., A forgetful wizard accidentally turns himself into different animals while trying to find his lost spellbook in his messy cottage. Each transformation leads to a humorous situation until he finally finds the book.",
            label_visibility="collapsed"
        )
        
        # Optional story details
        with st.expander("📌 Additional Story Details (Optional)"):
            story_genre = st.selectbox(
                "Genre",
                ["Auto-detect", "Comedy", "Adventure", "Drama", "Horror", "Romance", "Sci-Fi", "Fantasy", "Slice of Life"]
            )
            story_mood = st.selectbox(
                "Overall Mood",
                ["Auto-detect", "Light-hearted", "Serious", "Dramatic", "Mysterious", "Exciting", "Emotional"]
            )
            target_audience = st.selectbox(
                "Target Audience",
                ["General", "Children", "Young Adult", "Adult", "All Ages"]
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.subheader("2. 🎨 Art Style")
        
        # Style selection with preview
        selected_style = st.selectbox(
            "",
            list(STYLE_TRADITIONS.keys()),
            label_visibility="collapsed"
        )
        
        style_info = STYLE_TRADITIONS[selected_style]
        st.markdown(f"""
        **{style_info['icon']} {selected_style}**
        
        {style_info['description']}
        """)
        
        # Extract simple style name for backend
        art_style = selected_style.split(" (")[0].replace("-", " ")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Generate button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        generate_button = st.button(
            "🚀 Generate Comic Strip",
            use_container_width=True,
            type="primary"
        )
    
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
        evaluation_metrics = None
        success = False
        
        # Save reference images if uploaded
        ref_image_path = None
        char_image_path = None
        
        if reference_image:
            ref_image_path = save_uploaded_image(
                reference_image, 
                REFERENCE_FOLDER, 
                "style_reference.png"
            )
        
        if character_image:
            char_image_path = save_uploaded_image(
                character_image, 
                REFERENCE_FOLDER, 
                "character_reference.png"
            )

        try:
            progress_bar = st.progress(0, text="Starting comic generation...")
            
            # --- Step 1: Generate Story ---
            progress_bar.progress(10, text="⏳ Step 1/5: Generating story...")
            with st.spinner(""):
                story_data = generate_panels.generate_story(user_prompt, art_style)
            progress_bar.progress(20, text="✅ Story generated!")

            # --- Step 2: Generate Panels with proper num_panels ---
            progress_bar.progress(25, text=f"⏳ Step 2/5: Generating {num_panels} panel descriptions...")
            with st.spinner(""):
                # ALWAYS pass num_panels to generate_panels
                try:
                    panel_data = generate_panels.generate_panels(user_prompt, art_style, num_panels)
                except Exception as e:
                    print(f"Panel generation error: {e}")
                    st.error(f"Panel generation failed: {e}")
                    st.stop()
            
            # Validate and adjust panel count
            if not isinstance(panel_data, list):
                st.error("❌ Error: Panel data is invalid.")
                st.stop()
                
            # Ensure we have exactly num_panels
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
            
            progress_bar.progress(40, text=f"✅ {len(panel_data)} panels generated!")

            panel_texts = [panel.get("Text", "...") for panel in panel_data]

            # --- Step 3: Generate Images with FULL integration ---
            progress_bar.progress(45, text="⏳ Step 3/5: Generating panel images... (This may take a while!)")
            with st.spinner(""):
                # Check if Regional LoRA should be used
                use_regional = (
                    REGIONAL_LORA_AVAILABLE and 
                    use_regional_lora and 
                    character_image is not None
                )
                
                if use_regional:
                    # === REGIONAL LORA v2.0 GENERATION ===
                    st.info("🧬 Using Regional LoRA v2.0 for character consistency...")
                    
                    try:
                        # Initialize Regional Controller
                        regional_config = RegionalConfig(
                            image_width=1024,
                            image_height=1024,
                            backend=lora_backend.split()[0] if ' ' in lora_backend else lora_backend,
                            num_inference_steps=28,
                            guidance_scale=7.5
                        )
                        
                        controller = RegionalLoRAController(
                            config=regional_config,
                            lora_cache_dir="LORA_CACHE"
                        )
                        
                        # Save to session state for metrics viewing
                        st.session_state.regional_controller = controller
                        
                        # Register main character with reference image
                        if character_name and character_description and char_image_path:
                            lora_method_clean = "instant" if "instant" in lora_method else "trained"
                            controller.register_character(
                                name=character_name or "MainCharacter",
                                description=character_description or "The main character",
                                reference_image=char_image_path,
                                train_lora=True,
                                lora_method=lora_method_clean
                            )
                            st.success(f"✅ Registered {character_name} with LoRA")
                        
                        # Register secondary characters if provided
                        if character2_name and character2_desc:
                            char2_path = None
                            if character2_image:
                                char2_path = save_uploaded_image(
                                    character2_image, REFERENCE_FOLDER, "character2_reference.png"
                                )
                            controller.register_character(
                                name=character2_name,
                                description=character2_desc,
                                reference_image=char2_path,
                                train_lora=char2_path is not None,
                                lora_method="instant"
                            )
                            st.success(f"✅ Registered {character2_name}")
                        
                        if character3_name and character3_desc:
                            char3_path = None
                            if character3_image:
                                char3_path = save_uploaded_image(
                                    character3_image, REFERENCE_FOLDER, "character3_reference.png"
                                )
                            controller.register_character(
                                name=character3_name,
                                description=character3_desc,
                                reference_image=char3_path,
                                train_lora=char3_path is not None,
                                lora_method="instant"
                            )
                            st.success(f"✅ Registered {character3_name}")
                        
                        # Build character list
                        all_characters = []
                        if character_name:
                            all_characters.append(character_name)
                        if character2_name:
                            all_characters.append(character2_name)
                        if character3_name:
                            all_characters.append(character3_name)
                        
                        # Generate panels with Regional LoRA
                        image_paths = []
                        for i, panel in enumerate(panel_data):
                            progress_pct = 45 + int((i / len(panel_data)) * 30)
                            progress_bar.progress(progress_pct, text=f"⏳ Generating panel {i+1}/{len(panel_data)}...")
                            
                            # Get characters for this panel
                            panel_chars = panel.get("characters", all_characters)
                            if not panel_chars:
                                panel_chars = all_characters
                            
                            result = controller.generate_panel(
                                description=panel.get("Description", ""),
                                characters=panel_chars,
                                art_style=art_style,
                                panel_number=i + 1,
                                emotion=panel.get("emotion", "neutral")
                            )
                            
                            # Save image
                            output_path = os.path.join(PANEL_FOLDER, f"panel_{i+1:02d}.png")
                            result.image.save(output_path)
                            image_paths.append(output_path)
                            
                            # Show layout preview if enabled
                            if show_layout_preview and i == 0:
                                layout_preview = controller.visualize_layout(result, show_masks=True)
                                st.image(layout_preview, caption=f"Panel {i+1} Layout + Masks", width=400)
                        
                        st.success(f"✅ Generated {len(image_paths)} panels with Regional LoRA")
                        
                    except Exception as e:
                        st.warning(f"⚠️ Regional LoRA failed: {e}. Falling back to standard generation.")
                        print(f"Regional LoRA error: {e}")
                        import traceback
                        traceback.print_exc()
                        use_regional = False
                
                if not use_regional:
                    # === STANDARD IMAGE GENERATION ===
                    # Create enhanced image generator
                    try:
                        img_generator = generate_image.EnhancedImageGenerator()
                        
                        # Set character reference image if uploaded
                        if char_image_path:
                            img_generator.set_reference_image(char_image_path, "main_character")
                            st.info(f"📷 Using character reference: {char_image_path}")
                        
                        # Set style reference if uploaded
                        if ref_image_path:
                            img_generator.set_style_reference(ref_image_path)
                            st.info(f"🎨 Using style reference: {ref_image_path}")
                        
                        # Set character description if provided (this is KEY for consistency)
                        if character_description:
                            img_generator.set_character_description(character_description)
                            st.info(f"👤 Character: {character_description[:80]}...")
                        
                        # If character name provided, register it too
                        if character_name and character_description:
                            img_generator.set_character_description(character_description, character_name)
                        
                        # Generate images using enhanced generator with proper style
                        image_paths = list(img_generator.generate_images(panel_data, art_style, use_enhanced=True))
                        
                    except Exception as e:
                        print(f"Enhanced image generator error: {e}")
                        st.warning(f"⚠️ Using fallback image generator: {e}")
                        # Fallback to legacy generator
                        image_paths = list(generate_image.generate_images(panel_data, art_style))
            
            progress_bar.progress(75, text="✅ Images generated!")

            paths_ok = False
            if len(image_paths) >= num_panels:
                all_valid = True
                for i, img_path in enumerate(image_paths[:num_panels]):
                    is_str = isinstance(img_path, str)
                    exists = os.path.exists(img_path) if is_str else False
                    if not is_str or not exists:
                        all_valid = False
                        st.warning(f"⚠️ Problem with generated path/file for panel {i+1}: '{img_path}'")
                        break
                if all_valid:
                    paths_ok = True
                    image_paths = image_paths[:num_panels]
            else:
                st.warning(f"⚠️ Expected {num_panels} image paths, but received {len(image_paths)}.")

            # --- Step 4: Assemble Comic ---
            if paths_ok:
                output_image_path = os.path.join(OUTPUT_FOLDER, "comic_strip_with_text.png")
                pdf_output_path = os.path.join(OUTPUT_FOLDER, "comic_strip.pdf")

                progress_bar.progress(80, text="⏳ Step 4/5: Assembling comic strip...")
                with st.spinner(""):
                    # Use dynamic layout if enabled and panel_data has emotion info
                    title = story_data.get('title', 'My Comic') if story_data else 'My Comic'
                    
                    process_comic.create_comic_strip_with_text(
                        image_paths, 
                        panel_texts, 
                        output_image_path,
                        panel_data=panel_data if use_dynamic_layout else None,
                        title=title,
                        use_dynamic_layout=use_dynamic_layout
                    )
                
                progress_bar.progress(90, text="✅ Comic assembled!")

                # --- Step 5: Create PDF and Evaluate ---
                progress_bar.progress(92, text="⏳ Step 5/5: Creating PDF and evaluating quality...")
                with st.spinner(""):
                    # Create enhanced PDF
                    pdf_created = process_comic.create_comic_pdf(
                        output_image_path,
                        pdf_output_path,
                        story_data=story_data,
                        panel_data=panel_data,
                        evaluation_metrics=None  # Will add after evaluation
                    )
                    
                    # Evaluate quality if enhanced backend available
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
                            evaluation_metrics = None
                
                progress_bar.progress(100, text="✅ Complete!")

                # --- Display Results ---
                if os.path.exists(output_image_path):
                    st.balloons()
                    st.success("🎉 Comic Generated Successfully!")
                    
                    # Story and Comic Display
                    st.markdown('<div class="story-card">', unsafe_allow_html=True)
                    
                    result_col1, result_col2 = st.columns([1, 1.5])
                    
                    with result_col1:
                        st.subheader("📖 Your Story")
                        if story_data and 'title' in story_data:
                            st.markdown(f"### {story_data['title']}")
                            for section in ['introduction', 'storyline', 'climax', 'resolution', 'moral']:
                                if section in story_data and story_data[section]:
                                    st.markdown(f"**{section.capitalize()}:** {story_data[section]}")
                        else:
                            st.write("Story data not available")
                        
                        # Emotional arc display
                        if panel_data and ENHANCED_BACKEND:
                            display_emotional_arc(panel_data)
                    
                    with result_col2:
                        st.subheader("🎨 Your Comic")
                        st.image(
                            output_image_path, 
                            caption="Your Generated Comic Strip", 
                            use_container_width=True
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Evaluation Metrics
                    if evaluation_metrics:
                        st.divider()
                        display_metrics(evaluation_metrics)
                    
                    # Download Section
                    st.divider()
                    st.subheader("⬇️ Download Your Comic")
                    
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    
                    with dl_col1:
                        with open(output_image_path, "rb") as img_file:
                            st.download_button(
                                label="📷 Download as PNG",
                                data=img_file,
                                file_name="comic_strip.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    
                    with dl_col2:
                        if pdf_created and os.path.exists(pdf_output_path):
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
                        # Export individual panels
                        if st.button("🖼️ Export Panels", use_container_width=True):
                            panels_folder = os.path.join(OUTPUT_FOLDER, "individual_panels")
                            exported = process_comic.export_individual_panels(
                                image_paths, panels_folder, panel_texts
                            )
                            st.success(f"✅ Exported {len(exported)} panels to {panels_folder}")
                    
                    success = True

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
    
    #### 3. 🌍 Multi-Cultural Style Adaptation
    Supports 15+ global comic traditions:
    - 🇯🇵 Japanese Manga
    - 🇰🇷 Korean Manhwa
    - 🇨🇳 Chinese Manhua
    - 🇺🇸 American Superhero
    - 🇧🇪 Franco-Belgian
    - And many more!
    
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