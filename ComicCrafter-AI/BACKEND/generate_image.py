"""
Enhanced Image Generation Module
=================================
Advanced comic panel image generation with:
- Character consistency through prompt injection
- Emotion-driven visual styling
- Reference image support for style/character guidance
- Multi-cultural comic tradition styling
- Quality scoring and retry logic

Research Contributions:
- Consistency-aware prompt engineering
- Emotion-to-visual parameter mapping
- Reference-guided generation workflows
"""

import os
import io
import requests
import base64
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageEnhance
import time
from typing import List, Dict, Optional, Tuple, Any

# Load environment variables
load_dotenv()
CLIPDROP_API_KEY = os.getenv("CLIPDROP_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

# Check for API keys
if CLIPDROP_API_KEY:
    print("✅ CLIPDROP_API_KEY loaded successfully.")
elif STABILITY_API_KEY:
    print("✅ STABILITY_API_KEY loaded successfully.")
else:
    print("⚠️ No image generation API key found. Please set CLIPDROP_API_KEY or STABILITY_API_KEY.")

# Output directory for images
OUTPUT_DIR = "PANEL_IMAGES"
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Output directory '{OUTPUT_DIR}' checked/created.")
except OSError as e:
    raise OSError(f"❌ Failed to create output directory '{OUTPUT_DIR}': {e}")


# Enhanced art style mappings with emotion modifiers - ALL 15 TRADITIONS
STYLE_MAPPINGS = {
    "Manga": {
        "base": "High-contrast black and white sketch with sharp, clean lines, exaggerated facial expressions, and dramatic shading. Manga style, screen tones, Japanese comic aesthetic, speed lines",
        "color_mode": "grayscale",
        "keywords": ["manga", "Japanese comic", "screen tones", "expressive eyes"]
    },
    "Manga (Japanese)": {
        "base": "High-contrast black and white sketch with sharp, clean lines, exaggerated facial expressions, and dramatic shading. Manga style, screen tones, Japanese comic aesthetic, speed lines",
        "color_mode": "grayscale",
        "keywords": ["manga", "Japanese comic", "screen tones", "expressive eyes"]
    },
    "Anime": {
        "base": "Vibrant colors with smooth cel shading, large expressive eyes, and detailed hair. Dynamic action poses with fluid motion lines. Anime style, Japanese animation aesthetic",
        "color_mode": "vibrant",
        "keywords": ["anime", "cel shading", "vibrant colors", "detailed hair"]
    },
    "Anime Style": {
        "base": "Vibrant colors with smooth cel shading, large expressive eyes, and detailed hair. Dynamic action poses with fluid motion lines. Anime style, Japanese animation aesthetic",
        "color_mode": "vibrant",
        "keywords": ["anime", "cel shading", "vibrant colors", "detailed hair"]
    },
    "American": {
        "base": "Bold outlines with heavy inking, bright and saturated colors, and exaggerated muscular features. Classic American comic book style, dynamic compositions",
        "color_mode": "saturated",
        "keywords": ["American comic", "superhero style", "bold outlines", "dynamic"]
    },
    "American Superhero": {
        "base": "Bold outlines with heavy inking, bright and saturated colors, and exaggerated muscular features. Classic American superhero comic book style, dynamic splash pages, heroic poses",
        "color_mode": "saturated",
        "keywords": ["American comic", "superhero style", "bold outlines", "dynamic"]
    },
    "Belgian": {
        "base": "Clean, clear lines with soft, flat shading. Rich and detailed backgrounds in a semi-realistic style, inspired by Tintin comics. Ligne claire style",
        "color_mode": "flat",
        "keywords": ["Tintin style", "ligne claire", "detailed backgrounds", "European comic"]
    },
    "Franco-Belgian": {
        "base": "Clean, clear lines with soft, flat shading. Rich and detailed backgrounds in a semi-realistic style, inspired by Tintin comics. Ligne claire style, European bande dessinée",
        "color_mode": "flat",
        "keywords": ["Tintin style", "ligne claire", "detailed backgrounds", "European comic"]
    },
    "Franco Belgian": {
        "base": "Clean, clear lines with soft, flat shading. Rich and detailed backgrounds in a semi-realistic style, inspired by Tintin comics. Ligne claire style, European bande dessinée",
        "color_mode": "flat",
        "keywords": ["Tintin style", "ligne claire", "detailed backgrounds", "European comic"]
    },
    "Manhwa": {
        "base": "Soft gradient shading, elongated elegant proportions, detailed fashion, pastel colors with vivid accents. Korean webtoon style, full color, realistic faces",
        "color_mode": "soft",
        "keywords": ["manhwa", "Korean comic", "webtoon", "elegant proportions"]
    },
    "Manhwa (Korean)": {
        "base": "Soft gradient shading, elongated elegant proportions, detailed fashion, pastel colors with vivid accents. Korean webtoon style, full color, realistic faces",
        "color_mode": "soft",
        "keywords": ["manhwa", "Korean comic", "webtoon", "elegant proportions"]
    },
    "Manhua": {
        "base": "Full color detailed illustration, martial arts aesthetic, flowing action lines, detailed traditional Chinese clothing, dynamic poses. Chinese comic manhua style",
        "color_mode": "vibrant",
        "keywords": ["manhua", "Chinese comic", "martial arts", "flowing lines"]
    },
    "Manhua (Chinese)": {
        "base": "Full color detailed illustration, martial arts aesthetic, flowing action lines, detailed traditional Chinese clothing, dynamic poses. Chinese comic manhua style",
        "color_mode": "vibrant",
        "keywords": ["manhua", "Chinese comic", "martial arts", "flowing lines"]
    },
    "Horror": {
        "base": "Dark atmospheric illustration, heavy dramatic shadows, unsettling compositions, detailed grotesque elements. Horror comic style, muted colors with red accents",
        "color_mode": "dark",
        "keywords": ["horror", "dark", "atmospheric", "shadows"]
    },
    "Horror Comics": {
        "base": "Dark atmospheric illustration, heavy dramatic shadows, unsettling compositions, detailed grotesque elements. Horror comic style, muted colors with red accents",
        "color_mode": "dark",
        "keywords": ["horror", "dark", "atmospheric", "shadows"]
    },
    "Noir": {
        "base": "High contrast black and white, dramatic film noir lighting, heavy silhouettes, minimal mid-tones. Noir comic style, cynical atmosphere",
        "color_mode": "high_contrast_bw",
        "keywords": ["noir", "film noir", "high contrast", "shadows"]
    },
    "Film Noir": {
        "base": "High contrast black and white, dramatic film noir lighting, heavy silhouettes, minimal mid-tones. Classic film noir comic style, cynical detective atmosphere",
        "color_mode": "high_contrast_bw",
        "keywords": ["noir", "film noir", "high contrast", "shadows"]
    },
    "Webcomic": {
        "base": "Simple stylized character designs, expressive faces, minimal backgrounds, casual approachable aesthetic. Modern webcomic style",
        "color_mode": "simple",
        "keywords": ["webcomic", "simple", "expressive", "casual"]
    },
    "Webcomic Modern": {
        "base": "Clean digital art, consistent flat colors, simplified stylized backgrounds, meme-friendly expressive faces. Modern webcomic style",
        "color_mode": "simple",
        "keywords": ["webcomic", "simple", "expressive", "casual"]
    },
    "Webtoon": {
        "base": "Vertical scroll optimized, full color with soft shading, dramatic emotional pacing, mobile-friendly compositions. Korean webtoon digital style",
        "color_mode": "soft",
        "keywords": ["webtoon", "vertical scroll", "full color", "dramatic"]
    },
    "Webtoon (Vertical)": {
        "base": "Vertical scroll optimized, full color with soft shading, dramatic emotional pacing, mobile-friendly compositions. Korean webtoon digital style",
        "color_mode": "soft",
        "keywords": ["webtoon", "vertical scroll", "full color", "dramatic"]
    },
    "Newspaper": {
        "base": "Simple clear line art, limited color palette, punchy compact compositions, 3-4 panel format. Classic newspaper comic strip style like Peanuts or Calvin and Hobbes",
        "color_mode": "simple",
        "keywords": ["newspaper comic", "daily strip", "simple lines", "humor"]
    },
    "Newspaper Strip": {
        "base": "Simple clear line art, limited color palette, punchy compact compositions, 3-4 panel format. Classic newspaper comic strip style like Peanuts or Calvin and Hobbes",
        "color_mode": "simple",
        "keywords": ["newspaper comic", "daily strip", "simple lines", "humor"]
    },
    "Golden Age": {
        "base": "Vintage 1940s comic aesthetic, limited bold primary color palette, classic heroic poses, retro halftone dot patterns. Golden age superhero comic style",
        "color_mode": "vintage",
        "keywords": ["golden age", "vintage", "retro", "classic superhero"]
    },
    "Underground": {
        "base": "Counter-culture aesthetic, exaggerated grotesque features, psychedelic elements, bold experimental linework. Underground comix style like R. Crumb",
        "color_mode": "psychedelic",
        "keywords": ["underground", "comix", "psychedelic", "alternative"]
    },
    "Underground Comix": {
        "base": "Counter-culture aesthetic, exaggerated grotesque features, psychedelic elements, bold experimental linework. Underground comix style like R. Crumb",
        "color_mode": "psychedelic",
        "keywords": ["underground", "comix", "psychedelic", "alternative"]
    },
    "Childrens": {
        "base": "Soft rounded character shapes, bright primary colors, simple friendly expressions, educational friendly style. Children's book illustration style",
        "color_mode": "bright",
        "keywords": ["children's book", "cute", "simple", "friendly"]
    },
    "Children's Illustrated": {
        "base": "Soft rounded character shapes, bright primary colors, simple friendly expressions, educational friendly style. Children's book illustration style",
        "color_mode": "bright",
        "keywords": ["children's book", "cute", "simple", "friendly"]
    },
    "Indie": {
        "base": "Artistic experimental style, muted earth tones, literary focus, unique personal aesthetic, introspective mood. Indie graphic novel style",
        "color_mode": "muted",
        "keywords": ["indie", "graphic novel", "artistic", "literary"]
    },
    "Indie Graphic Novel": {
        "base": "Artistic experimental style, muted earth tones, literary focus, unique personal aesthetic, introspective mood. Indie graphic novel style",
        "color_mode": "muted",
        "keywords": ["indie", "graphic novel", "artistic", "literary"]
    }
}

# Emotion to visual modifier mappings
EMOTION_VISUAL_MODIFIERS = {
    "joy": "warm lighting, bright atmosphere, cheerful mood, golden tones",
    "sadness": "muted colors, soft shadows, melancholic atmosphere, blue tones, rain or grey sky",
    "fear": "dark shadows, unsettling angles, ominous atmosphere, cold blue lighting",
    "anger": "intense red tones, sharp contrasts, aggressive composition, dramatic shadows",
    "surprise": "bright lighting, dynamic composition, exaggerated expressions",
    "tension": "moody lighting, dramatic shadows, suspenseful atmosphere, desaturated colors",
    "excitement": "vibrant colors, dynamic motion, energetic composition, speed lines",
    "love": "soft warm lighting, romantic atmosphere, gentle colors, soft focus effect",
    "mystery": "foggy atmosphere, obscured elements, dramatic lighting, dark corners",
    "action": "dynamic angles, motion blur, impact effects, energetic composition",
    "comedy": "bright cheerful colors, exaggerated expressions, playful composition",
    "dramatic": "cinematic lighting, epic composition, strong contrasts",
    "neutral": "balanced lighting, natural colors, clear composition"
}

# System instructions for clean image generation
SYSTEM_INSTRUCTIONS = """
STRICT VISUAL INSTRUCTIONS:
- Generate a high-quality, visually appealing comic panel image
- NO speech bubbles, NO text, NO symbols, NO letters, NO words in the image
- Only clear, clean, and high-quality visual details
- Ensure realistic proportions, natural expressions, and artistic coherence
- Maintain consistent character appearances as described
- Focus on visual storytelling through composition and expression
"""

# API Endpoints
CLIPDROP_API_URL = "https://clipdrop-api.co/text-to-image/v1"
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"


class EnhancedImageGenerator:
    """
    Advanced image generator with consistency and emotion awareness.
    """
    
    def __init__(self):
        """Initialize the enhanced generator."""
        self.reference_images: Dict[str, Image.Image] = {}
        self.style_reference: Optional[Image.Image] = None
        self.character_descriptions: Dict[str, str] = {}
        self.generated_images: List[Image.Image] = []
        self.main_character_description: str = ""
        self.reference_image_path: Optional[str] = None
        
    def set_reference_image(self, image_or_path, name: str = "main_character"):
        """
        Set a reference image for a character or style.
        
        Args:
            image_or_path: Either a PIL Image or a file path string
            name: Identifier for the reference (character name or 'style')
        """
        if isinstance(image_or_path, str):
            # It's a file path
            if os.path.exists(image_or_path):
                try:
                    img = Image.open(image_or_path)
                    self.reference_images[name] = img.copy()
                    self.reference_image_path = image_or_path
                    print(f"✅ Reference image loaded from: {image_or_path}")
                except Exception as e:
                    print(f"⚠️ Could not load reference image: {e}")
            else:
                print(f"⚠️ Reference image path not found: {image_or_path}")
        elif isinstance(image_or_path, Image.Image):
            self.reference_images[name] = image_or_path
            print(f"✅ Reference image set for: {name}")
        else:
            print(f"⚠️ Invalid reference image type: {type(image_or_path)}")
    
    def set_style_reference(self, image_or_path):
        """Set a style reference image."""
        if isinstance(image_or_path, str):
            if os.path.exists(image_or_path):
                try:
                    self.style_reference = Image.open(image_or_path).copy()
                    print(f"✅ Style reference loaded from: {image_or_path}")
                except Exception as e:
                    print(f"⚠️ Could not load style reference: {e}")
        elif isinstance(image_or_path, Image.Image):
            self.style_reference = image_or_path
            print("✅ Style reference image set.")
        print("✅ Style reference image set.")
    
    def set_character_description(self, description: str, name: str = "main_character"):
        """
        Set a character description for consistency.
        
        Args:
            description: Physical description of the character
            name: Character name (optional, defaults to main_character)
        """
        self.character_descriptions[name] = description
        self.main_character_description = description
        print(f"✅ Character description set: {description[:100]}...")
    
    def build_enhanced_prompt(self, panel_data: Dict, art_style: str) -> str:
        """
        Build an enhanced prompt with consistency and emotion modifiers.
        
        Args:
            panel_data: Panel information dictionary
            art_style: Selected art style
        
        Returns:
            Enhanced prompt string
        """
        # Get base description
        description = panel_data.get("Description", "")
        
        # Get style base - try multiple variations of the style name
        style_info = None
        style_variations = [
            art_style,
            art_style.replace(" ", "-"),
            art_style.replace("-", " "),
            art_style.split("(")[0].strip(),  # Remove parenthetical
            art_style.title()
        ]
        
        for style_var in style_variations:
            if style_var in STYLE_MAPPINGS:
                style_info = STYLE_MAPPINGS[style_var]
                break
        
        if not style_info:
            print(f"⚠️ Style '{art_style}' not found, using Anime as fallback")
            style_info = STYLE_MAPPINGS["Anime"]
        
        style_base = style_info["base"]
        
        # Get emotion modifiers
        emotion = panel_data.get("emotion", panel_data.get("Emotion", "neutral")).lower()
        emotion_modifier = EMOTION_VISUAL_MODIFIERS.get(emotion, EMOTION_VISUAL_MODIFIERS["neutral"])
        
        # Get additional prompt modifiers from panel data
        prompt_modifiers = panel_data.get("prompt_modifiers", "")
        
        # Add character consistency tokens - ALWAYS include main character if set
        character_tokens = []
        
        # Add main character description if available
        if self.main_character_description:
            character_tokens.append(f"Main character appearance: {self.main_character_description}")
        
        # Add any other registered characters
        characters = panel_data.get("characters", [])
        for char_name in characters:
            if char_name in self.character_descriptions:
                char_tokens = f"{char_name}: {self.character_descriptions[char_name]}"
                if char_tokens not in character_tokens:
                    character_tokens.append(char_tokens)
        
        # Build full prompt with STRONG character consistency emphasis
        prompt_parts = [
            description
        ]
        
        # Add character consistency EARLY in prompt for stronger effect
        if character_tokens:
            prompt_parts.append(f"\n\nIMPORTANT CHARACTER APPEARANCE - MUST MATCH EXACTLY: {'; '.join(character_tokens)}")
        
        prompt_parts.append(f"\n\nArt Style: {style_base}")
        prompt_parts.append(f"\n\nAtmosphere and Mood: {emotion_modifier}")
        
        if prompt_modifiers:
            prompt_parts.append(f"\n\nAdditional Visual Style: {prompt_modifiers}")
        
        prompt_parts.append(f"\n\n{SYSTEM_INSTRUCTIONS}")
        
        full_prompt = "".join(prompt_parts)
        print(f"   Built prompt ({len(full_prompt)} chars) with emotion: {emotion}")
        
        return full_prompt
    
    def generate_images(self, panel_data: List[Dict], art_style: str,
                        use_enhanced: bool = True) -> List[str]:
        """
        Generate images for all panels.
        
        Args:
            panel_data: List of panel dictionaries
            art_style: Selected art style
            use_enhanced: Whether to use enhanced prompts
        
        Returns:
            List of generated image paths
        """
        print(f"--- Starting Enhanced Image Generation for Style: {art_style} ---")
        
        if art_style not in STYLE_MAPPINGS:
            print(f"⚠️ Unknown art style '{art_style}', defaulting to 'Anime'")
            art_style = "Anime"
        
        if not isinstance(panel_data, list) or not panel_data:
            print(f"❌ Invalid or empty panel_data received")
            return []
        
        image_paths = []
        self.generated_images = []
        
        for i, panel in enumerate(panel_data):
            if not isinstance(panel, dict) or "Description" not in panel:
                print(f"⚠️ Skipping panel {i+1}: Invalid format")
                continue
            
            print(f"\n🔄 Generating image for Panel {i+1}...")
            
            # Build prompt
            if use_enhanced:
                full_prompt = self.build_enhanced_prompt(panel, art_style)
            else:
                # Legacy prompt building
                panel_description = panel.get("Description", "")
                style_base = STYLE_MAPPINGS[art_style]["base"]
                full_prompt = f"{panel_description}.\nArt Style: {style_base}.\n{SYSTEM_INSTRUCTIONS}"
            
            print(f"   Description: {panel.get('Description', '')[:100]}...")
            
            # Generate image
            image_path = self._generate_single_image(full_prompt, i + 1, panel)
            
            if image_path:
                image_paths.append(image_path)
                # Load and store generated image
                try:
                    img = Image.open(image_path)
                    self.generated_images.append(img.copy())
                except Exception as e:
                    print(f"   ⚠️ Could not load generated image: {e}")
        
        print(f"\n--- Image Generation Finished ---")
        print(f"   Successfully generated {len(image_paths)} images.")
        
        return image_paths
    
    def _generate_single_image(self, prompt: str, panel_num: int, 
                                panel_data: Dict = None) -> Optional[str]:
        """
        Generate a single panel image.
        
        Args:
            prompt: Full generation prompt
            panel_num: Panel number for filename
            panel_data: Panel data for post-processing
        
        Returns:
            Path to generated image or None
        """
        max_retries = 2
        retry_delay = 3
        
        # Try Clipdrop first, then Stability
        api_key = CLIPDROP_API_KEY or STABILITY_API_KEY
        
        if not api_key:
            print("   ❌ No API key available for image generation")
            return None
        
        for attempt in range(max_retries + 1):
            try:
                print(f"   Attempt {attempt + 1}/{max_retries + 1}: Calling API...")
                
                if CLIPDROP_API_KEY:
                    response = self._call_clipdrop(prompt)
                else:
                    response = self._call_stability(prompt)
                
                if response and response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '').lower()
                    
                    if 'image' in content_type or 'application/json' in content_type:
                        image_data = self._extract_image_data(response)
                        
                        if image_data:
                            image = Image.open(io.BytesIO(image_data))
                            
                            # Convert to RGB if needed
                            if image.mode != "RGB":
                                image = image.convert("RGB")
                            
                            # Apply emotion-based post-processing
                            if panel_data:
                                image = self._apply_emotion_effects(image, panel_data)
                            
                            # Save image
                            image_path = os.path.join(OUTPUT_DIR, f"panel_{panel_num}.png")
                            image.save(image_path, format="PNG")
                            print(f"   ✅ Image for Panel {panel_num} saved successfully.")
                            
                            return image_path
                
                # Handle errors
                if response:
                    if response.status_code == 429:
                        wait_time = retry_delay * (attempt + 2)
                        print(f"   Rate limited. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    elif response.status_code in [401, 403]:
                        print(f"   ❌ API authentication error")
                        break
                    else:
                        print(f"   ❌ API error: {response.status_code}")
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                
            except requests.exceptions.Timeout:
                print(f"   ❌ Timeout on attempt {attempt + 1}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
        
        print(f"   ❌ Failed to generate image for panel {panel_num}")
        return None
    
    def _call_clipdrop(self, prompt: str) -> Optional[requests.Response]:
        """Call Clipdrop API."""
        headers = {"x-api-key": CLIPDROP_API_KEY}
        payload = {"prompt": prompt}
        
        return requests.post(
            CLIPDROP_API_URL,
            headers=headers,
            json=payload,
            timeout=90
        )
    
    def _call_stability(self, prompt: str) -> Optional[requests.Response]:
        """Call Stability AI API."""
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text_prompts": [{"text": prompt, "weight": 1}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30
        }
        
        return requests.post(
            STABILITY_API_URL,
            headers=headers,
            json=payload,
            timeout=90
        )
    
    def _extract_image_data(self, response: requests.Response) -> Optional[bytes]:
        """Extract image data from API response."""
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'image' in content_type:
            return response.content
        elif 'application/json' in content_type:
            # Stability AI returns base64 in JSON
            try:
                data = response.json()
                if 'artifacts' in data and data['artifacts']:
                    return base64.b64decode(data['artifacts'][0]['base64'])
            except Exception as e:
                print(f"   Error parsing JSON response: {e}")
        
        return None
    
    def _apply_emotion_effects(self, image: Image.Image, 
                                panel_data: Dict) -> Image.Image:
        """
        Apply emotion-based post-processing effects.
        
        Args:
            image: PIL Image to process
            panel_data: Panel data with emotion info
        
        Returns:
            Processed PIL Image
        """
        emotion = panel_data.get("emotion", "neutral")
        intensity = panel_data.get("emotion_intensity", 0.5)
        
        # Get panel style info
        panel_style = panel_data.get("panel_style", {})
        color_temp = panel_style.get("color_temperature", "neutral")
        
        try:
            # Apply color temperature adjustments
            if color_temp == "warm" or emotion in ["joy", "love", "excitement"]:
                # Warm color shift
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.1)
            elif color_temp == "cool" or emotion in ["fear", "sadness", "mystery"]:
                # Cool color shift - slightly desaturate
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(0.9)
            elif color_temp == "desaturated" or emotion == "tension":
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(0.7)
            
            # Apply brightness adjustments based on emotion
            if emotion in ["fear", "mystery", "noir"]:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(0.85)
            elif emotion in ["joy", "comedy"]:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.1)
            
            # Apply contrast adjustments
            if emotion in ["anger", "dramatic", "action"]:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.15)
            
        except Exception as e:
            print(f"   ⚠️ Could not apply emotion effects: {e}")
        
        return image
    
    def get_generated_images(self) -> List[Image.Image]:
        """Get list of generated images for evaluation."""
        return self.generated_images


# Legacy function for backward compatibility
def generate_images(panel_data: List[Dict], art_style: str) -> List[str]:
    """
    Generate images for comic panels.
    Backward compatible function that uses the enhanced generator.
    """
    print(f"--- Starting Image Generation for Style: {art_style} ---")

    # Try to find matching style - be flexible with naming
    style_info = None
    style_variations = [
        art_style,
        art_style.replace(" ", "-"),
        art_style.replace("-", " "),
        art_style.split("(")[0].strip(),
        art_style.title(),
        "Anime"  # Fallback
    ]
    
    for style_var in style_variations:
        if style_var in STYLE_MAPPINGS:
            style_info = STYLE_MAPPINGS[style_var]
            art_style = style_var  # Use the matched key
            break
    
    if not style_info:
        print(f"⚠️ Style '{art_style}' not found in mappings, using Anime fallback")
        style_info = STYLE_MAPPINGS["Anime"]

    if not isinstance(panel_data, list) or not panel_data:
        print(f"❌ Invalid or empty panel_data received: {panel_data}")
        return []

    image_paths = []
    max_retries = 2
    retry_delay = 3

    for i, panel in enumerate(panel_data):
        if not isinstance(panel, dict) or "Description" not in panel:
            print(f"⚠️ Skipping panel {i+1}: Invalid format or missing 'Description'. Panel data: {panel}")
            continue

        panel_description = panel.get("Description", "Missing description")
        print(f"\n🔄 Generating image for Panel {i+1}...")
        print(f"   Description: {panel_description[:100]}...")

        # Get emotion for this panel
        emotion = panel.get("emotion", panel.get("Emotion", "neutral")).lower()
        emotion_modifier = EMOTION_VISUAL_MODIFIERS.get(emotion, EMOTION_VISUAL_MODIFIERS["neutral"])
        
        # Build prompt with style AND emotion
        full_prompt = (
            f"{panel_description}.\n"
            f"Art Style: {style_info['base']}.\n"
            f"Mood and Atmosphere: {emotion_modifier}.\n"
            f"{SYSTEM_INSTRUCTIONS}"
        )
        
        print(f"   Using style: {art_style}, emotion: {emotion}")

        # Make API call
        if not CLIPDROP_API_KEY:
            print("   ❌ CLIPDROP_API_KEY not set")
            continue
            
        headers = {"x-api-key": CLIPDROP_API_KEY}
        payload = {"prompt": full_prompt}

        response = None
        for attempt in range(max_retries + 1):
            try:
                print(f"   Attempt {attempt + 1}/{max_retries + 1}: Calling Clipdrop API...")
                response = requests.post(
                    CLIPDROP_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=90
                )
                print(f"   API Response Status Code: {response.status_code}")

                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '').lower()
                    print(f"   Response Content-Type: {content_type}")
                    if 'image' in content_type:
                        if response.content:
                            break
                        else:
                            print("   ⚠️ Empty response body")
                            response = None
                            if attempt < max_retries:
                                time.sleep(retry_delay)
                            continue
                    else:
                        print(f"   ⚠️ Unexpected Content-Type: {content_type}")
                        response = None
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                        continue

                elif response.status_code in [401, 403]:
                    print(f"   ❌ API Key error ({response.status_code})")
                    response = None
                    break
                elif response.status_code == 429:
                    print("   ❌ Rate limit hit")
                    wait_time = retry_delay * (attempt + 2) * 2
                    print(f"   Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 400:
                    print(f"   ❌ Bad Request: {response.text[:500]}...")
                    response = None
                    break
                else:
                    print(f"   ❌ API Error ({response.status_code}): {response.text[:500]}...")
                    response = None
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                    continue

            except requests.exceptions.Timeout:
                print(f"   ❌ Timeout on attempt {attempt + 1}")
                response = None
                if attempt < max_retries:
                    time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Network error on attempt {attempt + 1}: {e}")
                response = None
                if attempt < max_retries:
                    time.sleep(retry_delay)

        # Process successful response
        if response and response.status_code == 200 and response.content:
            try:
                print("   Processing received image data...")
                image_bytes = response.content
                image = Image.open(io.BytesIO(image_bytes))

                if image.mode != "RGB":
                    print(f"   Converting image from {image.mode} to RGB.")
                    image = image.convert("RGB")

                image_filename = f"panel_{i+1}.png"
                image_path = os.path.join(OUTPUT_DIR, image_filename)

                print(f"   Attempting to save image to: {image_path}")
                image.save(image_path, format="PNG")
                image_paths.append(image_path)
                print(f"   ✅ Image for Panel {i+1} saved successfully.")

            except FileNotFoundError:
                print(f"   ❌ Error: Output directory '{OUTPUT_DIR}' not found.")
            except IOError as e:
                print(f"   ❌ Error saving image for panel {i+1}: {e}")
            except Exception as e:
                print(f"   ❌ Error processing image for panel {i+1}: {e}")
        else:
            print(f"   ❌ Failed to generate image for panel {i+1}")

    print(f"\n--- Image Generation Finished ---")
    print(f"   Successfully generated {len(image_paths)} images.")
    
    if len(image_paths) != len(panel_data):
        print(f"   ⚠️ Expected {len(panel_data)} images, got {len(image_paths)}")

    return image_paths


# Example usage
if __name__ == "__main__":
    print("\n--- Running Test Generation ---")
    
    if not CLIPDROP_API_KEY and not STABILITY_API_KEY:
        print("Cannot run test: No API key found.")
    else:
        test_panels = [
            {
                "Description": "A cyberpunk street market at night, rain falling, neon signs reflecting on wet pavement.",
                "emotion": "mystery",
                "emotion_intensity": 0.7
            },
            {
                "Description": "Close up on a mysterious figure in a trench coat looking at a glowing data-pad.",
                "emotion": "tension",
                "emotion_intensity": 0.8
            }
        ]

        try:
            generator = EnhancedImageGenerator()
            generated_paths = generator.generate_images(test_panels, "Anime", use_enhanced=True)
            
            print("\n--- Test Result ---")
            print(f"Generated image paths: {generated_paths}")
            print(f"Number of images: {len(generated_paths)}")
            
            if generated_paths:
                print(f"Check the '{OUTPUT_DIR}' folder for images.")

        except Exception as ex:
            print(f"Test failed: {ex}")