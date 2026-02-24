"""
Enhanced Panel Generation Module
=================================
Advanced comic panel generation with:
- Character consistency injection
- Emotion-aware panel descriptions
- Adaptive panel count based on story complexity
- Multi-cultural style support
- Reference image integration

Research Contributions:
- Novel prompt engineering for consistency
- Emotion-to-description mapping
- Adaptive narrative decomposition
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai

# Import our new modules
try:
    from .consistency import CharacterConsistencyEngine, CharacterProfile
    from .emotion import EmotionAnalyzer, EmotionToPanelMapper, DynamicLayoutEngine, EmotionType
    from .styles import StyleAdapter, ComicTradition, get_tradition_from_style_name
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("⚠️ Advanced modules not available. Using basic mode.")


@dataclass
class PanelData:
    """Enhanced panel data structure."""
    panel_number: int
    description: str
    dialogue: str
    emotion: str
    emotion_intensity: float
    characters: List[str]
    prompt_modifiers: str
    relative_size: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel_number": self.panel_number,
            "Description": self.description,
            "Text": self.dialogue,
            "emotion": self.emotion,
            "emotion_intensity": self.emotion_intensity,
            "characters": self.characters,
            "prompt_modifiers": self.prompt_modifiers,
            "relative_size": self.relative_size
        }


class EnhancedPanelGenerator:
    """
    Advanced panel generation with consistency and emotion awareness.
    """
    
    def __init__(self):
        """Initialize the enhanced generator."""
        self.api_key = self._load_api_key()
        self._configure_gemini()
        
        # Initialize sub-modules if available
        if MODULES_AVAILABLE:
            self.consistency_engine = CharacterConsistencyEngine()
            self.emotion_analyzer = EmotionAnalyzer()
            self.emotion_mapper = EmotionToPanelMapper()
            self.layout_engine = DynamicLayoutEngine()
        else:
            self.consistency_engine = None
            self.emotion_analyzer = None
            self.emotion_mapper = None
            self.layout_engine = None
        
        self.style_adapter = None
        
    def _load_api_key(self) -> str:
        """Load API key from environment."""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment.")
        return api_key
    
    def _configure_gemini(self):
        """Configure Gemini API."""
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.lite_model = genai.GenerativeModel("gemini-2.5-flash")
    
    def set_style(self, style_name: str):
        """Set the comic style/tradition."""
        if MODULES_AVAILABLE:
            tradition = get_tradition_from_style_name(style_name)
            self.style_adapter = StyleAdapter(tradition)
    
    def register_reference_character(self, name: str, description: str,
                                     reference_image=None):
        """Register a character with optional reference image."""
        if self.consistency_engine:
            return self.consistency_engine.register_character(
                name=name,
                description=description,
                reference_image=reference_image
            )
        return None
    
    def generate_story(self, scenario: str, art_style: str,
                       num_panels: int = 6) -> Dict[str, str]:
        """Generate a structured story from scenario."""
        template = """
        You are a professional storyteller and comic writer.
        Given a short scenario, create a structured story optimized for a {num_panels}-panel comic.
        
        **Art Style Context:** {art_style}
        
        Provide:
        **Title**: A catchy, engaging title (max 8 words)
        
        **Characters**: List main characters with brief physical descriptions (appearance, clothing, distinctive features)
        
        **Introduction**: Set the scene and introduce characters (2-3 sentences)
        
        **Storyline**: Main events leading to climax (3-4 sentences)
        
        **Climax**: The peak moment of tension or action (2 sentences)
        
        **Resolution**: How the story concludes (1-2 sentences)
        
        **Moral/Theme**: The takeaway message (1 sentence)
        
        Format:
        # Title
        [Title text]
        
        # Characters
        [Character 1]: [Physical description]
        [Character 2]: [Physical description]
        
        # Introduction
        [Text]
        
        # Storyline
        [Text]
        
        # Climax
        [Text]
        
        # Resolution
        [Text]
        
        # Moral
        [Text]
        
        Short Scenario:
        {scenario}
        """
        
        formatted_prompt = template.format(
            scenario=scenario,
            art_style=art_style,
            num_panels=num_panels
        )
        
        response = self.model.generate_content(formatted_prompt)
        
        if not response or not response.text:
            raise Exception("Failed to generate story.")
        
        return self._extract_story_info(response.text.strip())
    
    def _extract_story_info(self, text: str) -> Dict[str, str]:
        """Extract story sections from generated text."""
        story_info = {}
        
        sections = {
            "title": r"#\s*Title\s*\n(.*?)(?=#|$)",
            "characters": r"#\s*Characters\s*\n(.*?)(?=#|$)",
            "introduction": r"#\s*Introduction\s*\n(.*?)(?=#|$)",
            "storyline": r"#\s*Storyline\s*\n(.*?)(?=#|$)",
            "climax": r"#\s*Climax\s*\n(.*?)(?=#|$)",
            "resolution": r"#\s*Resolution\s*\n(.*?)(?=#|$)",
            "moral": r"#\s*Moral\s*\n(.*?)(?=#|$)"
        }
        
        for section_name, pattern in sections.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                story_info[section_name] = match.group(1).strip()
        
        # Parse characters into consistency engine
        if "characters" in story_info and self.consistency_engine:
            self._parse_and_register_characters(story_info["characters"])
        
        return story_info
    
    def _parse_and_register_characters(self, characters_text: str):
        """Parse character descriptions and register them."""
        if not self.consistency_engine:
            return
            
        lines = characters_text.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                name = parts[0].strip().strip('*').strip('-').strip()
                description = parts[1].strip() if len(parts) > 1 else ""
                
                if name and len(name) < 50:
                    self.consistency_engine.register_character(name, description)
    
    def generate_panels_enhanced(self, scenario: str, art_style: str,
                                  num_panels: int = 6,
                                  story_data: Dict = None) -> List[Dict]:
        """Generate panel descriptions with consistency and emotion awareness."""
        self.set_style(art_style)
        
        # Get registered characters for prompt
        character_info = ""
        if self.consistency_engine and self.consistency_engine.character_registry:
            char_list = []
            for name, profile in self.consistency_engine.character_registry.items():
                char_list.append(f"- {name}: {profile.description}")
            character_info = "\n".join(char_list)
        
        template = """
        You are an expert comic book creator with deep knowledge of visual storytelling.
        
        Create exactly {num_panels} comic panels for the following story.
        
        **Art Style:** {art_style}
        
        **IMPORTANT CHARACTER CONSISTENCY:**
        {character_info}
        
        For EACH panel, maintain these exact character appearances throughout.
        
        For each panel, provide:
        1. **Description**: Detailed visual description including:
           - Background/setting details
           - Character positions and actions
           - Character appearances (MUST match established descriptions)
           - Lighting and atmosphere
           - Camera angle (if dramatic)
        
        2. **Text**: Dialogue in format "[Character]: [Dialogue]" or "..." for no dialogue
        
        3. **Emotion**: Primary emotion (joy, sadness, fear, anger, surprise, tension, excitement, love, mystery, action, comedy, dramatic, neutral)
        
        Format strictly as:
        # Panel 1
        Description: [Detailed visual description]
        Text: "[Character]: [Dialogue]"
        Emotion: [emotion word]
        
        # Panel 2
        ...
        
        # end
        
        Story Scenario:
        {scenario}
        """
        
        formatted_prompt = template.format(
            scenario=scenario,
            art_style=art_style,
            num_panels=num_panels,
            character_info=character_info if character_info else "No pre-defined characters. Create consistent character appearances."
        )
        
        response = self.model.generate_content(formatted_prompt)
        
        if not response or not response.text:
            raise Exception("Failed to generate panel descriptions.")
        
        raw_panels = self._extract_panel_info_enhanced(response.text.strip(), num_panels)
        enhanced_panels = self._enhance_panels(raw_panels, art_style)
        
        return enhanced_panels
    
    def _extract_panel_info_enhanced(self, text: str, expected_panels: int) -> List[Dict]:
        """Extract panel information from generated text."""
        panel_list = []
        panel_blocks = re.split(r"#\s*Panel\s*\d+", text)
        
        for block in panel_blocks:
            if not block.strip():
                continue
            
            panel_info = {}
            
            desc_match = re.search(r"Description:\s*(.+?)(?=Text:|Emotion:|#|$)", 
                                   block, re.IGNORECASE | re.DOTALL)
            panel_info['Description'] = desc_match.group(1).strip() if desc_match else "Scene continues."
            
            text_match = re.search(r"Text:\s*(.+?)(?=Emotion:|#|$)", 
                                   block, re.IGNORECASE | re.DOTALL)
            raw_text = text_match.group(1).strip() if text_match else "..."
            panel_info['Text'] = raw_text.strip('"').strip()
            
            emotion_match = re.search(r"Emotion:\s*(\w+)", block, re.IGNORECASE)
            panel_info['Emotion'] = emotion_match.group(1).lower() if emotion_match else "neutral"
            
            panel_list.append(panel_info)
        
        while len(panel_list) < expected_panels:
            panel_list.append({
                'Description': "Scene continues with characters.",
                'Text': "...",
                'Emotion': "neutral"
            })
        
        return panel_list[:expected_panels]
    
    def _enhance_panels(self, raw_panels: List[Dict], art_style: str) -> List[Dict]:
        """Enhance panels with consistency, emotion, and style information."""
        enhanced = []
        
        for i, panel in enumerate(raw_panels):
            combined_text = f"{panel['Description']} {panel['Text']}"
            
            # Analyze emotion if module available
            if self.emotion_analyzer:
                detected_emotion, intensity = self.emotion_analyzer.analyze(combined_text)
                emotion_value = detected_emotion.value
            else:
                emotion_value = panel.get('Emotion', 'neutral')
                intensity = 0.5
            
            # Get panel style if module available
            if self.emotion_mapper:
                panel_style, _, _ = self.emotion_mapper.get_style_for_text(combined_text)
                prompt_modifiers = panel_style.to_prompt_modifiers()
                relative_size = panel_style.relative_size
                panel_style_dict = {
                    "aspect_ratio": panel_style.aspect_ratio,
                    "border_style": panel_style.border_style,
                    "color_temperature": panel_style.color_temperature,
                    "camera_angle": panel_style.camera_angle
                }
            else:
                prompt_modifiers = ""
                relative_size = "medium"
                panel_style_dict = {}
            
            # Extract characters mentioned
            characters = []
            if self.consistency_engine:
                for name in self.consistency_engine.character_registry.keys():
                    if name.lower() in combined_text.lower():
                        characters.append(name)
            
            # Generate consistency-enhanced description
            if self.consistency_engine:
                enhanced_desc = self.consistency_engine.generate_consistent_prompt(
                    panel['Description'], panel_number=i + 1
                )
            else:
                enhanced_desc = panel['Description']
            
            # Add style modifiers
            if self.style_adapter:
                enhanced_desc = self.style_adapter.adapt_prompt(enhanced_desc)
            
            enhanced_panel = {
                "panel_number": i + 1,
                "Description": enhanced_desc,
                "Text": panel['Text'],
                "emotion": emotion_value,
                "emotion_intensity": intensity,
                "characters": characters,
                "prompt_modifiers": prompt_modifiers,
                "relative_size": relative_size,
                "panel_style": panel_style_dict
            }
            
            enhanced.append(enhanced_panel)
        
        return enhanced
    
    def get_emotional_arc(self, panels: List[Dict]) -> Dict[str, Any]:
        """Get the emotional arc analysis for the comic."""
        emotions = [p.get('emotion', 'neutral') for p in panels]
        intensities = [p.get('emotion_intensity', 0.5) for p in panels]
        
        climax_idx = intensities.index(max(intensities)) if intensities else -1
        
        return {
            "emotions": emotions,
            "intensities": intensities,
            "climax_panel": climax_idx + 1,
            "emotional_variety": len(set(emotions)) / len(emotions) if emotions else 0,
            "average_intensity": sum(intensities) / len(intensities) if intensities else 0
        }
    
    def get_character_summary(self) -> Dict[str, Dict]:
        """Get summary of all registered characters."""
        if self.consistency_engine:
            return self.consistency_engine.get_character_summary()
        return {}
    
    def clear_session(self):
        """Clear all session data for new comic."""
        if self.consistency_engine:
            self.consistency_engine.clear()


# Legacy compatibility functions
def load_api_keys():
    """Loads API keys from .env file."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
    return api_key


def configure_gemini(api_key):
    """Configures the Gemini AI model."""
    genai.configure(api_key=api_key)


def generate_panels(scenario: str, art_style: str, num_panels: int = 6) -> List[Dict]:
    """Legacy function for backwards compatibility - now supports variable panel counts."""
    template = """
    You are a professional comic book creator.
    You will be given a short scenario, and you must split it into exactly {num_panels} comic panels.
    **Art Style:** {art_style}
    For each comic panel, provide:
    1. **Description**: A detailed background and character description (comma-separated, not full sentences).
    2. **Text**: Exact dialogue in quotation marks, or if no dialogue, leave it empty or use `...`.
    3. **Emotion**: The primary emotion of the scene (joy, sadness, fear, anger, surprise, tension, excitement, love, mystery, action, comedy, dramatic, neutral)
    Ensure all text is clear, meaningful, and in proper English.
    Format:
    # Panel 1
    Description: [Background and character details]
    Text: "[Character]: [Dialogue]" OR "..." if no dialogue.
    Emotion: [emotion word]
    
    # Panel 2
    Description: [Background and character details]
    Text: "[Character]: [Dialogue]" OR "..." if no dialogue.
    Emotion: [emotion word]
    
    ... continue for all {num_panels} panels ...
    
    # end
    Short Scenario:
    {scenario}
    """
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    formatted_prompt = template.format(scenario=scenario, art_style=art_style, num_panels=num_panels)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(formatted_prompt)
    
    if not response or not response.text:
        raise Exception("Error: Failed to generate panel descriptions.")
    
    return extract_panel_info_flexible(response.text.strip(), num_panels)


def extract_panel_info_flexible(text: str, expected_panels: int = 6) -> List[Dict]:
    """Extracts structured panel descriptions with flexible panel count."""
    panel_info_list = []
    panel_blocks = re.split(r"#\s*Panel\s*\d+", text)
    
    for block in panel_blocks:
        if not block.strip():
            continue
        
        panel_info = {}
        
        # Extract description
        desc_match = re.search(r"Description:\s*(.+?)(?=Text:|Emotion:|#|$)", block, re.IGNORECASE | re.DOTALL)
        panel_info['Description'] = desc_match.group(1).strip() if desc_match else "Scene continues."
        
        # Extract text/dialogue
        text_match = re.search(r"Text:\s*(.+?)(?=Emotion:|#|$)", block, re.IGNORECASE | re.DOTALL)
        raw_text = text_match.group(1).strip() if text_match else "..."
        panel_info['Text'] = raw_text.strip('"').strip()
        
        # Extract emotion
        emotion_match = re.search(r"Emotion:\s*(\w+)", block, re.IGNORECASE)
        emotion = emotion_match.group(1).lower() if emotion_match else "neutral"
        panel_info['emotion'] = emotion
        panel_info['Emotion'] = emotion
        
        # Set default panel style properties
        panel_info['relative_size'] = 'medium'
        panel_info['emotion_intensity'] = 0.5
        panel_info['characters'] = []
        panel_info['prompt_modifiers'] = ''
        
        panel_info_list.append(panel_info)
    
    # Ensure we have at least 1 panel
    if len(panel_info_list) < 1:
        raise ValueError(f"Failed to parse any panels from the response.")
    
    # Pad to expected panel count if needed
    while len(panel_info_list) < expected_panels:
        panel_info_list.append({
            "Description": "Scene continues with the story.",
            "Text": "...",
            "emotion": "neutral",
            "Emotion": "neutral",
            "relative_size": "medium",
            "emotion_intensity": 0.5,
            "characters": [],
            "prompt_modifiers": ""
        })
    
    # Trim if we got too many
    return panel_info_list[:expected_panels]


def extract_panel_info(text):
    """Legacy extract function - redirects to flexible version."""
    return extract_panel_info_flexible(text, 6)


def generate_story(scenario: str, art_style: str) -> Dict[str, str]:
    """Legacy function for story generation."""
    template = """
    You are a professional storyteller.
    Given a short scenario, create a structured story with a title and exactly 4 sections:
    **Title**: A catchy title for the story.
    1. **Introduction**: Introduce the main character(s) and setting in 1-2 sentences.
    2. **Storyline**: Describe the sequence of events leading to the climax in 2-3 sentences.
    3. **Climax**: Highlight the peak action or turning point in 1-2 sentences.
    4. **Moral**: Provide a concluding lesson or takeaway in 1 sentence.
    **Art Style Context:** {art_style}
    Format:
    # Title
    [Title text]
    # Introduction
    [Text]
    # Storyline
    [Text]
    # Climax
    [Text]
    # Moral
    [Text]
    Short Scenario:
    {scenario}
    """
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    formatted_prompt = template.format(scenario=scenario, art_style=art_style)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(formatted_prompt)
    
    if not response or not response.text:
        raise Exception("Error: Failed to generate story.")
    
    return extract_story_info(response.text.strip())


def extract_story_info(text):
    """Extracts the story sections from the generated text."""
    story_info = {}
    sections = re.split(r"#\s*(\w+)", text)
    for i in range(1, len(sections), 2):
        section_name = sections[i].lower()
        section_content = sections[i+1].strip()
        if section_name in ["title", "introduction", "storyline", "climax", "moral"]:
            story_info[section_name] = section_content
    return story_info


if __name__ == '__main__':
    try:
        generator = EnhancedPanelGenerator()
        
        generator.register_reference_character(
            "Luna",
            "Young girl with long silver hair, blue eyes, wearing a purple wizard robe"
        )
        
        scenario = "A young wizard discovers her powers during a magical storm"
        
        print("\n--- Generating Story ---")
        story = generator.generate_story(scenario, "Anime")
        for key, value in story.items():
            print(f"\n{key.upper()}: {value}")
        
        print("\n--- Generating Panels ---")
        panels = generator.generate_panels_enhanced(scenario, "Anime", 6, story)
        
        for panel in panels:
            print(f"\nPanel {panel['panel_number']}:")
            print(f"  Emotion: {panel['emotion']} (intensity: {panel['emotion_intensity']})")
            print(f"  Size: {panel['relative_size']}")
            print(f"  Characters: {panel['characters']}")
            print(f"  Description: {panel['Description'][:100]}...")
            print(f"  Dialogue: {panel['Text']}")
        
        print("\n--- Emotional Arc ---")
        arc = generator.get_emotional_arc(panels)
        print(f"  Climax at panel: {arc['climax_panel']}")
        print(f"  Emotional variety: {arc['emotional_variety']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
