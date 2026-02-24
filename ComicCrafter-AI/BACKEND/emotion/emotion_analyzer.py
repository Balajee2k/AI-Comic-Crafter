"""
Emotion Analyzer and Panel Layout Engine
=========================================
Maps narrative emotional content to visual comic grammar.

This module implements a novel approach to comic generation by:
1. Analyzing emotional content of each panel's narrative
2. Adjusting visual parameters (layout, color, composition) accordingly
3. Creating dynamic panel layouts that reflect story pacing

Comic Visual Grammar Rules (Research-Based):
- Tension: Wide panels, Dutch angles, cool colors, sharp borders
- Joy: Open panels, warm colors, rounded elements
- Fear: Claustrophobic small panels, shadows, desaturated colors
- Action: Dynamic angles, motion lines, irregular borders
- Sadness: Muted colors, isolated compositions, heavy negative space
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import re
import math


class EmotionType(Enum):
    """
    Core emotions mapped to visual comic grammar.
    Based on Plutchik's wheel of emotions adapted for visual storytelling.
    """
    NEUTRAL = "neutral"
    JOY = "joy"
    SADNESS = "sadness"
    FEAR = "fear"
    ANGER = "anger"
    SURPRISE = "surprise"
    TENSION = "tension"
    EXCITEMENT = "excitement"
    LOVE = "love"
    MYSTERY = "mystery"
    ACTION = "action"
    COMEDY = "comedy"
    DRAMATIC = "dramatic"


@dataclass
class PanelStyle:
    """
    Visual styling parameters for a comic panel.
    These parameters guide both image generation and layout.
    """
    # Layout parameters
    aspect_ratio: Tuple[float, float] = (1.0, 1.0)  # width, height ratio
    relative_size: str = "medium"  # "small", "medium", "large", "splash"
    panel_weight: float = 1.0  # For layout calculations
    
    # Border styling
    border_style: str = "solid"  # "solid", "wavy", "broken", "none", "double"
    border_thickness: int = 3
    border_color: str = "#000000"
    
    # Visual atmosphere
    color_temperature: str = "neutral"  # "warm", "cool", "neutral", "desaturated"
    brightness: float = 1.0  # 0.5 = darker, 1.5 = brighter
    contrast: float = 1.0  # Visual contrast level
    saturation: float = 1.0  # Color saturation
    
    # Composition
    camera_angle: str = "eye-level"  # "low", "high", "dutch", "bird-eye", "worm-eye"
    composition_style: str = "balanced"  # "balanced", "rule-of-thirds", "dynamic", "centered"
    depth_of_field: str = "normal"  # "shallow", "normal", "deep"
    
    # Visual effects
    motion_lines: bool = False
    speed_lines: bool = False
    impact_effects: bool = False
    emotional_overlay: Optional[str] = None  # "glow", "shadow", "vignette"
    
    def to_prompt_modifiers(self) -> str:
        """Generate prompt modifiers for image generation."""
        modifiers = []
        
        # Camera angle
        angle_map = {
            "low": "low angle shot, looking up",
            "high": "high angle shot, looking down", 
            "dutch": "dutch angle, tilted camera",
            "bird-eye": "bird's eye view, overhead shot",
            "worm-eye": "worm's eye view, extreme low angle"
        }
        if self.camera_angle in angle_map:
            modifiers.append(angle_map[self.camera_angle])
        
        # Color temperature
        temp_map = {
            "warm": "warm color palette, golden tones, orange and red hues",
            "cool": "cool color palette, blue and cyan tones",
            "desaturated": "desaturated colors, muted tones, low saturation"
        }
        if self.color_temperature in temp_map:
            modifiers.append(temp_map[self.color_temperature])
        
        # Effects
        if self.motion_lines:
            modifiers.append("motion blur, speed lines, dynamic movement")
        if self.impact_effects:
            modifiers.append("impact effects, action lines, dramatic composition")
        
        # Brightness
        if self.brightness < 0.8:
            modifiers.append("dark atmosphere, shadowy, low-key lighting")
        elif self.brightness > 1.2:
            modifiers.append("bright lighting, high-key, luminous")
        
        # Contrast
        if self.contrast > 1.3:
            modifiers.append("high contrast, dramatic shadows, chiaroscuro")
        
        return ", ".join(modifiers)


@dataclass
class EmotionalArc:
    """
    Represents the emotional arc of the entire story.
    Used for pacing and layout decisions.
    """
    panels: List[EmotionType] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    climax_panel: int = -1
    
    def add_panel(self, emotion: EmotionType, intensity: float):
        """Add a panel's emotion to the arc."""
        self.panels.append(emotion)
        self.intensities.append(min(max(intensity, 0.0), 1.0))
        
        # Update climax position
        if intensity > 0.8:
            self.climax_panel = len(self.panels) - 1
    
    def get_pacing(self) -> List[str]:
        """Determine story pacing based on emotional arc."""
        pacing = []
        for i, intensity in enumerate(self.intensities):
            if intensity < 0.3:
                pacing.append("slow")
            elif intensity < 0.6:
                pacing.append("moderate")
            elif intensity < 0.85:
                pacing.append("fast")
            else:
                pacing.append("climax")
        return pacing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "panels": [e.value for e in self.panels],
            "intensities": self.intensities,
            "climax_panel": self.climax_panel,
            "pacing": self.get_pacing()
        }


class EmotionAnalyzer:
    """
    Analyzes text to determine emotional content.
    Uses keyword matching and pattern recognition.
    Can be extended with transformer-based sentiment analysis.
    """
    
    # Emotion keywords (research-based lexicon)
    EMOTION_LEXICON = {
        EmotionType.JOY: [
            'happy', 'joy', 'laugh', 'smile', 'celebrate', 'wonderful', 
            'excited', 'delighted', 'cheerful', 'fun', 'love', 'amazing',
            'fantastic', 'great', 'beautiful', 'bright', 'success', 'victory'
        ],
        EmotionType.SADNESS: [
            'sad', 'cry', 'tears', 'grief', 'sorrow', 'lonely', 'depressed',
            'melancholy', 'heartbroken', 'loss', 'mourn', 'weep', 'tragic',
            'despair', 'miserable', 'gloomy', 'farewell', 'goodbye'
        ],
        EmotionType.FEAR: [
            'afraid', 'scared', 'terror', 'horror', 'frightened', 'panic',
            'dread', 'nightmare', 'creepy', 'spooky', 'dark', 'shadow',
            'monster', 'danger', 'threat', 'anxious', 'worried', 'trembling'
        ],
        EmotionType.ANGER: [
            'angry', 'rage', 'fury', 'mad', 'hate', 'furious', 'outraged',
            'irritated', 'frustrated', 'violent', 'aggressive', 'hostile',
            'revenge', 'scream', 'yell', 'punch', 'fight', 'destroy'
        ],
        EmotionType.SURPRISE: [
            'surprise', 'shock', 'unexpected', 'sudden', 'amazed', 'astonished',
            'startled', 'stunned', 'gasp', 'unbelievable', 'incredible', 'twist',
            'reveal', 'discover', 'realize', 'revelation'
        ],
        EmotionType.TENSION: [
            'tension', 'suspense', 'nervous', 'anxious', 'wait', 'anticipation',
            'uncertain', 'edge', 'stress', 'pressure', 'intense', 'standoff',
            'confrontation', 'standstill', 'hesitate', 'doubt'
        ],
        EmotionType.EXCITEMENT: [
            'exciting', 'thrill', 'adventure', 'rush', 'adrenaline', 'fast',
            'chase', 'escape', 'race', 'action', 'dynamic', 'energy',
            'explosive', 'intense', 'wild', 'extreme'
        ],
        EmotionType.LOVE: [
            'love', 'romance', 'kiss', 'embrace', 'heart', 'passion', 'tender',
            'affection', 'care', 'warm', 'together', 'forever', 'soulmate',
            'beloved', 'cherish', 'adore'
        ],
        EmotionType.MYSTERY: [
            'mystery', 'secret', 'hidden', 'clue', 'detective', 'investigate',
            'strange', 'curious', 'enigma', 'puzzle', 'unknown', 'shadow',
            'fog', 'whisper', 'lurk', 'suspicious'
        ],
        EmotionType.ACTION: [
            'fight', 'battle', 'attack', 'defend', 'run', 'jump', 'kick',
            'punch', 'sword', 'explosion', 'crash', 'chase', 'escape',
            'combat', 'warrior', 'hero', 'power'
        ],
        EmotionType.COMEDY: [
            'funny', 'joke', 'laugh', 'silly', 'humor', 'comic', 'ridiculous',
            'absurd', 'goofy', 'hilarious', 'wit', 'prank', 'comical',
            'amusing', 'playful', 'whimsical'
        ],
        EmotionType.DRAMATIC: [
            'dramatic', 'intense', 'powerful', 'epic', 'destiny', 'fate',
            'sacrifice', 'betray', 'oath', 'honor', 'glory', 'fallen',
            'rise', 'legend', 'prophecy', 'ultimate'
        ]
    }
    
    # Intensity modifiers
    INTENSITY_BOOSTERS = [
        'very', 'extremely', 'incredibly', 'absolutely', 'utterly',
        'completely', 'totally', 'so', 'really', 'deeply', 'intensely'
    ]
    
    INTENSITY_DAMPENERS = [
        'slightly', 'somewhat', 'a bit', 'little', 'barely', 'hardly',
        'mildly', 'gently', 'softly', 'quietly'
    ]
    
    def __init__(self):
        """Initialize the emotion analyzer."""
        self._build_word_index()
    
    def _build_word_index(self):
        """Build reverse index for quick lookups."""
        self.word_to_emotion = {}
        for emotion, words in self.EMOTION_LEXICON.items():
            for word in words:
                if word not in self.word_to_emotion:
                    self.word_to_emotion[word] = []
                self.word_to_emotion[word].append(emotion)
    
    def analyze(self, text: str) -> Tuple[EmotionType, float]:
        """
        Analyze text and return primary emotion with intensity.
        
        Args:
            text: Text to analyze (panel description or dialogue)
        
        Returns:
            Tuple of (EmotionType, intensity between 0-1)
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Count emotion matches
        emotion_scores = {e: 0.0 for e in EmotionType}
        
        for i, word in enumerate(words):
            if word in self.word_to_emotion:
                for emotion in self.word_to_emotion[word]:
                    score = 1.0
                    
                    # Check for intensity modifiers
                    if i > 0:
                        prev_word = words[i-1]
                        if prev_word in self.INTENSITY_BOOSTERS:
                            score *= 1.5
                        elif prev_word in self.INTENSITY_DAMPENERS:
                            score *= 0.6
                    
                    emotion_scores[emotion] += score
        
        # Find dominant emotion
        if max(emotion_scores.values()) == 0:
            return EmotionType.NEUTRAL, 0.3
        
        dominant = max(emotion_scores, key=emotion_scores.get)
        
        # Calculate intensity (normalized)
        raw_intensity = emotion_scores[dominant]
        # Normalize: 1 match = 0.4, 2 = 0.6, 3+ = 0.8+
        intensity = min(0.4 + (raw_intensity - 1) * 0.2, 1.0)
        
        # Boost for exclamation marks and caps
        if '!' in text:
            intensity = min(intensity + 0.1, 1.0)
        if text.upper() == text and len(text) > 10:
            intensity = min(intensity + 0.15, 1.0)
        
        return dominant, round(intensity, 2)
    
    def analyze_panels(self, panel_texts: List[str]) -> EmotionalArc:
        """
        Analyze all panels and build emotional arc.
        
        Args:
            panel_texts: List of panel descriptions/dialogues
        
        Returns:
            EmotionalArc object
        """
        arc = EmotionalArc()
        
        for text in panel_texts:
            emotion, intensity = self.analyze(text)
            arc.add_panel(emotion, intensity)
        
        return arc


class EmotionToPanelMapper:
    """
    Maps emotions to visual panel styling.
    Implements comic visual grammar based on emotional content.
    """
    
    # Mapping of emotions to panel styles
    EMOTION_STYLE_MAP = {
        EmotionType.NEUTRAL: PanelStyle(
            aspect_ratio=(1.0, 1.0),
            relative_size="medium",
            border_style="solid",
            color_temperature="neutral",
            camera_angle="eye-level"
        ),
        EmotionType.JOY: PanelStyle(
            aspect_ratio=(1.2, 1.0),
            relative_size="medium",
            border_style="none",
            border_thickness=0,
            color_temperature="warm",
            brightness=1.15,
            saturation=1.2,
            camera_angle="eye-level",
            composition_style="balanced",
            emotional_overlay="glow"
        ),
        EmotionType.SADNESS: PanelStyle(
            aspect_ratio=(1.0, 1.2),
            relative_size="medium",
            border_style="solid",
            border_thickness=2,
            color_temperature="desaturated",
            brightness=0.85,
            saturation=0.7,
            contrast=0.9,
            camera_angle="high",
            composition_style="centered",
            emotional_overlay="vignette"
        ),
        EmotionType.FEAR: PanelStyle(
            aspect_ratio=(0.8, 1.0),
            relative_size="small",
            border_style="broken",
            border_thickness=4,
            color_temperature="cool",
            brightness=0.7,
            contrast=1.4,
            saturation=0.6,
            camera_angle="dutch",
            composition_style="dynamic",
            emotional_overlay="shadow"
        ),
        EmotionType.ANGER: PanelStyle(
            aspect_ratio=(1.3, 1.0),
            relative_size="large",
            border_style="double",
            border_thickness=5,
            border_color="#AA0000",
            color_temperature="warm",
            brightness=1.1,
            contrast=1.3,
            camera_angle="low",
            composition_style="dynamic",
            impact_effects=True
        ),
        EmotionType.SURPRISE: PanelStyle(
            aspect_ratio=(1.0, 1.0),
            relative_size="large",
            border_style="none",
            color_temperature="neutral",
            brightness=1.2,
            camera_angle="eye-level",
            composition_style="centered"
        ),
        EmotionType.TENSION: PanelStyle(
            aspect_ratio=(2.0, 1.0),
            relative_size="large",
            border_style="solid",
            border_thickness=4,
            color_temperature="cool",
            brightness=0.9,
            contrast=1.2,
            camera_angle="dutch",
            composition_style="rule-of-thirds"
        ),
        EmotionType.EXCITEMENT: PanelStyle(
            aspect_ratio=(1.5, 1.0),
            relative_size="large",
            border_style="solid",
            color_temperature="warm",
            brightness=1.1,
            saturation=1.3,
            camera_angle="dynamic",
            composition_style="dynamic",
            motion_lines=True,
            speed_lines=True
        ),
        EmotionType.ACTION: PanelStyle(
            aspect_ratio=(1.4, 1.0),
            relative_size="large",
            border_style="broken",
            border_thickness=3,
            color_temperature="neutral",
            contrast=1.2,
            camera_angle="dutch",
            composition_style="dynamic",
            motion_lines=True,
            impact_effects=True
        ),
        EmotionType.LOVE: PanelStyle(
            aspect_ratio=(1.0, 1.0),
            relative_size="medium",
            border_style="none",
            color_temperature="warm",
            brightness=1.1,
            saturation=1.1,
            camera_angle="eye-level",
            composition_style="centered",
            depth_of_field="shallow",
            emotional_overlay="glow"
        ),
        EmotionType.MYSTERY: PanelStyle(
            aspect_ratio=(1.0, 1.3),
            relative_size="medium",
            border_style="solid",
            color_temperature="cool",
            brightness=0.75,
            contrast=1.3,
            saturation=0.8,
            camera_angle="high",
            depth_of_field="shallow",
            emotional_overlay="shadow"
        ),
        EmotionType.COMEDY: PanelStyle(
            aspect_ratio=(1.0, 1.0),
            relative_size="medium",
            border_style="wavy",
            color_temperature="warm",
            brightness=1.2,
            saturation=1.2,
            camera_angle="eye-level",
            composition_style="balanced"
        ),
        EmotionType.DRAMATIC: PanelStyle(
            aspect_ratio=(1.5, 1.0),
            relative_size="splash",
            border_style="double",
            border_thickness=4,
            color_temperature="cool",
            contrast=1.4,
            camera_angle="low",
            composition_style="rule-of-thirds",
            emotional_overlay="vignette"
        )
    }
    
    def __init__(self):
        """Initialize the mapper."""
        self.emotion_analyzer = EmotionAnalyzer()
    
    def get_style_for_emotion(self, emotion: EmotionType, 
                              intensity: float = 0.5) -> PanelStyle:
        """
        Get panel style for an emotion with intensity adjustment.
        
        Args:
            emotion: The emotion type
            intensity: Emotion intensity (0-1)
        
        Returns:
            PanelStyle with adjusted parameters
        """
        base_style = self.EMOTION_STYLE_MAP.get(emotion, 
                                                 self.EMOTION_STYLE_MAP[EmotionType.NEUTRAL])
        
        # Create a copy and adjust based on intensity
        style = PanelStyle(
            aspect_ratio=base_style.aspect_ratio,
            relative_size=base_style.relative_size,
            panel_weight=1.0 + (intensity * 0.5),  # Higher intensity = more weight
            border_style=base_style.border_style,
            border_thickness=base_style.border_thickness,
            border_color=base_style.border_color,
            color_temperature=base_style.color_temperature,
            brightness=base_style.brightness,
            contrast=base_style.contrast + (intensity * 0.2),  # More contrast at high intensity
            saturation=base_style.saturation,
            camera_angle=base_style.camera_angle,
            composition_style=base_style.composition_style,
            depth_of_field=base_style.depth_of_field,
            motion_lines=base_style.motion_lines,
            speed_lines=base_style.speed_lines,
            impact_effects=base_style.impact_effects or intensity > 0.8,
            emotional_overlay=base_style.emotional_overlay
        )
        
        # Upgrade to splash panel at very high intensity
        if intensity > 0.9 and style.relative_size != "splash":
            style.relative_size = "large"
        
        return style
    
    def get_style_for_text(self, text: str) -> Tuple[PanelStyle, EmotionType, float]:
        """
        Analyze text and return appropriate panel style.
        
        Args:
            text: Panel description or dialogue
        
        Returns:
            Tuple of (PanelStyle, emotion, intensity)
        """
        emotion, intensity = self.emotion_analyzer.analyze(text)
        style = self.get_style_for_emotion(emotion, intensity)
        return style, emotion, intensity


class DynamicLayoutEngine:
    """
    Generates dynamic panel layouts based on emotional arc.
    
    Novel Contribution:
    - Non-uniform panel grids based on story pacing
    - Climax-aware layout sizing
    - Visual rhythm through panel arrangement
    """
    
    # Size multipliers for relative sizes
    SIZE_MULTIPLIERS = {
        "small": 0.7,
        "medium": 1.0,
        "large": 1.4,
        "splash": 2.0
    }
    
    def __init__(self):
        """Initialize the layout engine."""
        self.mapper = EmotionToPanelMapper()
    
    def generate_layout(self, panel_styles: List[PanelStyle], 
                       canvas_width: int = 1200,
                       canvas_height: int = 1800,
                       num_columns: int = 2) -> List[Dict[str, Any]]:
        """
        Generate dynamic layout positions for panels.
        
        Args:
            panel_styles: List of PanelStyle for each panel
            canvas_width: Total canvas width
            canvas_height: Total canvas height
            num_columns: Base number of columns
        
        Returns:
            List of layout dictionaries with position and size info
        """
        layouts = []
        total_weight = sum(s.panel_weight for s in panel_styles)
        
        # Calculate available height per "unit weight"
        num_panels = len(panel_styles)
        num_rows = math.ceil(num_panels / num_columns)
        
        margin = 15
        gutter = 10
        
        available_width = canvas_width - (2 * margin) - ((num_columns - 1) * gutter)
        available_height = canvas_height - (2 * margin) - ((num_rows - 1) * gutter)
        
        base_cell_width = available_width // num_columns
        base_cell_height = available_height // num_rows
        
        current_y = margin
        current_row = 0
        panels_in_row = 0
        row_panels = []
        
        for i, style in enumerate(panel_styles):
            is_splash = style.relative_size == "splash"
            is_large = style.relative_size == "large"
            
            # Determine panel dimensions
            if is_splash:
                # Splash panel takes full width
                width = available_width
                height = int(base_cell_height * 1.5)
                col_span = num_columns
            elif is_large:
                # Large panel takes 1.4x space
                width = int(base_cell_width * 1.3)
                height = int(base_cell_height * 1.2)
                col_span = 1
            else:
                multiplier = self.SIZE_MULTIPLIERS.get(style.relative_size, 1.0)
                width = int(base_cell_width * multiplier)
                height = int(base_cell_height * multiplier)
                col_span = 1
            
            # Apply aspect ratio
            ar_w, ar_h = style.aspect_ratio
            if ar_w > ar_h:
                width = int(width * (ar_w / ar_h))
            else:
                height = int(height * (ar_h / ar_w))
            
            # Clamp dimensions
            width = min(width, available_width)
            height = min(height, base_cell_height * 2)
            
            layout = {
                "panel_index": i,
                "x": 0,  # Will be calculated
                "y": 0,  # Will be calculated
                "width": width,
                "height": height,
                "style": style,
                "col_span": col_span,
                "border_style": style.border_style,
                "border_thickness": style.border_thickness,
                "border_color": style.border_color
            }
            
            layouts.append(layout)
        
        # Simple row-based positioning
        current_x = margin
        current_y = margin
        row_height = 0
        
        for layout in layouts:
            if current_x + layout["width"] > canvas_width - margin:
                # Move to next row
                current_x = margin
                current_y += row_height + gutter
                row_height = 0
            
            layout["x"] = current_x
            layout["y"] = current_y
            
            current_x += layout["width"] + gutter
            row_height = max(row_height, layout["height"])
        
        return layouts
    
    def generate_adaptive_layout(self, panel_texts: List[str],
                                 num_panels: int = 6,
                                 canvas_width: int = 1200,
                                 canvas_height: int = 1800) -> Tuple[List[Dict], EmotionalArc]:
        """
        Generate a complete adaptive layout based on panel text analysis.
        
        Args:
            panel_texts: List of panel descriptions
            num_panels: Expected number of panels
            canvas_width: Canvas width
            canvas_height: Canvas height
        
        Returns:
            Tuple of (layouts list, emotional arc)
        """
        # Analyze each panel
        styles = []
        emotions = []
        intensities = []
        
        for text in panel_texts:
            style, emotion, intensity = self.mapper.get_style_for_text(text)
            styles.append(style)
            emotions.append(emotion)
            intensities.append(intensity)
        
        # Build emotional arc
        arc = EmotionalArc()
        for e, i in zip(emotions, intensities):
            arc.add_panel(e, i)
        
        # Find climax and adjust sizes
        if arc.climax_panel >= 0:
            styles[arc.climax_panel].relative_size = "splash"
            styles[arc.climax_panel].panel_weight = 2.0
        
        # Generate layout
        layouts = self.generate_layout(styles, canvas_width, canvas_height)
        
        # Add emotion info to layouts
        for i, layout in enumerate(layouts):
            layout["emotion"] = emotions[i].value
            layout["intensity"] = intensities[i]
            layout["prompt_modifiers"] = styles[i].to_prompt_modifiers()
        
        return layouts, arc


# Convenience functions
def analyze_emotion(text: str) -> Tuple[EmotionType, float]:
    """Quick emotion analysis."""
    analyzer = EmotionAnalyzer()
    return analyzer.analyze(text)


def get_panel_style(text: str) -> PanelStyle:
    """Get panel style for text."""
    mapper = EmotionToPanelMapper()
    style, _, _ = mapper.get_style_for_text(text)
    return style


def create_layout_engine() -> DynamicLayoutEngine:
    """Factory function for layout engine."""
    return DynamicLayoutEngine()
