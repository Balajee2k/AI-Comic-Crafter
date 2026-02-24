"""
Comic Tradition Style Adapter
==============================
Adapts comic generation to different cultural comic traditions.

This module implements style adaptation for:
1. Japanese Manga
2. Japanese Anime
3. American Superhero Comics
4. Franco-Belgian (Tintin style)
5. Korean Manhwa
6. Chinese Manhua
7. Webcomic styles
8. Indie Graphic Novels
9. Classic Newspaper Comics
10. Horror Comics

Each tradition has specific:
- Visual style parameters
- Panel layout conventions
- Reading direction
- Color palette preferences
- Character design standards
- Background styles
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


class ComicTradition(Enum):
    """
    Enumeration of comic traditions from around the world.
    """
    # Japanese
    MANGA = "manga"
    ANIME = "anime"
    
    # Western
    AMERICAN_SUPERHERO = "american_superhero"
    AMERICAN_INDIE = "american_indie"
    
    # European
    FRANCO_BELGIAN = "franco_belgian"
    BRITISH = "british"
    
    # Asian
    MANHWA = "manhwa"  # Korean
    MANHUA = "manhua"  # Chinese
    
    # Digital/Modern
    WEBCOMIC = "webcomic"
    WEBTOON = "webtoon"
    
    # Genre-specific
    HORROR = "horror"
    COMEDY = "comedy"
    NOIR = "noir"
    
    # Classic
    NEWSPAPER = "newspaper"
    GOLDEN_AGE = "golden_age"


@dataclass
class TraditionParameters:
    """
    Visual and layout parameters for a comic tradition.
    """
    # Reading direction
    reading_direction: str = "ltr"  # "ltr" (left-to-right) or "rtl" (right-to-left)
    panel_flow: str = "western"  # "western" (L-R, T-B) or "manga" (R-L, T-B)
    
    # Visual style
    line_style: str = "clean"  # "clean", "sketchy", "bold", "thin", "variable"
    shading_style: str = "flat"  # "flat", "cel", "screen-tones", "hatching", "gradient"
    color_mode: str = "color"  # "color", "grayscale", "black-white", "limited"
    
    # Color palette
    color_temperature: str = "neutral"  # "warm", "cool", "neutral"
    saturation: str = "normal"  # "vibrant", "normal", "muted", "desaturated"
    contrast: str = "normal"  # "high", "normal", "low"
    
    # Character design
    eye_style: str = "realistic"  # "large", "realistic", "stylized", "minimal"
    body_proportions: str = "realistic"  # "realistic", "heroic", "chibi", "elongated"
    expression_style: str = "subtle"  # "subtle", "exaggerated", "dramatic"
    
    # Panel characteristics
    panel_shape: str = "rectangular"  # "rectangular", "dynamic", "circular", "irregular"
    gutter_style: str = "white"  # "white", "black", "none", "colored"
    border_style: str = "thin"  # "thin", "thick", "none", "decorative"
    
    # Background style
    background_detail: str = "moderate"  # "detailed", "moderate", "minimal", "abstract"
    speed_lines: bool = False
    emotional_backgrounds: bool = False
    
    # Special effects
    motion_effects: str = "none"  # "none", "lines", "blur", "afterimages"
    sound_effects: str = "none"  # "none", "integrated", "separate"
    
    # Typical aspect ratio for single panel
    typical_aspect_ratio: Tuple[float, float] = (1.0, 1.0)
    
    # Prompt style modifiers
    style_keywords: List[str] = field(default_factory=list)
    
    # System prompt additions
    system_prompt_additions: str = ""


# Complete tradition definitions
TRADITION_CONFIGS: Dict[ComicTradition, TraditionParameters] = {
    
    ComicTradition.MANGA: TraditionParameters(
        reading_direction="rtl",
        panel_flow="manga",
        line_style="clean",
        shading_style="screen-tones",
        color_mode="black-white",
        color_temperature="neutral",
        saturation="desaturated",
        contrast="high",
        eye_style="large",
        body_proportions="realistic",
        expression_style="exaggerated",
        panel_shape="dynamic",
        gutter_style="none",
        border_style="thin",
        background_detail="minimal",
        speed_lines=True,
        emotional_backgrounds=True,
        motion_effects="lines",
        typical_aspect_ratio=(0.7, 1.0),
        style_keywords=[
            "manga style", "black and white", "screen tones",
            "clean lineart", "expressive eyes", "dynamic poses",
            "speed lines", "dramatic shading", "Japanese manga"
        ],
        system_prompt_additions="""
        Use classic Japanese manga visual language:
        - High contrast black and white with screen tone shading
        - Large expressive eyes, detailed hair
        - Dynamic action lines and speed effects
        - Emotional background patterns (flowers for romance, dark aura for anger)
        - Clean precise linework
        """
    ),
    
    ComicTradition.ANIME: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="clean",
        shading_style="cel",
        color_mode="color",
        color_temperature="neutral",
        saturation="vibrant",
        contrast="normal",
        eye_style="large",
        body_proportions="realistic",
        expression_style="exaggerated",
        panel_shape="rectangular",
        gutter_style="white",
        border_style="thin",
        background_detail="moderate",
        speed_lines=True,
        emotional_backgrounds=True,
        motion_effects="lines",
        typical_aspect_ratio=(16, 9),
        style_keywords=[
            "anime style", "vibrant colors", "cel shading",
            "large expressive eyes", "detailed hair", "dynamic poses",
            "light effects", "soft gradients", "Japanese anime aesthetic"
        ],
        system_prompt_additions="""
        Use colorful anime visual style:
        - Vibrant color palette with cel shading
        - Large detailed eyes with light reflections
        - Detailed flowing hair with highlights
        - Soft lighting and gradient backgrounds
        - Clean smooth linework
        """
    ),
    
    ComicTradition.AMERICAN_SUPERHERO: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="bold",
        shading_style="gradient",
        color_mode="color",
        color_temperature="neutral",
        saturation="vibrant",
        contrast="high",
        eye_style="realistic",
        body_proportions="heroic",
        expression_style="dramatic",
        panel_shape="dynamic",
        gutter_style="white",
        border_style="thick",
        background_detail="detailed",
        speed_lines=False,
        emotional_backgrounds=False,
        motion_effects="blur",
        typical_aspect_ratio=(1.0, 1.0),
        style_keywords=[
            "American comic book style", "bold outlines", "heroic proportions",
            "dramatic lighting", "vibrant saturated colors", "heavy shadows",
            "dynamic action poses", "muscular figures", "DC Marvel style"
        ],
        system_prompt_additions="""
        Use classic American superhero comic style:
        - Bold heavy outlines with confident linework
        - Heroic muscular proportions
        - Dramatic chiaroscuro lighting
        - Vibrant primary colors (reds, blues, yellows)
        - Dynamic foreshortening and perspective
        - Detailed urban backgrounds
        """
    ),
    
    ComicTradition.FRANCO_BELGIAN: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="clean",
        shading_style="flat",
        color_mode="color",
        color_temperature="warm",
        saturation="normal",
        contrast="normal",
        eye_style="stylized",
        body_proportions="realistic",
        expression_style="subtle",
        panel_shape="rectangular",
        gutter_style="white",
        border_style="thin",
        background_detail="detailed",
        speed_lines=False,
        emotional_backgrounds=False,
        motion_effects="none",
        typical_aspect_ratio=(1.0, 1.0),
        style_keywords=[
            "Tintin style", "ligne claire", "clear line art",
            "flat colors", "detailed backgrounds", "European comic",
            "Belgian comic style", "Hergé inspired", "clean outlines"
        ],
        system_prompt_additions="""
        Use Franco-Belgian ligne claire style:
        - Clean uniform line weight throughout
        - Flat colors without gradients
        - Highly detailed realistic backgrounds
        - Semi-realistic character proportions
        - Subtle expressions, clear storytelling
        - Warm nostalgic color palette
        """
    ),
    
    ComicTradition.MANHWA: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="clean",
        shading_style="gradient",
        color_mode="color",
        color_temperature="cool",
        saturation="normal",
        contrast="normal",
        eye_style="large",
        body_proportions="elongated",
        expression_style="dramatic",
        panel_shape="rectangular",
        gutter_style="white",
        border_style="thin",
        background_detail="moderate",
        speed_lines=True,
        emotional_backgrounds=True,
        motion_effects="blur",
        typical_aspect_ratio=(0.6, 1.0),
        style_keywords=[
            "Korean manhwa style", "webtoon aesthetic", "soft shading",
            "detailed character designs", "fantasy aesthetic", "romantic style",
            "vertical scroll format", "pastel colors"
        ],
        system_prompt_additions="""
        Use Korean manhwa/webtoon style:
        - Soft gradient shading
        - Elongated elegant character proportions
        - Detailed fashion and clothing
        - Soft pastel color palette with occasional vivid accents
        - Romantic fantasy aesthetic
        - Clean precise linework
        """
    ),
    
    ComicTradition.MANHUA: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="variable",
        shading_style="gradient",
        color_mode="color",
        color_temperature="warm",
        saturation="vibrant",
        contrast="high",
        eye_style="stylized",
        body_proportions="realistic",
        expression_style="dramatic",
        panel_shape="dynamic",
        gutter_style="none",
        border_style="none",
        background_detail="detailed",
        speed_lines=True,
        emotional_backgrounds=True,
        motion_effects="afterimages",
        typical_aspect_ratio=(1.0, 1.0),
        style_keywords=[
            "Chinese manhua style", "wuxia aesthetic", "martial arts",
            "flowing robes", "traditional Chinese elements", "dynamic action",
            "ink wash influence", "fantasy adventure"
        ],
        system_prompt_additions="""
        Use Chinese manhua style:
        - Dynamic flowing compositions
        - Traditional Chinese clothing and architecture
        - Martial arts action with flowing movement
        - Rich detailed backgrounds with Chinese aesthetics
        - Variable brush-like line weight
        - Warm color palette with gold accents
        """
    ),
    
    ComicTradition.WEBCOMIC: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="sketchy",
        shading_style="flat",
        color_mode="color",
        color_temperature="neutral",
        saturation="normal",
        contrast="normal",
        eye_style="minimal",
        body_proportions="stylized",
        expression_style="exaggerated",
        panel_shape="rectangular",
        gutter_style="white",
        border_style="thin",
        background_detail="minimal",
        speed_lines=False,
        emotional_backgrounds=False,
        motion_effects="none",
        typical_aspect_ratio=(1.0, 1.0),
        style_keywords=[
            "webcomic style", "simple character designs", "expressive faces",
            "minimal backgrounds", "casual art style", "relatable humor",
            "internet comic aesthetic"
        ],
        system_prompt_additions="""
        Use casual webcomic style:
        - Simple stylized character designs
        - Minimal but expressive features
        - Simple or abstract backgrounds
        - Focus on facial expressions
        - Casual approachable aesthetic
        """
    ),
    
    ComicTradition.WEBTOON: TraditionParameters(
        reading_direction="vertical",
        panel_flow="vertical",
        line_style="clean",
        shading_style="gradient",
        color_mode="color",
        color_temperature="cool",
        saturation="normal",
        contrast="normal",
        eye_style="large",
        body_proportions="realistic",
        expression_style="dramatic",
        panel_shape="rectangular",
        gutter_style="colored",
        border_style="none",
        background_detail="moderate",
        speed_lines=True,
        emotional_backgrounds=True,
        motion_effects="blur",
        typical_aspect_ratio=(0.5, 1.0),
        style_keywords=[
            "webtoon style", "vertical scroll format", "soft shading",
            "pastel colors", "clean lineart", "emotional storytelling",
            "digital art aesthetic"
        ],
        system_prompt_additions="""
        Use modern webtoon style:
        - Vertical scroll-optimized composition
        - Soft gradient shading
        - Clean digital linework
        - Pastel and soft color palette
        - Focus on emotional expression
        - Seamless panel transitions
        """
    ),
    
    ComicTradition.HORROR: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="variable",
        shading_style="hatching",
        color_mode="limited",
        color_temperature="cool",
        saturation="desaturated",
        contrast="high",
        eye_style="realistic",
        body_proportions="realistic",
        expression_style="dramatic",
        panel_shape="irregular",
        gutter_style="black",
        border_style="thick",
        background_detail="detailed",
        speed_lines=False,
        emotional_backgrounds=True,
        motion_effects="blur",
        typical_aspect_ratio=(1.0, 1.2),
        style_keywords=[
            "horror comic style", "dark atmospheric", "heavy shadows",
            "creepy imagery", "detailed grotesque", "EC Comics style",
            "unsettling compositions", "dramatic lighting"
        ],
        system_prompt_additions="""
        Use horror comic visual language:
        - Heavy dramatic shadows
        - Unsettling compositions and angles
        - Desaturated or limited color palette
        - Detailed cross-hatching for texture
        - Emphasis on atmosphere and dread
        - Irregular threatening panel shapes
        """
    ),
    
    ComicTradition.NOIR: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="bold",
        shading_style="hatching",
        color_mode="black-white",
        color_temperature="cool",
        saturation="desaturated",
        contrast="high",
        eye_style="realistic",
        body_proportions="realistic",
        expression_style="subtle",
        panel_shape="rectangular",
        gutter_style="black",
        border_style="thick",
        background_detail="moderate",
        speed_lines=False,
        emotional_backgrounds=True,
        motion_effects="none",
        typical_aspect_ratio=(1.0, 1.0),
        style_keywords=[
            "noir comic style", "high contrast black and white",
            "heavy shadows", "film noir aesthetic", "dramatic lighting",
            "Sin City style", "stark compositions"
        ],
        system_prompt_additions="""
        Use noir comic visual language:
        - High contrast black and white
        - Dramatic film noir lighting
        - Heavy use of silhouettes and shadows
        - Minimal mid-tones
        - Moody atmospheric compositions
        - Urban night settings
        """
    ),
    
    ComicTradition.NEWSPAPER: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="clean",
        shading_style="flat",
        color_mode="color",
        color_temperature="warm",
        saturation="normal",
        contrast="normal",
        eye_style="stylized",
        body_proportions="stylized",
        expression_style="exaggerated",
        panel_shape="rectangular",
        gutter_style="white",
        border_style="thin",
        background_detail="minimal",
        speed_lines=False,
        emotional_backgrounds=False,
        motion_effects="none",
        typical_aspect_ratio=(3.0, 1.0),
        style_keywords=[
            "newspaper comic strip", "classic comic strip style",
            "simple clear art", "Sunday funnies aesthetic",
            "Peanuts style", "cartoon style"
        ],
        system_prompt_additions="""
        Use classic newspaper comic strip style:
        - Simple clean character designs
        - Minimal backgrounds
        - Clear readable expressions
        - Warm friendly color palette
        - Consistent simple style throughout
        """
    ),
    
    ComicTradition.GOLDEN_AGE: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="bold",
        shading_style="flat",
        color_mode="limited",
        color_temperature="warm",
        saturation="vibrant",
        contrast="normal",
        eye_style="realistic",
        body_proportions="heroic",
        expression_style="dramatic",
        panel_shape="rectangular",
        gutter_style="white",
        border_style="thick",
        background_detail="moderate",
        speed_lines=False,
        emotional_backgrounds=False,
        motion_effects="none",
        typical_aspect_ratio=(1.0, 1.0),
        style_keywords=[
            "golden age comic style", "1940s comic aesthetic",
            "vintage comic book", "retro superhero style",
            "limited color palette", "classic four-color printing"
        ],
        system_prompt_additions="""
        Use Golden Age comic style:
        - Bold simple linework
        - Limited retro color palette (CMYK printing look)
        - Heroic poses and proportions
        - Vintage 1940s-50s aesthetic
        - Clear narrative compositions
        """
    ),
    
    ComicTradition.AMERICAN_INDIE: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="variable",
        shading_style="hatching",
        color_mode="grayscale",
        color_temperature="neutral",
        saturation="desaturated",
        contrast="normal",
        eye_style="stylized",
        body_proportions="realistic",
        expression_style="subtle",
        panel_shape="rectangular",
        gutter_style="white",
        border_style="thin",
        background_detail="detailed",
        speed_lines=False,
        emotional_backgrounds=False,
        motion_effects="none",
        typical_aspect_ratio=(1.0, 1.0),
        style_keywords=[
            "indie comic style", "alternative comics aesthetic",
            "artistic linework", "literary graphic novel",
            "expressive personal style", "underground comics"
        ],
        system_prompt_additions="""
        Use indie/alternative comic style:
        - Expressive personal artistic style
        - Variable line weight and texture
        - Focus on emotional storytelling
        - Detailed character-driven art
        - Often grayscale or limited palette
        """
    ),
    
    ComicTradition.COMEDY: TraditionParameters(
        reading_direction="ltr",
        panel_flow="western",
        line_style="sketchy",
        shading_style="flat",
        color_mode="color",
        color_temperature="warm",
        saturation="vibrant",
        contrast="normal",
        eye_style="stylized",
        body_proportions="stylized",
        expression_style="exaggerated",
        panel_shape="rectangular",
        gutter_style="white",
        border_style="wavy",
        background_detail="minimal",
        speed_lines=True,
        emotional_backgrounds=True,
        motion_effects="lines",
        typical_aspect_ratio=(1.0, 1.0),
        style_keywords=[
            "comedy comic style", "humorous illustration",
            "exaggerated expressions", "cartoon style", "funny faces",
            "playful compositions", "bright cheerful colors"
        ],
        system_prompt_additions="""
        Use comedy comic visual language:
        - Exaggerated facial expressions
        - Playful character proportions
        - Bright cheerful color palette
        - Dynamic comedic timing through composition
        - Fun cartoony aesthetic
        """
    ),
}


class StyleAdapter:
    """
    Main class for adapting comic generation to different traditions.
    """
    
    def __init__(self, tradition: ComicTradition = ComicTradition.ANIME):
        """
        Initialize with a specific tradition.
        
        Args:
            tradition: Comic tradition to use
        """
        self.tradition = tradition
        self.parameters = TRADITION_CONFIGS.get(
            tradition, 
            TRADITION_CONFIGS[ComicTradition.ANIME]
        )
    
    def set_tradition(self, tradition: ComicTradition):
        """Change the current tradition."""
        self.tradition = tradition
        self.parameters = TRADITION_CONFIGS.get(
            tradition,
            TRADITION_CONFIGS[ComicTradition.ANIME]
        )
    
    def adapt_prompt(self, original_prompt: str) -> str:
        """
        Adapt a generation prompt to include tradition-specific styling.
        
        Args:
            original_prompt: Original panel description
        
        Returns:
            Enhanced prompt with style keywords
        """
        style_additions = ", ".join(self.parameters.style_keywords)
        
        enhanced_prompt = f"{original_prompt}\n\nArt Style: {style_additions}"
        
        if self.parameters.system_prompt_additions:
            enhanced_prompt += f"\n\n{self.parameters.system_prompt_additions}"
        
        return enhanced_prompt
    
    def get_layout_parameters(self) -> Dict[str, Any]:
        """
        Get layout parameters for this tradition.
        
        Returns:
            Dictionary of layout parameters
        """
        return {
            "reading_direction": self.parameters.reading_direction,
            "panel_flow": self.parameters.panel_flow,
            "panel_shape": self.parameters.panel_shape,
            "gutter_style": self.parameters.gutter_style,
            "border_style": self.parameters.border_style,
            "typical_aspect_ratio": self.parameters.typical_aspect_ratio
        }
    
    def get_color_parameters(self) -> Dict[str, Any]:
        """
        Get color parameters for this tradition.
        
        Returns:
            Dictionary of color parameters
        """
        return {
            "color_mode": self.parameters.color_mode,
            "color_temperature": self.parameters.color_temperature,
            "saturation": self.parameters.saturation,
            "contrast": self.parameters.contrast
        }
    
    def get_character_parameters(self) -> Dict[str, Any]:
        """
        Get character design parameters for this tradition.
        
        Returns:
            Dictionary of character parameters
        """
        return {
            "eye_style": self.parameters.eye_style,
            "body_proportions": self.parameters.body_proportions,
            "expression_style": self.parameters.expression_style
        }
    
    def should_flip_for_reading(self) -> bool:
        """Check if panels should be flipped for reading direction."""
        return self.parameters.reading_direction == "rtl"
    
    def get_effect_parameters(self) -> Dict[str, Any]:
        """
        Get visual effect parameters.
        
        Returns:
            Dictionary of effect parameters
        """
        return {
            "speed_lines": self.parameters.speed_lines,
            "emotional_backgrounds": self.parameters.emotional_backgrounds,
            "motion_effects": self.parameters.motion_effects,
            "sound_effects": self.parameters.sound_effects
        }
    
    def get_full_style_description(self) -> str:
        """Get a complete style description for documentation."""
        return f"""
        Comic Tradition: {self.tradition.value}
        
        Visual Style:
        - Line Style: {self.parameters.line_style}
        - Shading: {self.parameters.shading_style}
        - Color Mode: {self.parameters.color_mode}
        
        Character Design:
        - Eye Style: {self.parameters.eye_style}
        - Body Proportions: {self.parameters.body_proportions}
        - Expression Style: {self.parameters.expression_style}
        
        Layout:
        - Reading Direction: {self.parameters.reading_direction}
        - Panel Shape: {self.parameters.panel_shape}
        - Border Style: {self.parameters.border_style}
        
        Effects:
        - Speed Lines: {self.parameters.speed_lines}
        - Motion Effects: {self.parameters.motion_effects}
        - Emotional Backgrounds: {self.parameters.emotional_backgrounds}
        
        Keywords: {', '.join(self.parameters.style_keywords[:5])}
        """


# Convenience functions
def get_tradition_parameters(tradition: ComicTradition) -> TraditionParameters:
    """Get parameters for a tradition."""
    return TRADITION_CONFIGS.get(tradition, TRADITION_CONFIGS[ComicTradition.ANIME])


def adapt_prompt_to_tradition(prompt: str, tradition: ComicTradition) -> str:
    """Quick function to adapt a prompt."""
    adapter = StyleAdapter(tradition)
    return adapter.adapt_prompt(prompt)


def get_available_traditions() -> List[str]:
    """Get list of available tradition names."""
    return [t.value for t in ComicTradition]


# Mapping from simple style names to traditions
SIMPLE_STYLE_MAP = {
    "manga": ComicTradition.MANGA,
    "anime": ComicTradition.ANIME,
    "american": ComicTradition.AMERICAN_SUPERHERO,
    "belgian": ComicTradition.FRANCO_BELGIAN,
    "franco-belgian": ComicTradition.FRANCO_BELGIAN,
    "manhwa": ComicTradition.MANHWA,
    "manhua": ComicTradition.MANHUA,
    "webcomic": ComicTradition.WEBCOMIC,
    "webtoon": ComicTradition.WEBTOON,
    "horror": ComicTradition.HORROR,
    "noir": ComicTradition.NOIR,
    "comedy": ComicTradition.COMEDY,
    "indie": ComicTradition.AMERICAN_INDIE,
    "newspaper": ComicTradition.NEWSPAPER,
    "golden age": ComicTradition.GOLDEN_AGE,
    "vintage": ComicTradition.GOLDEN_AGE
}


def get_tradition_from_style_name(style_name: str) -> ComicTradition:
    """Convert simple style name to ComicTradition enum."""
    return SIMPLE_STYLE_MAP.get(style_name.lower(), ComicTradition.ANIME)
