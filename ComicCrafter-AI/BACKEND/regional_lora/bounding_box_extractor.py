"""
Bounding Box Extractor
======================
Extracts character bounding boxes from panel descriptions using Gemini Vision.

Research Innovation:
Uses LLM vision capabilities to understand semantic regions in images,
rather than traditional object detection which fails on stylized comic art.

The extractor can work in two modes:
1. Pre-generation: Predict bounding boxes from text description
2. Post-generation: Analyze generated image to verify/refine boxes

This enables a two-pass refinement:
- Pass 1: Generate with predicted regions
- Pass 2: Refine using actual image analysis (if needed)
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from PIL import Image
import io
import base64

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class CharacterPosition(Enum):
    """Standard character positions in comic panels."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


@dataclass
class CharacterBoundingBox:
    """
    Bounding box for a character in a panel.
    
    Attributes:
        character_name: Name of the character
        x1, y1: Top-left corner (normalized 0-1)
        x2, y2: Bottom-right corner (normalized 0-1)
        confidence: Confidence score of detection (0-1)
        pose: Detected pose (standing, sitting, action, etc.)
        facing: Direction character is facing
        z_order: Depth order (higher = closer to viewer)
        semantic_region: Which part of panel (left, center, right)
    """
    character_name: str
    x1: float  # Normalized coordinates (0-1)
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    pose: str = "standing"
    facing: str = "forward"
    z_order: int = 0
    semantic_region: CharacterPosition = CharacterPosition.CENTER
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_pixel_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coords to pixel coordinates."""
        return (
            int(self.x1 * img_width),
            int(self.y1 * img_height),
            int(self.x2 * img_width),
            int(self.y2 * img_height)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "character_name": self.character_name,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": self.confidence,
            "pose": self.pose,
            "facing": self.facing,
            "z_order": self.z_order,
            "semantic_region": self.semantic_region.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterBoundingBox':
        bbox = data.get("bbox", [0, 0, 1, 1])
        return cls(
            character_name=data.get("character_name", "unknown"),
            x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
            confidence=data.get("confidence", 1.0),
            pose=data.get("pose", "standing"),
            facing=data.get("facing", "forward"),
            z_order=data.get("z_order", 0),
            semantic_region=CharacterPosition(data.get("semantic_region", "center"))
        )


@dataclass
class PanelLayoutAnalysis:
    """
    Complete layout analysis for a panel.
    
    Contains all character bounding boxes plus panel-level metadata.
    """
    panel_number: int
    character_boxes: List[CharacterBoundingBox]
    background_description: str = ""
    composition_type: str = "standard"  # close-up, wide-shot, over-shoulder, etc.
    focal_point: Tuple[float, float] = (0.5, 0.5)
    has_overlap: bool = False
    overlap_regions: List[Tuple[str, str]] = field(default_factory=list)
    
    def get_box_for_character(self, name: str) -> Optional[CharacterBoundingBox]:
        """Get bounding box for a specific character."""
        for box in self.character_boxes:
            if box.character_name.lower() == name.lower():
                return box
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel_number": self.panel_number,
            "character_boxes": [b.to_dict() for b in self.character_boxes],
            "background_description": self.background_description,
            "composition_type": self.composition_type,
            "focal_point": self.focal_point,
            "has_overlap": self.has_overlap,
            "overlap_regions": self.overlap_regions
        }


class BoundingBoxExtractor:
    """
    Extracts character bounding boxes using Gemini Vision.
    
    Two extraction modes:
    1. From Description (pre-generation): Predict where characters should be
    2. From Image (post-generation): Analyze actual image for character regions
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the extractor.
        
        Args:
            api_key: Google API key (uses env var if not provided)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        
        genai.configure(api_key=self.api_key)
        self.vision_model = genai.GenerativeModel("gemini-2.0-flash")
        self.text_model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Cache for character profiles
        self.character_profiles: Dict[str, Dict] = {}
    
    def register_character(self, name: str, description: str, 
                          typical_position: CharacterPosition = CharacterPosition.CENTER):
        """
        Register a character for improved detection.
        
        Args:
            name: Character name
            description: Physical description
            typical_position: Where this character usually appears
        """
        self.character_profiles[name.lower()] = {
            "name": name,
            "description": description,
            "typical_position": typical_position
        }
    
    def extract_from_description(self, 
                                  panel_description: str,
                                  characters: List[str],
                                  panel_number: int = 1,
                                  aspect_ratio: float = 1.0) -> PanelLayoutAnalysis:
        """
        Predict character bounding boxes from panel description.
        
        This is used BEFORE image generation to plan regional LoRA application.
        
        Args:
            panel_description: Text description of the panel
            characters: List of character names expected in panel
            panel_number: Panel number for reference
            aspect_ratio: Width/height ratio (1.0 = square)
        
        Returns:
            PanelLayoutAnalysis with predicted bounding boxes
        """
        # Build character context
        char_context = ""
        for char in characters:
            if char.lower() in self.character_profiles:
                profile = self.character_profiles[char.lower()]
                char_context += f"- {profile['name']}: {profile['description']}\n"
            else:
                char_context += f"- {char}: (no profile registered)\n"
        
        prompt = f"""
You are an expert comic book layout artist. Analyze this panel description and determine 
where each character should be positioned.

**Panel Description:**
{panel_description}

**Characters in this panel:**
{char_context}

**Panel Aspect Ratio:** {aspect_ratio:.2f} (width/height)

For each character, provide:
1. Bounding box coordinates (normalized 0-1): x1, y1 (top-left), x2, y2 (bottom-right)
2. Pose (standing, sitting, running, fighting, close-up, etc.)
3. Facing direction (left, right, forward, backward)
4. Z-order (depth - higher number = closer to viewer)
5. Semantic region (left, center, right, foreground, background)

Also provide:
- Background description (brief)
- Composition type (close-up, medium-shot, wide-shot, over-shoulder, two-shot, group)
- Focal point (x, y normalized coordinates where viewer's eye should focus)
- Whether any characters overlap

**IMPORTANT RULES:**
- Characters should NOT overlap unless the narrative requires it
- Leave space for action and backgrounds
- Close-ups should have bbox filling most of the frame
- Group shots should distribute characters evenly
- Action scenes may have more dynamic positioning

Respond in this exact JSON format:
```json
{{
    "characters": [
        {{
            "character_name": "Name",
            "bbox": [x1, y1, x2, y2],
            "pose": "standing",
            "facing": "forward",
            "z_order": 0,
            "semantic_region": "center"
        }}
    ],
    "background_description": "Brief description",
    "composition_type": "medium-shot",
    "focal_point": [0.5, 0.5],
    "has_overlap": false,
    "overlap_regions": []
}}
```
"""
        
        response = self.text_model.generate_content(prompt)
        
        if not response or not response.text:
            return self._fallback_layout(characters, panel_number)
        
        return self._parse_layout_response(response.text, panel_number)
    
    def extract_from_image(self, 
                           image: Image.Image,
                           characters: List[str],
                           panel_number: int = 1) -> PanelLayoutAnalysis:
        """
        Extract character bounding boxes from an actual image.
        
        Uses Gemini Vision to analyze the generated image and identify
        where each character actually appears.
        
        Args:
            image: PIL Image to analyze
            characters: List of character names to find
            panel_number: Panel number for reference
        
        Returns:
            PanelLayoutAnalysis with detected bounding boxes
        """
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_data = base64.b64encode(buffered.getvalue()).decode()
        
        # Build character context
        char_descriptions = []
        for char in characters:
            if char.lower() in self.character_profiles:
                profile = self.character_profiles[char.lower()]
                char_descriptions.append(f"- {profile['name']}: {profile['description']}")
            else:
                char_descriptions.append(f"- {char}")
        
        prompt = f"""
Analyze this comic panel image and locate each character.

**Characters to find:**
{chr(10).join(char_descriptions)}

For each character you can identify in the image, provide:
1. Bounding box (normalized 0-1): x1, y1 (top-left), x2, y2 (bottom-right)
2. Pose
3. Facing direction
4. Confidence score (0-1)

Respond in JSON format:
```json
{{
    "characters": [
        {{
            "character_name": "Name",
            "bbox": [x1, y1, x2, y2],
            "pose": "standing",
            "facing": "forward",
            "confidence": 0.9,
            "z_order": 0,
            "semantic_region": "center"
        }}
    ],
    "background_description": "Description of background",
    "composition_type": "shot type"
}}
```
"""
        
        # Create image part for Gemini
        image_part = {
            "mime_type": "image/png",
            "data": img_data
        }
        
        response = self.vision_model.generate_content([prompt, image_part])
        
        if not response or not response.text:
            return self._fallback_layout(characters, panel_number)
        
        return self._parse_layout_response(response.text, panel_number)
    
    def _parse_layout_response(self, response_text: str, 
                                panel_number: int) -> PanelLayoutAnalysis:
        """Parse Gemini's JSON response into PanelLayoutAnalysis."""
        # Extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return self._fallback_layout([], panel_number)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return self._fallback_layout([], panel_number)
        
        # Parse character boxes
        character_boxes = []
        for char_data in data.get("characters", []):
            bbox = char_data.get("bbox", [0.1, 0.1, 0.9, 0.9])
            
            # Validate and clamp coordinates
            x1 = max(0.0, min(1.0, float(bbox[0])))
            y1 = max(0.0, min(1.0, float(bbox[1])))
            x2 = max(0.0, min(1.0, float(bbox[2])))
            y2 = max(0.0, min(1.0, float(bbox[3])))
            
            # Ensure valid box
            if x2 <= x1:
                x2 = min(1.0, x1 + 0.2)
            if y2 <= y1:
                y2 = min(1.0, y1 + 0.3)
            
            try:
                region = CharacterPosition(char_data.get("semantic_region", "center"))
            except ValueError:
                region = CharacterPosition.CENTER
            
            box = CharacterBoundingBox(
                character_name=char_data.get("character_name", "unknown"),
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(char_data.get("confidence", 1.0)),
                pose=char_data.get("pose", "standing"),
                facing=char_data.get("facing", "forward"),
                z_order=int(char_data.get("z_order", 0)),
                semantic_region=region
            )
            character_boxes.append(box)
        
        # Check for overlaps
        has_overlap = False
        overlap_regions = []
        for i, box1 in enumerate(character_boxes):
            for box2 in character_boxes[i+1:]:
                if self._boxes_overlap(box1, box2):
                    has_overlap = True
                    overlap_regions.append((box1.character_name, box2.character_name))
        
        focal_point = data.get("focal_point", [0.5, 0.5])
        if not isinstance(focal_point, (list, tuple)) or len(focal_point) != 2:
            focal_point = (0.5, 0.5)
        
        return PanelLayoutAnalysis(
            panel_number=panel_number,
            character_boxes=character_boxes,
            background_description=data.get("background_description", ""),
            composition_type=data.get("composition_type", "medium-shot"),
            focal_point=tuple(focal_point),
            has_overlap=has_overlap,
            overlap_regions=overlap_regions
        )
    
    def _boxes_overlap(self, box1: CharacterBoundingBox, 
                        box2: CharacterBoundingBox,
                        threshold: float = 0.1) -> bool:
        """Check if two bounding boxes overlap significantly."""
        # Calculate intersection
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection_area = (x2 - x1) * (y2 - y1)
        min_area = min(box1.area, box2.area)
        
        return (intersection_area / min_area) > threshold
    
    def _fallback_layout(self, characters: List[str], 
                          panel_number: int) -> PanelLayoutAnalysis:
        """Generate a simple fallback layout when extraction fails."""
        boxes = []
        n = len(characters)
        
        if n == 0:
            pass
        elif n == 1:
            # Single character - center
            boxes.append(CharacterBoundingBox(
                character_name=characters[0],
                x1=0.2, y1=0.1, x2=0.8, y2=0.95,
                semantic_region=CharacterPosition.CENTER
            ))
        elif n == 2:
            # Two characters - left and right
            boxes.append(CharacterBoundingBox(
                character_name=characters[0],
                x1=0.05, y1=0.1, x2=0.45, y2=0.95,
                semantic_region=CharacterPosition.LEFT
            ))
            boxes.append(CharacterBoundingBox(
                character_name=characters[1],
                x1=0.55, y1=0.1, x2=0.95, y2=0.95,
                semantic_region=CharacterPosition.RIGHT
            ))
        else:
            # Multiple characters - distribute evenly
            width = 0.9 / n
            for i, char in enumerate(characters):
                x1 = 0.05 + i * width
                boxes.append(CharacterBoundingBox(
                    character_name=char,
                    x1=x1, y1=0.1, x2=x1 + width * 0.9, y2=0.95,
                    semantic_region=CharacterPosition.CENTER
                ))
        
        return PanelLayoutAnalysis(
            panel_number=panel_number,
            character_boxes=boxes,
            composition_type="fallback"
        )
    
    def visualize_layout(self, layout: PanelLayoutAnalysis,
                         image: Optional[Image.Image] = None,
                         img_size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Create a visualization of the bounding boxes.
        
        Args:
            layout: The layout to visualize
            image: Optional background image
            img_size: Size for visualization if no image provided
        
        Returns:
            PIL Image with bounding boxes drawn
        """
        from PIL import ImageDraw, ImageFont
        
        if image:
            vis_img = image.copy()
            img_size = image.size
        else:
            vis_img = Image.new('RGB', img_size, color=(240, 240, 240))
        
        draw = ImageDraw.Draw(vis_img)
        
        # Color palette for different characters
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 165, 0), (128, 0, 128), (0, 255, 255)
        ]
        
        for i, box in enumerate(layout.character_boxes):
            color = colors[i % len(colors)]
            
            # Convert to pixel coordinates
            x1, y1, x2, y2 = box.to_pixel_coords(img_size[0], img_size[1])
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{box.character_name} ({box.confidence:.2f})"
            draw.text((x1, y1 - 15), label, fill=color)
        
        return vis_img
