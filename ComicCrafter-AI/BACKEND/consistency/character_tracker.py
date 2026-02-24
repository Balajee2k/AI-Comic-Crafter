"""
Character Consistency Tracker
=============================
Novel approach for maintaining visual identity across AI-generated comic panels.

Key Innovations:
1. NLP-based character extraction from story text
2. Visual attribute encoding for each character
3. Color palette persistence across panels
4. Reference image integration for style consistency
5. Cross-panel similarity metrics using CLIP embeddings

This module addresses a critical gap in current AI comic generation:
the inability to maintain consistent character appearances across panels.
"""

import os
import re
import json
import base64
from io import BytesIO
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np
from collections import Counter

# Try to import optional dependencies
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available. Install transformers and torch for visual similarity features.")


@dataclass
class CharacterProfile:
    """
    Comprehensive character profile for maintaining visual consistency.
    
    Attributes:
        name: Character's name extracted from story
        description: Physical description compiled from narrative
        visual_attributes: Specific visual features (hair color, clothing, etc.)
        color_palette: Dominant colors associated with this character
        reference_image_path: Optional user-provided reference image
        appearance_history: Track how character was described in each panel
        embedding: CLIP embedding for visual similarity (if available)
    """
    name: str
    description: str = ""
    visual_attributes: Dict[str, str] = field(default_factory=dict)
    color_palette: List[str] = field(default_factory=list)
    reference_image_path: Optional[str] = None
    appearance_history: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    def to_prompt_injection(self) -> str:
        """Generate a prompt injection string for consistent image generation."""
        prompt_parts = []
        
        if self.description:
            prompt_parts.append(f"{self.name}: {self.description}")
        
        if self.visual_attributes:
            attrs = ", ".join([f"{k}: {v}" for k, v in self.visual_attributes.items()])
            prompt_parts.append(f"Visual features: {attrs}")
        
        if self.color_palette:
            prompt_parts.append(f"Color scheme: {', '.join(self.color_palette[:5])}")
        
        return " | ".join(prompt_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert numpy array to list if needed
        if self.embedding is not None and hasattr(self.embedding, 'tolist'):
            data['embedding'] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterProfile':
        """Create from dictionary."""
        return cls(**data)


def extract_color_palette(image: Image.Image, n_colors: int = 5) -> List[str]:
    """
    Extract dominant colors from an image.
    
    Uses k-means clustering on pixel values to find dominant colors.
    Returns colors as hex strings for consistent representation.
    
    Args:
        image: PIL Image to analyze
        n_colors: Number of dominant colors to extract
    
    Returns:
        List of hex color strings (e.g., ['#FF5733', '#C70039'])
    """
    # Resize for faster processing
    img_small = image.copy()
    img_small.thumbnail((150, 150))
    
    # Convert to RGB if necessary
    if img_small.mode != 'RGB':
        img_small = img_small.convert('RGB')
    
    # Get pixels
    pixels = np.array(img_small).reshape(-1, 3)
    
    # Simple k-means implementation
    from collections import Counter
    
    # Quantize colors to reduce complexity
    quantized = (pixels // 32) * 32
    
    # Find most common colors
    color_counts = Counter(map(tuple, quantized))
    common_colors = color_counts.most_common(n_colors)
    
    # Convert to hex
    hex_colors = []
    for color, _ in common_colors:
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
        hex_colors.append(hex_color)
    
    return hex_colors


def compute_visual_similarity(image1: Image.Image, image2: Image.Image, 
                               model=None, processor=None) -> float:
    """
    Compute visual similarity between two images using CLIP embeddings.
    
    This is crucial for measuring character consistency across panels.
    Higher scores indicate more visually similar images.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        model: Pre-loaded CLIP model (optional, will load if None)
        processor: Pre-loaded CLIP processor (optional)
    
    Returns:
        Similarity score between 0 and 1
    """
    if not CLIP_AVAILABLE:
        # Fallback: Use color histogram similarity
        return _color_histogram_similarity(image1, image2)
    
    try:
        if model is None or processor is None:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Process images
        inputs1 = processor(images=image1, return_tensors="pt")
        inputs2 = processor(images=image2, return_tensors="pt")
        
        # Get embeddings
        with torch.no_grad():
            emb1 = model.get_image_features(**inputs1)
            emb2 = model.get_image_features(**inputs2)
        
        # Normalize
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = (emb1 @ emb2.T).item()
        
        # Convert from [-1, 1] to [0, 1] range
        return (similarity + 1) / 2
        
    except Exception as e:
        print(f"⚠️ CLIP similarity failed: {e}. Using fallback.")
        return _color_histogram_similarity(image1, image2)


def _color_histogram_similarity(image1: Image.Image, image2: Image.Image) -> float:
    """Fallback similarity using color histograms."""
    # Resize for consistency
    size = (100, 100)
    img1 = image1.copy().resize(size).convert('RGB')
    img2 = image2.copy().resize(size).convert('RGB')
    
    # Compute histograms
    hist1 = img1.histogram()
    hist2 = img2.histogram()
    
    # Compute correlation
    h1 = np.array(hist1, dtype=np.float32)
    h2 = np.array(hist2, dtype=np.float32)
    
    # Normalize
    h1 = h1 / (h1.sum() + 1e-10)
    h2 = h2 / (h2.sum() + 1e-10)
    
    # Compute histogram intersection
    similarity = np.minimum(h1, h2).sum()
    
    return float(similarity)


class CharacterConsistencyEngine:
    """
    Main engine for maintaining character consistency across comic panels.
    
    This is a novel contribution that addresses the fundamental challenge
    of AI-generated sequential art: maintaining visual identity.
    
    Key Features:
    1. Automatic character extraction from story text
    2. Visual attribute tracking and enforcement
    3. Reference image support for user-defined characters
    4. Cross-panel consistency scoring
    5. Prompt injection for consistent generation
    
    Research Applications:
    - Quantitative measurement of visual consistency
    - Ablation studies on consistency factors
    - User studies on perceived character identity
    """
    
    def __init__(self):
        """Initialize the consistency engine."""
        self.character_registry: Dict[str, CharacterProfile] = {}
        self.scene_context: List[Dict[str, Any]] = []
        self.panel_history: List[Dict[str, Any]] = []
        self.clip_model = None
        self.clip_processor = None
        self._load_clip_model()
    
    def _load_clip_model(self):
        """Load CLIP model for visual similarity (if available)."""
        if CLIP_AVAILABLE:
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print("✅ CLIP model loaded for visual consistency analysis.")
            except Exception as e:
                print(f"⚠️ Could not load CLIP model: {e}")
    
    def extract_characters_from_story(self, story_text: str) -> List[str]:
        """
        Extract character names from story text using NLP heuristics.
        
        Uses pattern matching and capitalization rules to identify
        potential character names. More sophisticated NER can be added.
        
        Args:
            story_text: Full story or panel description text
        
        Returns:
            List of identified character names
        """
        characters = set()
        
        # Pattern 1: Quoted dialogue with speaker
        dialogue_pattern = r'([A-Z][a-z]+):\s*["\']'
        matches = re.findall(dialogue_pattern, story_text)
        characters.update(matches)
        
        # Pattern 2: Common character introductions
        intro_patterns = [
            r'(?:named|called|known as)\s+([A-Z][a-z]+)',
            r'([A-Z][a-z]+)\s+(?:was|is|said|asked|replied|thought)',
            r'(?:the|a)\s+(?:young|old|brave|wise)?\s*([A-Z][a-z]+)',
        ]
        
        for pattern in intro_patterns:
            matches = re.findall(pattern, story_text)
            characters.update(matches)
        
        # Pattern 3: Capitalized words that appear multiple times
        words = re.findall(r'\b([A-Z][a-z]{2,})\b', story_text)
        word_counts = Counter(words)
        
        # Words appearing 2+ times might be character names
        for word, count in word_counts.items():
            if count >= 2 and word not in ['The', 'And', 'But', 'When', 'Then', 'This', 'That']:
                characters.add(word)
        
        # Filter out common non-name words
        excluded = {'The', 'And', 'But', 'When', 'Then', 'This', 'That', 'There', 
                   'Here', 'Where', 'What', 'Who', 'How', 'Why', 'Panel', 'Scene',
                   'Chapter', 'Part', 'Introduction', 'Climax', 'Moral'}
        characters = [c for c in characters if c not in excluded]
        
        return list(characters)
    
    def extract_character_description(self, story_text: str, character_name: str) -> str:
        """
        Extract physical description for a specific character.
        
        Looks for descriptive phrases near character name mentions.
        
        Args:
            story_text: Full story text
            character_name: Name of character to find description for
        
        Returns:
            Compiled description string
        """
        descriptions = []
        
        # Find sentences containing the character name
        sentences = re.split(r'[.!?]', story_text)
        
        for sentence in sentences:
            if character_name.lower() in sentence.lower():
                # Extract descriptive adjectives and features
                desc_patterns = [
                    r'(\w+\s+hair)',
                    r'(\w+\s+eyes)',
                    r'(wearing\s+[\w\s]+)',
                    r'(tall|short|young|old|beautiful|handsome)',
                    r'(\w+\s+skin)',
                    r'(dressed in\s+[\w\s]+)',
                ]
                
                for pattern in desc_patterns:
                    matches = re.findall(pattern, sentence, re.IGNORECASE)
                    descriptions.extend(matches)
        
        return ", ".join(set(descriptions)) if descriptions else ""
    
    def register_character(self, name: str, description: str = "", 
                          visual_attributes: Dict[str, str] = None,
                          reference_image: Image.Image = None,
                          reference_image_path: str = None) -> CharacterProfile:
        """
        Register a character in the consistency engine.
        
        Args:
            name: Character name
            description: Physical description
            visual_attributes: Dict of visual features (hair_color, clothing, etc.)
            reference_image: Optional PIL Image for color extraction
            reference_image_path: Path to reference image file
        
        Returns:
            Created CharacterProfile
        """
        profile = CharacterProfile(
            name=name,
            description=description,
            visual_attributes=visual_attributes or {},
            reference_image_path=reference_image_path
        )
        
        # Extract color palette from reference image
        if reference_image:
            profile.color_palette = extract_color_palette(reference_image)
            
            # Compute CLIP embedding if available
            if self.clip_model and self.clip_processor:
                try:
                    inputs = self.clip_processor(images=reference_image, return_tensors="pt")
                    with torch.no_grad():
                        embedding = self.clip_model.get_image_features(**inputs)
                    profile.embedding = embedding.squeeze().tolist()
                except Exception as e:
                    print(f"⚠️ Could not compute embedding: {e}")
        
        self.character_registry[name] = profile
        return profile
    
    def register_characters_from_story(self, story_text: str) -> List[CharacterProfile]:
        """
        Automatically extract and register all characters from story.
        
        Args:
            story_text: Full story text
        
        Returns:
            List of created CharacterProfiles
        """
        character_names = self.extract_characters_from_story(story_text)
        profiles = []
        
        for name in character_names:
            if name not in self.character_registry:
                description = self.extract_character_description(story_text, name)
                profile = self.register_character(name, description)
                profiles.append(profile)
        
        return profiles
    
    def update_character_from_panel(self, character_name: str, 
                                    panel_description: str,
                                    generated_image: Image.Image = None):
        """
        Update character profile based on generated panel.
        
        This allows the system to learn and maintain consistency
        as the comic is generated panel by panel.
        
        Args:
            character_name: Name of character
            panel_description: Description used for this panel
            generated_image: The generated panel image
        """
        if character_name not in self.character_registry:
            self.register_character(character_name)
        
        profile = self.character_registry[character_name]
        profile.appearance_history.append(panel_description)
        
        # Update color palette from generated image
        if generated_image:
            new_colors = extract_color_palette(generated_image)
            if not profile.color_palette:
                profile.color_palette = new_colors
            else:
                # Merge color palettes (keep most consistent colors)
                all_colors = profile.color_palette + new_colors
                color_counts = Counter(all_colors)
                profile.color_palette = [c for c, _ in color_counts.most_common(5)]
    
    def generate_consistent_prompt(self, panel_description: str, 
                                   panel_number: int = 1) -> str:
        """
        Generate an enhanced prompt that includes character consistency tokens.
        
        This is the key function that injects consistency information
        into the image generation prompt.
        
        Args:
            panel_description: Original panel description
            panel_number: Current panel number (for context)
        
        Returns:
            Enhanced prompt with consistency tokens
        """
        # Extract characters mentioned in this panel
        mentioned_characters = []
        for name in self.character_registry.keys():
            if name.lower() in panel_description.lower():
                mentioned_characters.append(name)
        
        if not mentioned_characters:
            return panel_description
        
        # Build consistency injection
        consistency_parts = []
        
        for char_name in mentioned_characters:
            profile = self.character_registry[char_name]
            injection = profile.to_prompt_injection()
            if injection:
                consistency_parts.append(injection)
        
        # Add global consistency note
        consistency_note = (
            "\n[CHARACTER CONSISTENCY DIRECTIVE]: "
            "Maintain exact visual appearance for all characters. "
        )
        
        if consistency_parts:
            consistency_note += "Character details: " + " || ".join(consistency_parts)
        
        return panel_description + consistency_note
    
    def compute_panel_consistency_score(self, panel_images: List[Image.Image]) -> Dict[str, float]:
        """
        Compute consistency scores across all panels.
        
        This metric quantifies how well character appearances are
        maintained throughout the comic.
        
        Args:
            panel_images: List of generated panel images
        
        Returns:
            Dictionary with various consistency metrics
        """
        if len(panel_images) < 2:
            return {"overall": 1.0, "pairwise_avg": 1.0}
        
        # Compute pairwise similarities
        pairwise_scores = []
        
        for i in range(len(panel_images)):
            for j in range(i + 1, len(panel_images)):
                score = compute_visual_similarity(
                    panel_images[i], 
                    panel_images[j],
                    self.clip_model,
                    self.clip_processor
                )
                pairwise_scores.append(score)
        
        # Compute sequential consistency (adjacent panels)
        sequential_scores = []
        for i in range(len(panel_images) - 1):
            score = compute_visual_similarity(
                panel_images[i],
                panel_images[i + 1],
                self.clip_model,
                self.clip_processor
            )
            sequential_scores.append(score)
        
        return {
            "overall": np.mean(pairwise_scores) if pairwise_scores else 1.0,
            "pairwise_avg": np.mean(pairwise_scores) if pairwise_scores else 1.0,
            "sequential_avg": np.mean(sequential_scores) if sequential_scores else 1.0,
            "min_similarity": min(pairwise_scores) if pairwise_scores else 1.0,
            "max_similarity": max(pairwise_scores) if pairwise_scores else 1.0,
            "std_deviation": np.std(pairwise_scores) if pairwise_scores else 0.0
        }
    
    def get_character_summary(self) -> Dict[str, Dict]:
        """Get summary of all registered characters."""
        return {
            name: profile.to_dict() 
            for name, profile in self.character_registry.items()
        }
    
    def save_state(self, filepath: str):
        """Save consistency engine state to file."""
        state = {
            'characters': {
                name: profile.to_dict() 
                for name, profile in self.character_registry.items()
            },
            'scene_context': self.scene_context,
            'panel_history': self.panel_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load consistency engine state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.character_registry = {
            name: CharacterProfile.from_dict(data)
            for name, data in state.get('characters', {}).items()
        }
        self.scene_context = state.get('scene_context', [])
        self.panel_history = state.get('panel_history', [])
    
    def clear(self):
        """Clear all stored data."""
        self.character_registry.clear()
        self.scene_context.clear()
        self.panel_history.clear()


# Convenience function for quick usage
def create_consistency_engine() -> CharacterConsistencyEngine:
    """Factory function to create a new consistency engine."""
    return CharacterConsistencyEngine()
