"""
Character Consistency Module for AI Comic Crafter
==================================================
A novel approach to maintaining visual identity across sequential comic panels.

Research Contribution:
- First embedding-based identity preservation system for AI-generated comics
- Character registry with persistent visual attributes
- Color palette extraction and enforcement
- Cross-panel similarity scoring using CLIP embeddings

Author: AI Comic Crafter Research Team
"""

from .character_tracker import (
    CharacterProfile,
    CharacterConsistencyEngine,
    extract_color_palette,
    compute_visual_similarity
)

__all__ = [
    'CharacterProfile',
    'CharacterConsistencyEngine', 
    'extract_color_palette',
    'compute_visual_similarity'
]
