"""
Emotion Analysis Module for AI Comic Crafter
=============================================
Novel approach to emotion-driven panel composition and layout.

Research Contribution:
- First systematic mapping of narrative emotions to comic visual grammar
- Dynamic panel layout based on emotional intensity
- Color temperature and composition adjustment for emotional impact
- Quantitative emotion-visual correlation metrics

Author: AI Comic Crafter Research Team
"""

from .emotion_analyzer import (
    EmotionType,
    PanelStyle,
    EmotionalArc,
    EmotionAnalyzer,
    EmotionToPanelMapper,
    DynamicLayoutEngine
)

__all__ = [
    'EmotionType',
    'PanelStyle', 
    'EmotionalArc',
    'EmotionAnalyzer',
    'EmotionToPanelMapper',
    'DynamicLayoutEngine'
]
