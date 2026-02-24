"""
Comic Style Adapter Module
===========================
Adapts comic generation to different cultural traditions.

Research Contribution:
- Multi-cultural comic style parameterization
- Tradition-specific visual grammar encoding
- Reading direction and layout adaptation
- Style-specific prompt engineering

Author: AI Comic Crafter Research Team
"""

from .tradition_adapter import (
    ComicTradition,
    TraditionParameters,
    StyleAdapter,
    get_tradition_parameters,
    adapt_prompt_to_tradition
)

__all__ = [
    'ComicTradition',
    'TraditionParameters',
    'StyleAdapter',
    'get_tradition_parameters',
    'adapt_prompt_to_tradition'
]
