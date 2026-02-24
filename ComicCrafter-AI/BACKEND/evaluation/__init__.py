"""
Evaluation Metrics Module for AI Comic Crafter
===============================================
Novel quantitative framework for assessing AI-generated comics.

Research Contribution:
- First comprehensive evaluation framework for AI comics
- Multiple quality dimensions with weighted scoring
- Cross-panel consistency metrics
- Narrative coherence scoring
- Comic grammar compliance checking

Author: AI Comic Crafter Research Team
"""

from .comic_metrics import (
    ComicQualityMetrics,
    ComicEvaluator,
    NarrativeCoherenceScorer,
    VisualQualityScorer,
    ComicGrammarChecker,
    generate_evaluation_report
)

__all__ = [
    'ComicQualityMetrics',
    'ComicEvaluator',
    'NarrativeCoherenceScorer', 
    'VisualQualityScorer',
    'ComicGrammarChecker',
    'generate_evaluation_report'
]
