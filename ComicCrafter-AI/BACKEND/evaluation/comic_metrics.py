"""
Comic Quality Evaluation Framework
===================================
A novel quantitative framework for assessing AI-generated comics.

This module provides comprehensive metrics for evaluating:
1. Visual Consistency - Character/setting consistency across panels
2. Narrative Flow - Story coherence and pacing
3. Panel Composition - Adherence to comic visual grammar
4. Emotion Conveyance - How well emotions are visually communicated
5. Technical Quality - Image quality and text readability

Research Applications:
- Benchmark comparisons between generation methods
- Ablation studies on consistency mechanisms
- User study correlation with subjective ratings
- Automatic quality filtering for datasets
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import numpy as np
import re
from collections import Counter
import json


@dataclass
class ComicQualityMetrics:
    """
    Comprehensive quality metrics for an AI-generated comic.
    
    Each metric is scored from 0.0 to 1.0, where:
    - 0.0-0.3: Poor quality
    - 0.3-0.5: Below average
    - 0.5-0.7: Average
    - 0.7-0.85: Good
    - 0.85-1.0: Excellent
    """
    # Visual metrics
    visual_consistency_score: float = 0.0  # Character/setting consistency
    color_harmony_score: float = 0.0       # Color palette coherence
    style_consistency_score: float = 0.0    # Art style uniformity
    
    # Narrative metrics
    narrative_flow_score: float = 0.0      # Story coherence
    pacing_score: float = 0.0              # Panel pacing appropriateness
    dialogue_quality_score: float = 0.0     # Dialogue naturalness
    
    # Composition metrics
    panel_composition_score: float = 0.0   # Comic grammar compliance
    layout_effectiveness_score: float = 0.0 # Layout quality
    text_placement_score: float = 0.0       # Speech bubble/text clarity
    
    # Emotional metrics
    emotion_conveyance_score: float = 0.0  # Emotional clarity
    emotional_arc_score: float = 0.0        # Story emotional progression
    
    # Technical metrics
    image_quality_score: float = 0.0       # Resolution, artifacts
    readability_score: float = 0.0          # Text legibility
    
    # Weights for overall calculation
    _weights: Dict[str, float] = field(default_factory=lambda: {
        'visual_consistency_score': 0.15,
        'color_harmony_score': 0.05,
        'style_consistency_score': 0.10,
        'narrative_flow_score': 0.15,
        'pacing_score': 0.08,
        'dialogue_quality_score': 0.07,
        'panel_composition_score': 0.10,
        'layout_effectiveness_score': 0.05,
        'text_placement_score': 0.03,
        'emotion_conveyance_score': 0.08,
        'emotional_arc_score': 0.05,
        'image_quality_score': 0.05,
        'readability_score': 0.04
    })
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        total = 0.0
        for attr, weight in self._weights.items():
            if hasattr(self, attr):
                total += getattr(self, attr) * weight
        return round(total, 3)
    
    @property
    def visual_score(self) -> float:
        """Aggregate visual quality score."""
        return round((
            self.visual_consistency_score * 0.4 +
            self.color_harmony_score * 0.3 +
            self.style_consistency_score * 0.3
        ), 3)
    
    @property
    def narrative_score(self) -> float:
        """Aggregate narrative quality score."""
        return round((
            self.narrative_flow_score * 0.5 +
            self.pacing_score * 0.25 +
            self.dialogue_quality_score * 0.25
        ), 3)
    
    @property
    def composition_score(self) -> float:
        """Aggregate composition quality score."""
        return round((
            self.panel_composition_score * 0.5 +
            self.layout_effectiveness_score * 0.3 +
            self.text_placement_score * 0.2
        ), 3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            'overall_score': self.overall_score,
            'visual_score': self.visual_score,
            'narrative_score': self.narrative_score,
            'composition_score': self.composition_score,
            'detailed_metrics': {}
        }
        
        for key in self._weights.keys():
            result['detailed_metrics'][key] = getattr(self, key)
        
        return result
    
    def get_grade(self) -> str:
        """Get letter grade based on overall score."""
        score = self.overall_score
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        elif score >= 0.4:
            return "D"
        else:
            return "F"
    
    def get_improvement_suggestions(self) -> List[str]:
        """Generate improvement suggestions based on weak scores."""
        suggestions = []
        
        if self.visual_consistency_score < 0.6:
            suggestions.append("Consider using character reference images to improve visual consistency across panels.")
        
        if self.color_harmony_score < 0.6:
            suggestions.append("Try selecting a more cohesive color palette for the entire comic.")
        
        if self.narrative_flow_score < 0.6:
            suggestions.append("The story progression could be clearer. Add transitional elements between panels.")
        
        if self.pacing_score < 0.6:
            suggestions.append("Adjust panel sizes to better reflect story pacing - larger panels for important moments.")
        
        if self.dialogue_quality_score < 0.6:
            suggestions.append("Dialogue could be more natural. Consider shorter, punchier lines.")
        
        if self.emotion_conveyance_score < 0.6:
            suggestions.append("Emotional content isn't clearly conveyed. Use appropriate colors and compositions for each mood.")
        
        if self.panel_composition_score < 0.6:
            suggestions.append("Panel compositions could follow comic conventions better (rule of thirds, clear focus).")
        
        return suggestions


class NarrativeCoherenceScorer:
    """
    Evaluates narrative coherence of comic panels.
    
    Checks:
    - Story progression
    - Cause-and-effect relationships
    - Character continuity in dialogue
    - Temporal flow
    """
    
    # Transition words indicating good narrative flow
    TRANSITION_WORDS = [
        'then', 'next', 'after', 'before', 'suddenly', 'meanwhile',
        'later', 'finally', 'however', 'but', 'therefore', 'so',
        'because', 'when', 'while', 'as', 'since', 'although'
    ]
    
    def __init__(self):
        """Initialize the scorer."""
        pass
    
    def score_narrative_flow(self, panel_descriptions: List[str]) -> float:
        """
        Score the narrative flow between panels.
        
        Args:
            panel_descriptions: List of panel description texts
        
        Returns:
            Score from 0.0 to 1.0
        """
        if len(panel_descriptions) < 2:
            return 0.5  # Can't evaluate flow with single panel
        
        scores = []
        
        # Check for narrative continuity between adjacent panels
        for i in range(len(panel_descriptions) - 1):
            current = panel_descriptions[i].lower()
            next_panel = panel_descriptions[i + 1].lower()
            
            # Check for transition words
            has_transition = any(tw in next_panel for tw in self.TRANSITION_WORDS)
            
            # Check for repeated entities (characters, places)
            current_words = set(re.findall(r'\b[A-Z][a-z]+\b', panel_descriptions[i]))
            next_words = set(re.findall(r'\b[A-Z][a-z]+\b', panel_descriptions[i + 1]))
            entity_overlap = len(current_words.intersection(next_words)) / max(len(current_words.union(next_words)), 1)
            
            # Check for action continuity
            current_verbs = set(re.findall(r'\b(?:ing|ed|s)\b', current))
            next_verbs = set(re.findall(r'\b(?:ing|ed|s)\b', next_panel))
            
            panel_score = 0.0
            panel_score += 0.3 if has_transition else 0.0
            panel_score += 0.4 * entity_overlap
            panel_score += 0.3 * (1.0 if len(next_panel) > 20 else 0.5)
            
            scores.append(panel_score)
        
        return round(np.mean(scores), 3) if scores else 0.5
    
    def score_dialogue_quality(self, dialogues: List[str]) -> float:
        """
        Score the quality of dialogue.
        
        Args:
            dialogues: List of dialogue texts
        
        Returns:
            Score from 0.0 to 1.0
        """
        if not dialogues or all(d == "..." for d in dialogues):
            return 0.5  # No dialogue to evaluate
        
        scores = []
        
        for dialogue in dialogues:
            if dialogue == "...":
                continue
            
            # Check dialogue length (too long is bad for comics)
            length_score = 1.0 if 5 <= len(dialogue) <= 100 else 0.6
            
            # Check for natural speech patterns
            has_punctuation = any(p in dialogue for p in ['.', '!', '?', '...'])
            
            # Check for character attribution
            has_speaker = ':' in dialogue
            
            score = length_score * 0.4 + (0.3 if has_punctuation else 0.0) + (0.3 if has_speaker else 0.15)
            scores.append(score)
        
        return round(np.mean(scores), 3) if scores else 0.5
    
    def score_pacing(self, panel_descriptions: List[str], 
                     emotions: List[str] = None) -> float:
        """
        Score the pacing of the comic.
        
        Good pacing has:
        - Setup panels at the start
        - Rising action
        - Climax
        - Resolution
        
        Args:
            panel_descriptions: Panel texts
            emotions: Optional emotion labels
        
        Returns:
            Score from 0.0 to 1.0
        """
        if len(panel_descriptions) < 3:
            return 0.5
        
        # Check for classic story structure
        has_setup = len(panel_descriptions[0]) > 30  # Setup usually has more description
        
        # Check for varied lengths (indicates pacing variation)
        lengths = [len(p) for p in panel_descriptions]
        length_variance = np.std(lengths) / (np.mean(lengths) + 1)
        
        # Check if last panel seems like conclusion
        last_panel = panel_descriptions[-1].lower()
        has_conclusion = any(word in last_panel for word in 
                           ['end', 'finally', 'ever after', 'moral', 'learned', 'conclusion'])
        
        score = 0.0
        score += 0.3 if has_setup else 0.15
        score += min(0.4, length_variance * 0.5)  # Some variance is good
        score += 0.3 if has_conclusion else 0.15
        
        return round(score, 3)


class VisualQualityScorer:
    """
    Evaluates visual quality of generated images.
    """
    
    def __init__(self):
        """Initialize the scorer."""
        pass
    
    def score_color_harmony(self, images: List[Image.Image]) -> float:
        """
        Score color harmony across images.
        
        Args:
            images: List of PIL images
        
        Returns:
            Score from 0.0 to 1.0
        """
        if not images:
            return 0.0
        
        all_palettes = []
        
        for img in images:
            # Get dominant colors
            img_small = img.copy()
            img_small.thumbnail((100, 100))
            if img_small.mode != 'RGB':
                img_small = img_small.convert('RGB')
            
            pixels = np.array(img_small).reshape(-1, 3)
            # Quantize
            quantized = (pixels // 32) * 32
            colors = Counter(map(tuple, quantized))
            top_colors = [c for c, _ in colors.most_common(5)]
            all_palettes.extend(top_colors)
        
        # Check for palette consistency
        all_colors = Counter(all_palettes)
        most_common = all_colors.most_common(5)
        
        if not most_common:
            return 0.5
        
        # Higher consistency if fewer unique colors dominate
        top_count = sum(count for _, count in most_common)
        total_count = sum(all_colors.values())
        
        consistency = top_count / max(total_count, 1)
        
        return round(min(1.0, consistency * 1.2), 3)
    
    def score_visual_consistency(self, images: List[Image.Image]) -> float:
        """
        Score visual consistency using histogram comparison.
        
        Args:
            images: List of PIL images
        
        Returns:
            Score from 0.0 to 1.0
        """
        if len(images) < 2:
            return 1.0
        
        similarities = []
        
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                # Compute histogram similarity
                img1 = images[i].copy().resize((100, 100)).convert('RGB')
                img2 = images[j].copy().resize((100, 100)).convert('RGB')
                
                hist1 = np.array(img1.histogram(), dtype=np.float32)
                hist2 = np.array(img2.histogram(), dtype=np.float32)
                
                hist1 /= hist1.sum() + 1e-10
                hist2 /= hist2.sum() + 1e-10
                
                similarity = np.minimum(hist1, hist2).sum()
                similarities.append(similarity)
        
        return round(np.mean(similarities), 3)
    
    def score_style_consistency(self, images: List[Image.Image]) -> float:
        """
        Score art style consistency.
        
        Uses texture and edge detection patterns.
        
        Args:
            images: List of PIL images
        
        Returns:
            Score from 0.0 to 1.0
        """
        if len(images) < 2:
            return 1.0
        
        # Simple edge detection for style analysis
        edge_densities = []
        
        for img in images:
            gray = img.convert('L')
            pixels = np.array(gray)
            
            # Simple edge detection using gradient
            gx = np.abs(np.diff(pixels, axis=1))
            gy = np.abs(np.diff(pixels, axis=0))
            
            edge_density = (gx.mean() + gy.mean()) / 2 / 255
            edge_densities.append(edge_density)
        
        # Lower variance in edge density = more consistent style
        variance = np.std(edge_densities)
        
        # Map variance to score (lower variance = higher score)
        score = max(0.0, 1.0 - variance * 5)
        
        return round(score, 3)
    
    def score_image_quality(self, images: List[Image.Image]) -> float:
        """
        Score overall image quality.
        
        Args:
            images: List of PIL images
        
        Returns:
            Score from 0.0 to 1.0
        """
        if not images:
            return 0.0
        
        scores = []
        
        for img in images:
            # Check resolution
            min_dim = min(img.size)
            resolution_score = min(1.0, min_dim / 512)  # 512+ is good
            
            # Check for excessive noise (high variance in small patches)
            gray = np.array(img.convert('L'))
            
            # Compute local variance
            local_variances = []
            patch_size = 16
            for y in range(0, gray.shape[0] - patch_size, patch_size):
                for x in range(0, gray.shape[1] - patch_size, patch_size):
                    patch = gray[y:y+patch_size, x:x+patch_size]
                    local_variances.append(np.var(patch))
            
            avg_variance = np.mean(local_variances)
            # Very high variance might indicate noise
            noise_score = max(0.0, 1.0 - (avg_variance / 5000))
            
            img_score = resolution_score * 0.6 + noise_score * 0.4
            scores.append(img_score)
        
        return round(np.mean(scores), 3)


class ComicGrammarChecker:
    """
    Checks adherence to comic visual grammar conventions.
    """
    
    def __init__(self):
        """Initialize the checker."""
        pass
    
    def score_composition(self, layouts: List[Dict] = None, 
                         num_panels: int = 6) -> float:
        """
        Score panel composition and layout.
        
        Args:
            layouts: Layout information for panels
            num_panels: Number of panels
        
        Returns:
            Score from 0.0 to 1.0
        """
        score = 0.5  # Base score
        
        if num_panels == 6:
            score += 0.2  # Standard 6-panel is good
        elif 4 <= num_panels <= 8:
            score += 0.1
        
        if layouts:
            # Check for variety in panel sizes
            sizes = [l.get('relative_size', 'medium') for l in layouts]
            size_variety = len(set(sizes)) / len(sizes)
            score += size_variety * 0.3
        else:
            score += 0.15  # Assume standard layout
        
        return round(min(1.0, score), 3)
    
    def score_layout_effectiveness(self, layouts: List[Dict] = None) -> float:
        """
        Score how effective the layout is for storytelling.
        
        Args:
            layouts: Layout information
        
        Returns:
            Score from 0.0 to 1.0
        """
        if not layouts:
            return 0.6  # Default score for standard layout
        
        score = 0.5
        
        # Check for climax panel (should be larger)
        has_large = any(l.get('relative_size') in ['large', 'splash'] for l in layouts)
        if has_large:
            score += 0.25
        
        # Check for proper reading flow
        # Panels should generally flow left-to-right, top-to-bottom
        x_positions = [l.get('x', 0) for l in layouts]
        y_positions = [l.get('y', 0) for l in layouts]
        
        # Check monotonic increase (roughly)
        proper_flow = sum(1 for i in range(1, len(y_positions)) 
                        if y_positions[i] >= y_positions[i-1]) / max(len(y_positions) - 1, 1)
        score += proper_flow * 0.25
        
        return round(score, 3)


class ComicEvaluator:
    """
    Main evaluator class that combines all scoring components.
    """
    
    def __init__(self):
        """Initialize all scorer components."""
        self.narrative_scorer = NarrativeCoherenceScorer()
        self.visual_scorer = VisualQualityScorer()
        self.grammar_checker = ComicGrammarChecker()
    
    def evaluate(self, 
                 panel_images: List[Image.Image] = None,
                 panel_descriptions: List[str] = None,
                 panel_dialogues: List[str] = None,
                 layouts: List[Dict] = None,
                 emotions: List[str] = None) -> ComicQualityMetrics:
        """
        Perform comprehensive evaluation of a comic.
        
        Args:
            panel_images: List of panel images
            panel_descriptions: Panel description texts
            panel_dialogues: Panel dialogue texts
            layouts: Layout information
            emotions: Emotion labels for each panel
        
        Returns:
            ComicQualityMetrics with all scores
        """
        metrics = ComicQualityMetrics()
        
        # Visual metrics (require images)
        if panel_images:
            metrics.visual_consistency_score = self.visual_scorer.score_visual_consistency(panel_images)
            metrics.color_harmony_score = self.visual_scorer.score_color_harmony(panel_images)
            metrics.style_consistency_score = self.visual_scorer.score_style_consistency(panel_images)
            metrics.image_quality_score = self.visual_scorer.score_image_quality(panel_images)
        else:
            # Default scores if no images
            metrics.visual_consistency_score = 0.5
            metrics.color_harmony_score = 0.5
            metrics.style_consistency_score = 0.5
            metrics.image_quality_score = 0.5
        
        # Narrative metrics (require text)
        if panel_descriptions:
            metrics.narrative_flow_score = self.narrative_scorer.score_narrative_flow(panel_descriptions)
            metrics.pacing_score = self.narrative_scorer.score_pacing(panel_descriptions, emotions)
        else:
            metrics.narrative_flow_score = 0.5
            metrics.pacing_score = 0.5
        
        if panel_dialogues:
            metrics.dialogue_quality_score = self.narrative_scorer.score_dialogue_quality(panel_dialogues)
        else:
            metrics.dialogue_quality_score = 0.5
        
        # Composition metrics
        num_panels = len(panel_images) if panel_images else len(panel_descriptions) if panel_descriptions else 6
        metrics.panel_composition_score = self.grammar_checker.score_composition(layouts, num_panels)
        metrics.layout_effectiveness_score = self.grammar_checker.score_layout_effectiveness(layouts)
        
        # Text placement (would need OCR for full analysis)
        metrics.text_placement_score = 0.7  # Default
        
        # Emotion conveyance
        if emotions and panel_descriptions:
            # Check if emotion keywords appear in descriptions
            emotion_matches = sum(1 for e, d in zip(emotions, panel_descriptions) 
                                 if e.lower() in d.lower()) / max(len(emotions), 1)
            metrics.emotion_conveyance_score = 0.5 + emotion_matches * 0.5
        else:
            metrics.emotion_conveyance_score = 0.5
        
        # Emotional arc
        if emotions:
            # Check for variety in emotions
            unique_emotions = len(set(emotions)) / max(len(emotions), 1)
            metrics.emotional_arc_score = 0.4 + unique_emotions * 0.6
        else:
            metrics.emotional_arc_score = 0.5
        
        # Readability (default)
        metrics.readability_score = 0.7
        
        return metrics
    
    def generate_report(self, metrics: ComicQualityMetrics) -> Dict[str, Any]:
        """
        Generate a detailed evaluation report.
        
        Args:
            metrics: Computed metrics
        
        Returns:
            Detailed report dictionary
        """
        return {
            "overall_grade": metrics.get_grade(),
            "overall_score": metrics.overall_score,
            "summary_scores": {
                "visual": metrics.visual_score,
                "narrative": metrics.narrative_score,
                "composition": metrics.composition_score
            },
            "detailed_metrics": metrics.to_dict()['detailed_metrics'],
            "improvement_suggestions": metrics.get_improvement_suggestions(),
            "interpretation": self._interpret_scores(metrics)
        }
    
    def _interpret_scores(self, metrics: ComicQualityMetrics) -> str:
        """Generate human-readable interpretation."""
        overall = metrics.overall_score
        
        if overall >= 0.8:
            return "Excellent comic quality with strong visual consistency and narrative flow."
        elif overall >= 0.65:
            return "Good comic quality. Some areas could be improved for professional-level output."
        elif overall >= 0.5:
            return "Average quality. Consider improvements in visual consistency and story pacing."
        else:
            return "Below average quality. Significant improvements needed in multiple areas."


def generate_evaluation_report(panel_images: List[Image.Image] = None,
                               panel_descriptions: List[str] = None,
                               panel_dialogues: List[str] = None,
                               layouts: List[Dict] = None,
                               emotions: List[str] = None) -> Tuple[ComicQualityMetrics, Dict]:
    """
    Convenience function to generate a complete evaluation.
    
    Returns:
        Tuple of (metrics, report_dict)
    """
    evaluator = ComicEvaluator()
    metrics = evaluator.evaluate(
        panel_images=panel_images,
        panel_descriptions=panel_descriptions,
        panel_dialogues=panel_dialogues,
        layouts=layouts,
        emotions=emotions
    )
    report = evaluator.generate_report(metrics)
    
    return metrics, report
