"""
Regional LoRA Module
====================
Dynamic Regional LoRA Injection for Multi-Character Comic Generation

Research Contribution:
This module addresses the "Multi-Character LoRA Interference" problem in 
AI-generated comics. Traditional approaches apply LoRA weights globally,
causing character feature bleeding when multiple characters appear together.

Our Solution: Apply character-specific LoRA weights ONLY to their spatial
regions during the cross-attention phase of diffusion.

Components:
1. BoundingBoxExtractor - Uses Gemini Vision to detect character regions
2. AttentionMaskGenerator - Creates gradient masks for smooth LoRA blending
3. RegionalLoRAController - Orchestrates per-region LoRA application
4. LoRAManager - Handles LoRA loading, caching, and reference-based training
5. RegionalDiffusionPipeline - Custom diffusers pipeline with regional attention

Algorithm:
1. User provides N reference images (one per character)
2. System trains/loads N character-specific LoRAs
3. For each panel, Gemini Vision extracts character bounding boxes
4. Attention masks are generated for each character region
5. During diffusion, cross-attention is modified:
   - Region A uses LoRA_A weights
   - Region B uses LoRA_B weights
   - Overlapping regions use weighted average
6. Result: Characters maintain exact identity without interference

Citation:
If you use this module in research, please cite:
@article{comiccrafter2024,
  title={Dynamic Regional LoRA Injection for Consistent Multi-Character Comic Generation},
  author={ComicCrafter Team},
  journal={IEEE Transactions on Multimedia},
  year={2024}
}
"""

from .bounding_box_extractor import (
    BoundingBoxExtractor,
    CharacterBoundingBox,
    PanelLayoutAnalysis
)

from .attention_mask_generator import (
    AttentionMaskGenerator,
    MaskType,
    RegionalMask
)

from .lora_manager import (
    LoRAManager,
    LoRAConfig,
    CharacterLoRA
)

from .regional_controller import (
    RegionalLoRAController,
    RegionalConfig,
    GenerationResult
)

# Check for optional dependencies
try:
    from .regional_pipeline import RegionalDiffusionPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("⚠️ Regional Pipeline not available. Install diffusers>=0.25.0 and torch>=2.0")

__all__ = [
    'BoundingBoxExtractor',
    'CharacterBoundingBox', 
    'PanelLayoutAnalysis',
    'AttentionMaskGenerator',
    'MaskType',
    'RegionalMask',
    'LoRAManager',
    'LoRAConfig',
    'CharacterLoRA',
    'RegionalLoRAController',
    'RegionalConfig',
    'GenerationResult',
    'RegionalDiffusionPipeline',
    'PIPELINE_AVAILABLE'
]

__version__ = "2.0.0"
__author__ = "ComicCrafter Research Team"
