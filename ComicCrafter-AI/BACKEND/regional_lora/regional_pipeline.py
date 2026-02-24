"""
Regional Diffusion Pipeline
===========================
Custom Stable Diffusion pipeline with Regional LoRA Attention.

This is the CORE TECHNICAL INNOVATION for the IEEE paper.

Standard diffusion applies LoRA weights globally to all cross-attention layers.
Our approach:
1. During each denoising step, we intercept cross-attention computation
2. For each spatial position, we determine which LoRA(s) should apply
3. We compute attention separately for each LoRA and combine using masks
4. The result: Different LoRAs affect different pixel regions

Mathematical Formulation:
Let A_base(Q, K, V) be standard cross-attention
Let A_lora_i(Q, K, V) be attention with LoRA_i applied
Let M_i be the attention mask for character i

Standard: A_final = A_base + Σ A_lora_i (global mixture, causes bleeding)
Ours: A_final = A_base + Σ (M_i ⊙ A_lora_i) (regional, no bleeding)

Implementation uses custom attention processor hooks in diffusers.

Requirements:
- torch >= 2.0.0
- diffusers >= 0.25.0
- transformers >= 4.35.0
- accelerate >= 0.25.0
- safetensors >= 0.4.0

For local GPU: NVIDIA GPU with >= 8GB VRAM (16GB+ recommended)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path

# Check for dependencies
TORCH_AVAILABLE = False
DIFFUSERS_AVAILABLE = False
SAFETENSORS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available. Regional pipeline requires torch >= 2.0")

try:
    from diffusers import StableDiffusionXLPipeline
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("⚠️ Diffusers not available. Run: pip install diffusers>=0.25.0")

try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class LoRALayer:
    """Represents a loaded LoRA layer."""
    name: str
    up_weight: Any  # torch.Tensor when available
    down_weight: Any  # torch.Tensor when available
    alpha: float = 1.0
    rank: int = 0


class RegionalAttnProcessor:
    """
    Custom attention processor that applies different LoRAs to different regions.
    
    This is the KEY INNOVATION.
    
    Instead of: output = attention(x) + lora_adjustment(x)
    We do: output = attention(x) + Σ(mask_i * lora_i_adjustment(x))
    """
    
    def __init__(self,
                 hidden_size: int,
                 cross_attention_dim: Optional[int] = None,
                 base_processor: Optional[Any] = None):
        """
        Initialize the regional attention processor.
        
        Args:
            hidden_size: Hidden dimension of attention
            cross_attention_dim: Cross attention dimension (for encoder KV)
            base_processor: Base attention processor to use
        """
        if not TORCH_AVAILABLE or not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "RegionalAttnProcessor requires PyTorch and diffusers. "
                "Install with: pip install torch diffusers>=0.25.0"
            )
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.base_processor = base_processor or AttnProcessor2_0()
        
        # LoRA layers by name
        self.lora_layers: Dict[str, Dict[str, LoRALayer]] = {}
        
        # Regional masks (set during generation)
        self.regional_masks: Dict[str, Any] = {}  # Dict[str, Tensor]
        
        # Current spatial size (set during generation)
        self.current_hw: Optional[Tuple[int, int]] = None
    
    def set_lora_layers(self, name: str, layers: Dict[str, LoRALayer]):
        """Register LoRA layers for a character."""
        self.lora_layers[name] = layers
    
    def set_regional_masks(self, masks: Dict[str, Any]):
        """Set regional masks for current generation."""
        self.regional_masks = masks
    
    def set_spatial_size(self, height: int, width: int):
        """Set current spatial dimensions."""
        self.current_hw = (height, width)
    
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        """
        Custom attention with regional LoRA application.
        """
        # Get base attention output
        base_output = self._compute_base_attention(
            attn, hidden_states, encoder_hidden_states, attention_mask
        )
        
        if not self.lora_layers or not self.regional_masks:
            return base_output
        
        # Apply regional LoRA adjustments
        lora_output = torch.zeros_like(base_output)
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Infer spatial dimensions from sequence length
        if self.current_hw:
            h, w = self.current_hw
        else:
            # Assume square
            hw = int(seq_len ** 0.5)
            h, w = hw, hw
        
        for char_name, lora_dict in self.lora_layers.items():
            if char_name not in self.regional_masks:
                continue
            
            # Get mask for this character
            mask = self.regional_masks[char_name]  # [1, 1, H, W]
            
            # Resize mask to match attention spatial dimensions
            if mask.shape[-2:] != (h, w):
                mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
            
            # Flatten mask to sequence dimension: [1, 1, H, W] -> [1, H*W, 1]
            mask_flat = mask.flatten(2).transpose(1, 2)  # [1, H*W, 1]
            
            # Compute LoRA adjustment for this character
            lora_adjustment = self._compute_lora_adjustment(
                attn, hidden_states, encoder_hidden_states, lora_dict
            )
            
            # Apply mask: adjustment only where mask is active
            masked_adjustment = lora_adjustment * mask_flat
            
            lora_output = lora_output + masked_adjustment
        
        return base_output + lora_output
    
    def _compute_base_attention(self, attn, hidden_states, encoder_hidden_states, attention_mask):
        """Compute standard attention without LoRA."""
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb=None)
        
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        # Project to Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Compute attention using PyTorch's scaled_dot_product_attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        return hidden_states / attn.rescale_output_factor
    
    def _compute_lora_adjustment(self, attn, hidden_states, encoder_hidden_states, lora_layers):
        """Compute the LoRA adjustment for attention."""
        adjustment = torch.zeros_like(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        # Check for Q LoRA
        if "to_q" in lora_layers:
            lora = lora_layers["to_q"]
            scale = lora.alpha / lora.rank if lora.rank > 0 else 1.0
            delta = hidden_states @ lora.down_weight @ lora.up_weight * scale
            adjustment = adjustment + delta
        
        # Check for K LoRA
        if "to_k" in lora_layers:
            lora = lora_layers["to_k"]
            scale = lora.alpha / lora.rank if lora.rank > 0 else 1.0
            delta = encoder_hidden_states @ lora.down_weight @ lora.up_weight * scale
            adjustment = adjustment + delta
        
        # Check for V LoRA
        if "to_v" in lora_layers:
            lora = lora_layers["to_v"]
            scale = lora.alpha / lora.rank if lora.rank > 0 else 1.0
            delta = encoder_hidden_states @ lora.down_weight @ lora.up_weight * scale
            adjustment = adjustment + delta
        
        return adjustment


class RegionalDiffusionPipeline:
    """
    Custom diffusion pipeline with regional LoRA support.
    
    This wraps a standard Stable Diffusion pipeline and adds
    the ability to apply different LoRAs to different spatial regions.
    """
    
    def __init__(self,
                 model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "cuda",
                 torch_dtype: Optional[Any] = None,
                 use_safetensors: bool = True):
        """
        Initialize the regional pipeline.
        
        Args:
            model_id: HuggingFace model ID or path
            device: Device to use
            torch_dtype: Torch dtype (defaults to float16 on GPU)
            use_safetensors: Whether to use safetensors format
        """
        if not TORCH_AVAILABLE or not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "RegionalDiffusionPipeline requires PyTorch and diffusers. "
                "Install with: pip install torch diffusers>=0.25.0"
            )
        
        self.device = device
        self.torch_dtype = torch_dtype or (torch.float16 if device == "cuda" else torch.float32)
        
        print(f"Loading model: {model_id}")
        
        # Load base pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            use_safetensors=use_safetensors,
            variant="fp16" if self.torch_dtype == torch.float16 else None
        ).to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            self.pipe.enable_model_cpu_offload()
        
        # Store original attention processors
        self.original_attn_processors = self.pipe.unet.attn_processors.copy()
        
        # Regional attention processors
        self.regional_processors: Dict[str, RegionalAttnProcessor] = {}
        
        # Loaded LoRAs
        self.loaded_loras: Dict[str, Dict] = {}
        
        # Set up regional processors
        self._setup_regional_processors()
        
        print(f"✅ Regional pipeline ready on {device}")
    
    def _setup_regional_processors(self):
        """Replace attention processors with regional versions."""
        attn_procs = {}
        
        for name, processor in self.pipe.unet.attn_processors.items():
            # Create regional processor
            regional_proc = RegionalAttnProcessor(
                hidden_size=self.pipe.unet.config.cross_attention_dim or 768,
                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                base_processor=processor
            )
            
            attn_procs[name] = regional_proc
            self.regional_processors[name] = regional_proc
        
        self.pipe.unet.set_attn_processor(attn_procs)
    
    def load_lora(self, 
                  lora_path: str,
                  character_name: str,
                  weight: float = 1.0) -> bool:
        """
        Load a LoRA for a specific character.
        
        Args:
            lora_path: Path to the .safetensors LoRA file
            character_name: Name to associate with this LoRA
            weight: Weight to apply (0-1)
        
        Returns:
            True if successful
        """
        if not SAFETENSORS_AVAILABLE:
            print("⚠️ safetensors not available")
            return False
        
        if not os.path.exists(lora_path):
            print(f"⚠️ LoRA file not found: {lora_path}")
            return False
        
        print(f"Loading LoRA for {character_name}: {lora_path}")
        
        # Load LoRA weights
        lora_weights = load_safetensors(lora_path)
        
        # Parse and organize weights by layer
        parsed_layers = self._parse_lora_weights(lora_weights)
        
        # Register with attention processors
        for proc_name, proc in self.regional_processors.items():
            # Find matching layers for this processor
            matching_layers = {}
            
            for layer_name, layer_weights in parsed_layers.items():
                if any(part in proc_name for part in layer_name.split('.')):
                    key = layer_name.split('.')[-1]  # to_q, to_k, to_v, etc.
                    matching_layers[key] = layer_weights
            
            if matching_layers:
                proc.set_lora_layers(character_name, matching_layers)
        
        self.loaded_loras[character_name] = {
            "path": lora_path,
            "weight": weight,
            "layers": parsed_layers
        }
        
        print(f"✅ Loaded LoRA for {character_name}")
        return True
    
    def _parse_lora_weights(self, weights: Dict[str, Any]) -> Dict[str, LoRALayer]:
        """Parse raw LoRA weights into LoRALayer objects."""
        layers = {}
        
        # Group by layer name
        layer_groups = {}
        for key, tensor in weights.items():
            # Extract layer name and weight type (up/down)
            if '.lora_down.' in key:
                layer_name = key.replace('.lora_down.weight', '')
                layer_groups.setdefault(layer_name, {})['down'] = tensor
            elif '.lora_up.' in key:
                layer_name = key.replace('.lora_up.weight', '')
                layer_groups.setdefault(layer_name, {})['up'] = tensor
            elif '.alpha' in key:
                layer_name = key.replace('.alpha', '')
                layer_groups.setdefault(layer_name, {})['alpha'] = tensor.item()
        
        # Create LoRALayer objects
        for layer_name, group in layer_groups.items():
            if 'up' in group and 'down' in group:
                down = group['down'].to(self.device, dtype=self.torch_dtype)
                up = group['up'].to(self.device, dtype=self.torch_dtype)
                
                layers[layer_name] = LoRALayer(
                    name=layer_name,
                    up_weight=up,
                    down_weight=down,
                    alpha=group.get('alpha', 1.0),
                    rank=down.shape[0]
                )
        
        return layers
    
    def unload_lora(self, character_name: str):
        """Unload a character's LoRA."""
        if character_name in self.loaded_loras:
            del self.loaded_loras[character_name]
            
            for proc in self.regional_processors.values():
                if character_name in proc.lora_layers:
                    del proc.lora_layers[character_name]
    
    def generate_regional(
        self,
        prompt: str,
        attention_masks: Dict[str, Any],
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ):
        """
        Generate an image with regional LoRA application.
        
        Args:
            prompt: Text prompt
            attention_masks: Dict mapping character name to attention mask tensor
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            seed: Random seed
        
        Returns:
            Generated PIL Image
        """
        from PIL import Image
        
        # Set regional masks on all processors
        for proc in self.regional_processors.values():
            proc.set_regional_masks(attention_masks)
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        return result.images[0]
    
    def generate_with_layout(
        self,
        prompt: str,
        character_masks: List,
        background_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Generate with explicit character layout masks.
        
        Args:
            prompt: Main prompt
            character_masks: List of RegionalMask objects
            background_prompt: Optional background-specific prompt
            **kwargs: Additional generation args
        
        Returns:
            Generated image
        """
        # Convert masks to attention tensors
        attention_masks = {}
        
        latent_h = kwargs.get('height', 1024) // 8
        latent_w = kwargs.get('width', 1024) // 8
        
        for mask in character_masks:
            # Resize mask to latent dimensions
            resized = mask.resize_for_latent(latent_h, latent_w)
            tensor = resized.to_torch(self.device)
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            char_name = mask.character_name
            if char_name in self.loaded_loras:
                attention_masks[char_name] = tensor
        
        return self.generate_regional(
            prompt=prompt,
            attention_masks=attention_masks,
            **kwargs
        )
    
    def reset_processors(self):
        """Reset to original attention processors (disable regional LoRA)."""
        self.pipe.unet.set_attn_processor(self.original_attn_processors)
    
    def enable_regional_processors(self):
        """Re-enable regional attention processors."""
        procs = {name: proc for name, proc in self.regional_processors.items()}
        self.pipe.unet.set_attn_processor(procs)


def generate_with_regional_lora(
    prompt: str,
    character_loras: Dict[str, str],  # character_name -> lora_path
    character_regions: Dict[str, Tuple[float, float, float, float]],  # character_name -> bbox
    output_path: Optional[str] = None,
    **kwargs
):
    """
    Quick function to generate with regional LoRA.
    
    Args:
        prompt: Text prompt
        character_loras: Dict mapping character names to LoRA paths
        character_regions: Dict mapping character names to bounding boxes (x1, y1, x2, y2)
        output_path: Optional path to save the image
        **kwargs: Additional generation parameters
    
    Returns:
        Generated PIL Image
    """
    from .attention_mask_generator import AttentionMaskGenerator
    
    # Initialize pipeline
    pipe = RegionalDiffusionPipeline(
        model_id=kwargs.pop('model_id', 'stabilityai/stable-diffusion-xl-base-1.0'),
        device=kwargs.pop('device', 'cuda')
    )
    
    # Load LoRAs
    for char_name, lora_path in character_loras.items():
        pipe.load_lora(lora_path, char_name)
    
    # Generate masks
    width = kwargs.get('width', 1024)
    height = kwargs.get('height', 1024)
    
    mask_gen = AttentionMaskGenerator()
    masks = []
    
    for char_name, bbox in character_regions.items():
        mask = mask_gen.generate_mask(
            bbox=bbox,
            image_size=(width, height),
            character_name=char_name,
            lora_id=char_name
        )
        masks.append(mask)
    
    # Generate
    image = pipe.generate_with_layout(
        prompt=prompt,
        character_masks=masks,
        **kwargs
    )
    
    if output_path:
        image.save(output_path)
    
    return image
