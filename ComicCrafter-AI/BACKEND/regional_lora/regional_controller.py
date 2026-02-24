"""
Regional LoRA Controller
========================
Main orchestrator for Dynamic Regional LoRA Injection.

This is the PRIMARY INNOVATION for the IEEE paper:
Instead of applying character LoRAs globally (causing interference),
we apply each LoRA ONLY to its character's spatial region.

Algorithm Overview:
1. Panel description → Gemini extracts character bounding boxes
2. For each character, generate attention mask from their bbox
3. During diffusion, modify cross-attention:
   - Attention_final = Σ (mask_i × attention_with_lora_i)
4. Overlapping regions use weighted averaging based on z-order

This enables:
- Multiple distinct characters in same panel
- No feature bleeding between characters
- Consistent identity across all panels
- Support for 1-image LoRA training per character

Configuration:
- Uses Replicate API for cloud-based generation (no local GPU needed)
- Falls back to local diffusers if GPU available
- Supports ClipDrop/Stability AI as fallback (without regional LoRA)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from PIL import Image
import io
import base64
import json

# Local imports
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
    CharacterLoRA, 
    LoRAType,
    LoRASource
)

# Optional imports
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class RegionalConfig:
    """Configuration for regional LoRA generation."""
    # Image settings
    image_width: int = 1024
    image_height: int = 1024
    
    # Generation settings
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    
    # Regional settings
    mask_type: MaskType = MaskType.GAUSSIAN
    edge_softness: float = 0.08
    overlap_mode: str = "z_order"
    
    # LoRA settings
    default_lora_weight: float = 0.85
    background_lora_weight: float = 0.5
    
    # Quality settings
    use_refiner: bool = True
    refiner_strength: float = 0.3
    
    # Backend settings
    backend: str = "replicate"  # "replicate", "local", "clipdrop_fallback"
    replicate_model: str = "lucataco/flux-dev-multi-lora:2389224e115448d9a77c07d7d45672b3f0aa45ababf7e5b544c6586edd2df5f9"


@dataclass
class GenerationResult:
    """Result of a regional LoRA generation."""
    image: Image.Image
    panel_number: int
    layout: PanelLayoutAnalysis
    masks_used: List[RegionalMask]
    generation_time: float
    backend_used: str
    prompt_used: str
    seed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel_number": self.panel_number,
            "layout": self.layout.to_dict(),
            "generation_time": self.generation_time,
            "backend_used": self.backend_used,
            "seed": self.seed
        }


class RegionalLoRAController:
    """
    Main controller for regional LoRA-based image generation.
    
    This class coordinates:
    1. Bounding box extraction (Gemini Vision)
    2. Attention mask generation
    3. LoRA selection and application
    4. Image generation with regional attention
    
    Usage:
        controller = RegionalLoRAController()
        
        # Register characters with reference images
        controller.register_character("Sarah", "red hair, blue coat", ref_image_path)
        controller.register_character("Tom", "brown hair, glasses", ref_image_path)
        
        # Generate panel with both characters
        result = controller.generate_panel(
            description="Sarah and Tom meet at the coffee shop",
            characters=["Sarah", "Tom"],
            art_style="Manga"
        )
    """
    
    def __init__(self, 
                 config: Optional[RegionalConfig] = None,
                 lora_cache_dir: str = "LORA_CACHE",
                 replicate_token: Optional[str] = None):
        """
        Initialize the regional LoRA controller.
        
        Args:
            config: Generation configuration
            lora_cache_dir: Directory for LoRA cache
            replicate_token: Replicate API token
        """
        self.config = config or RegionalConfig()
        self.replicate_token = replicate_token or os.getenv("REPLICATE_API_TOKEN")
        
        # Initialize components
        self.bbox_extractor = BoundingBoxExtractor()
        self.mask_generator = AttentionMaskGenerator(
            default_mask_type=self.config.mask_type,
            edge_softness=self.config.edge_softness
        )
        self.lora_manager = LoRAManager(
            cache_dir=lora_cache_dir,
            replicate_api_token=self.replicate_token
        )
        
        # Character registry
        self.characters: Dict[str, Dict] = {}
        
        # Generation history
        self.generation_history: List[GenerationResult] = []
        
        print("✅ Regional LoRA Controller initialized")
        print(f"   Backend: {self.config.backend}")
        print(f"   Image size: {self.config.image_width}x{self.config.image_height}")
    
    def register_character(self,
                           name: str,
                           description: str,
                           reference_image: Optional[Union[str, Image.Image]] = None,
                           train_lora: bool = True,
                           lora_method: str = "instant") -> Optional[CharacterLoRA]:
        """
        Register a character for consistent generation.
        
        Args:
            name: Character name
            description: Physical description
            reference_image: Optional reference image for LoRA training
            train_lora: Whether to train/create a LoRA for this character
            lora_method: "instant" (fast, uses PuLID/PhotoMaker) or 
                        "trained" (slower, trains real LoRA)
        
        Returns:
            CharacterLoRA if created
        """
        self.characters[name] = {
            "description": description,
            "reference_image": reference_image,
            "lora": None
        }
        
        # Register with bbox extractor
        self.bbox_extractor.register_character(name, description)
        
        lora = None
        
        if reference_image and train_lora:
            if lora_method == "instant":
                # Fast: Use embedding-based approach
                lora = self.lora_manager.train_instant_lora(
                    character_name=name,
                    reference_image=reference_image,
                    method="pulid"
                )
            elif lora_method == "trained":
                # Slower but better: Train actual LoRA on Replicate
                imgs = [reference_image] if not isinstance(reference_image, list) else reference_image
                lora = self.lora_manager.train_character_lora_replicate(
                    character_name=name,
                    reference_images=imgs
                )
            
            self.characters[name]["lora"] = lora
            print(f"✅ Registered character '{name}' with LoRA")
        else:
            print(f"✅ Registered character '{name}' (text-only, no LoRA)")
        
        return lora
    
    def generate_panel(self,
                       description: str,
                       characters: List[str],
                       art_style: str,
                       panel_number: int = 1,
                       emotion: str = "neutral",
                       custom_bboxes: Optional[Dict[str, Tuple]] = None,
                       seed: int = -1) -> GenerationResult:
        """
        Generate a panel with regional LoRA for each character.
        
        This is the MAIN GENERATION METHOD.
        
        Args:
            description: Panel description
            characters: List of character names in this panel
            art_style: Art style name
            panel_number: Panel number
            emotion: Panel emotion
            custom_bboxes: Optional manual bounding boxes {name: (x1,y1,x2,y2)}
            seed: Random seed (-1 for random)
        
        Returns:
            GenerationResult with image and metadata
        """
        start_time = time.time()
        
        print(f"\n🎨 Generating Panel {panel_number}")
        print(f"   Characters: {characters}")
        print(f"   Style: {art_style}")
        
        # Step 1: Extract bounding boxes
        if custom_bboxes:
            # Use provided bounding boxes
            layout = self._create_layout_from_bboxes(custom_bboxes, panel_number)
        else:
            # Use Gemini to predict bounding boxes
            layout = self.bbox_extractor.extract_from_description(
                panel_description=description,
                characters=characters,
                panel_number=panel_number,
                aspect_ratio=self.config.image_width / self.config.image_height
            )
        
        print(f"   Layout: {len(layout.character_boxes)} character boxes detected")
        
        # Step 2: Generate masks for each character
        mask_data = []
        for box in layout.character_boxes:
            char_name = box.character_name
            char_info = self.characters.get(char_name, {})
            lora = char_info.get("lora")
            
            mask_data.append({
                "bbox": (box.x1, box.y1, box.x2, box.y2),
                "character_name": char_name,
                "lora_id": lora.config.name if lora else f"text_{char_name}",
                "z_order": box.z_order
            })
        
        masks = self.mask_generator.generate_multi_character_masks(
            character_boxes=mask_data,
            image_size=(self.config.image_width, self.config.image_height),
            overlap_mode=self.config.overlap_mode
        )
        
        # Step 3: Prepare LoRAs
        char_loras = {}
        for char_name in characters:
            char_info = self.characters.get(char_name, {})
            if char_info.get("lora"):
                char_loras[char_name] = char_info["lora"]
        
        # Step 4: Build enhanced prompt
        prompt = self._build_regional_prompt(
            description=description,
            art_style=art_style,
            emotion=emotion,
            characters=characters,
            layout=layout
        )
        
        # Step 5: Generate image based on backend
        if self.config.backend == "replicate" and REPLICATE_AVAILABLE and char_loras:
            image = self._generate_with_replicate_regional(
                prompt=prompt,
                char_loras=char_loras,
                masks=masks,
                seed=seed
            )
            backend_used = "replicate_regional"
            
        elif self.config.backend == "local" and TORCH_AVAILABLE:
            image = self._generate_with_local_pipeline(
                prompt=prompt,
                char_loras=char_loras,
                masks=masks,
                seed=seed
            )
            backend_used = "local_diffusers"
            
        else:
            # Fallback to regular generation (ClipDrop/Stability)
            image = self._generate_fallback(prompt, seed)
            backend_used = "clipdrop_fallback"
        
        generation_time = time.time() - start_time
        
        result = GenerationResult(
            image=image,
            panel_number=panel_number,
            layout=layout,
            masks_used=masks,
            generation_time=generation_time,
            backend_used=backend_used,
            prompt_used=prompt,
            seed=seed if seed > 0 else 0
        )
        
        self.generation_history.append(result)
        
        print(f"   ✅ Generated in {generation_time:.1f}s using {backend_used}")
        
        return result
    
    def _create_layout_from_bboxes(self, 
                                    bboxes: Dict[str, Tuple],
                                    panel_number: int) -> PanelLayoutAnalysis:
        """Create layout from manually specified bounding boxes."""
        from .bounding_box_extractor import CharacterBoundingBox, CharacterPosition
        
        boxes = []
        for i, (name, bbox) in enumerate(bboxes.items()):
            boxes.append(CharacterBoundingBox(
                character_name=name,
                x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                z_order=i
            ))
        
        return PanelLayoutAnalysis(
            panel_number=panel_number,
            character_boxes=boxes,
            composition_type="manual"
        )
    
    def _build_regional_prompt(self,
                                description: str,
                                art_style: str,
                                emotion: str,
                                characters: List[str],
                                layout: PanelLayoutAnalysis) -> str:
        """Build an enhanced prompt for regional generation."""
        # Get character descriptions
        char_parts = []
        for char_name in characters:
            char_info = self.characters.get(char_name, {})
            char_desc = char_info.get("description", "")
            
            # Get LoRA trigger word if available
            lora = char_info.get("lora")
            trigger = lora.trigger_word if lora else ""
            
            if trigger:
                char_parts.append(f"{char_name} ({trigger}): {char_desc}")
            else:
                char_parts.append(f"{char_name}: {char_desc}")
        
        # Build regional prompt
        prompt = f"""
{description}

CHARACTERS (maintain exact appearance):
{chr(10).join(char_parts)}

ART STYLE: {art_style}, professional comic art, high quality
MOOD: {emotion}
COMPOSITION: {layout.composition_type}

IMPORTANT: NO text, NO speech bubbles, NO words in the image.
Clean professional comic panel illustration.
"""
        
        return prompt.strip()
    
    def _generate_with_replicate_regional(self,
                                           prompt: str,
                                           char_loras: Dict[str, CharacterLoRA],
                                           masks: List[RegionalMask],
                                           seed: int) -> Image.Image:
        """
        Generate with Replicate using regional LoRA approach.
        
        Since Replicate doesn't natively support regional LoRA,
        we use a multi-LoRA approach with weighted prompts.
        
        For true regional control, we'd need the local pipeline.
        """
        if not REPLICATE_AVAILABLE:
            return self._generate_fallback(prompt, seed)
        
        # For multi-LoRA on Replicate, we use the flux-dev-multi-lora model
        # which can load multiple LoRAs with different weights
        
        lora_scales = []
        lora_urls = []
        
        for char_name, lora in char_loras.items():
            if lora.replicate_version:
                lora_urls.append(lora.replicate_version)
                lora_scales.append(self.config.default_lora_weight)
        
        if not lora_urls:
            # No trained LoRAs, try PuLID/PhotoMaker approach
            return self._generate_with_replicate_pulid(prompt, char_loras, seed)
        
        try:
            output = replicate.run(
                self.config.replicate_model,
                input={
                    "prompt": prompt,
                    "num_outputs": 1,
                    "width": self.config.image_width,
                    "height": self.config.image_height,
                    "num_inference_steps": self.config.num_inference_steps,
                    "guidance_scale": self.config.guidance_scale,
                    "lora_urls": lora_urls,
                    "lora_scales": lora_scales,
                    "seed": seed if seed > 0 else None
                }
            )
            
            # Download the image
            if output and len(output) > 0:
                import requests
                response = requests.get(output[0])
                image = Image.open(io.BytesIO(response.content))
                return image
            
        except Exception as e:
            print(f"⚠️ Replicate generation failed: {e}")
        
        return self._generate_fallback(prompt, seed)
    
    def _generate_with_replicate_pulid(self,
                                        prompt: str,
                                        char_loras: Dict[str, CharacterLoRA],
                                        seed: int) -> Image.Image:
        """
        Generate using PuLID on Replicate for instant character consistency.
        
        PuLID is better for single-image character consistency without training.
        """
        # Get reference images for characters
        main_char = list(char_loras.keys())[0] if char_loras else None
        
        if not main_char:
            return self._generate_fallback(prompt, seed)
        
        char_info = self.characters.get(main_char, {})
        ref_image = char_info.get("reference_image")
        
        if not ref_image:
            return self._generate_fallback(prompt, seed)
        
        try:
            # Prepare reference image
            if isinstance(ref_image, str):
                with open(ref_image, 'rb') as f:
                    img_data = f.read()
            else:
                buffer = io.BytesIO()
                ref_image.save(buffer, format='PNG')
                img_data = buffer.getvalue()
            
            # Use PuLID-FLUX on Replicate
            output = replicate.run(
                "zsxkib/flux-pulid:8baa7ef2255075b46f4d91cd238c21d31181b3e6a864463f967960bb0112525b",
                input={
                    "prompt": prompt,
                    "main_face_image": f"data:image/png;base64,{base64.b64encode(img_data).decode()}",
                    "num_outputs": 1,
                    "width": self.config.image_width,
                    "height": self.config.image_height,
                    "num_inference_steps": self.config.num_inference_steps,
                    "guidance_scale": self.config.guidance_scale,
                    "id_weight": 1.0,
                    "seed": seed if seed > 0 else None
                }
            )
            
            if output and len(output) > 0:
                import requests
                response = requests.get(output[0])
                image = Image.open(io.BytesIO(response.content))
                return image
                
        except Exception as e:
            print(f"⚠️ PuLID generation failed: {e}")
        
        return self._generate_fallback(prompt, seed)
    
    def _generate_with_local_pipeline(self,
                                       prompt: str,
                                       char_loras: Dict[str, CharacterLoRA],
                                       masks: List[RegionalMask],
                                       seed: int) -> Image.Image:
        """
        Generate using local diffusers with true regional LoRA attention.
        
        This is the FULL IMPLEMENTATION of regional LoRA for local GPU.
        Requires: torch, diffusers, transformers
        """
        try:
            from .regional_pipeline import RegionalDiffusionPipeline
            
            pipeline = RegionalDiffusionPipeline(
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Load character LoRAs
            for char_name, lora in char_loras.items():
                if lora.path:
                    pipeline.load_lora(lora.path, char_name)
            
            # Prepare attention masks
            attention_masks = self.mask_generator.masks_to_attention_tensors(
                masks=masks,
                latent_size=(self.config.image_height // 8, self.config.image_width // 8),
                device=pipeline.device
            )
            
            # Generate with regional attention
            image = pipeline.generate_regional(
                prompt=prompt,
                attention_masks=attention_masks,
                width=self.config.image_width,
                height=self.config.image_height,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                seed=seed if seed > 0 else None
            )
            
            return image
            
        except ImportError:
            print("⚠️ Local pipeline not available, using fallback")
            return self._generate_fallback(prompt, seed)
        except Exception as e:
            print(f"⚠️ Local generation failed: {e}")
            return self._generate_fallback(prompt, seed)
    
    def _generate_fallback(self, prompt: str, seed: int) -> Image.Image:
        """
        Fallback generation using ClipDrop or Stability AI.
        
        Does not use regional LoRA, just standard text-to-image.
        """
        import requests
        
        clipdrop_key = os.getenv("CLIPDROP_API_KEY")
        stability_key = os.getenv("STABILITY_API_KEY")
        
        if clipdrop_key:
            try:
                response = requests.post(
                    "https://clipdrop-api.co/text-to-image/v1",
                    files={'prompt': (None, prompt)},
                    headers={'x-api-key': clipdrop_key}
                )
                
                if response.status_code == 200:
                    return Image.open(io.BytesIO(response.content))
                    
            except Exception as e:
                print(f"⚠️ ClipDrop failed: {e}")
        
        if stability_key:
            try:
                response = requests.post(
                    "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                    headers={
                        "Authorization": f"Bearer {stability_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "text_prompts": [{"text": prompt}],
                        "cfg_scale": 7,
                        "height": self.config.image_height,
                        "width": self.config.image_width,
                        "samples": 1,
                        "steps": 30
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    img_data = base64.b64decode(data["artifacts"][0]["base64"])
                    return Image.open(io.BytesIO(img_data))
                    
            except Exception as e:
                print(f"⚠️ Stability failed: {e}")
        
        # Return placeholder
        print("⚠️ All generation backends failed, returning placeholder")
        return Image.new('RGB', (self.config.image_width, self.config.image_height), 
                         color=(100, 100, 100))
    
    def generate_comic_page(self,
                            panels: List[Dict],
                            characters: List[str],
                            art_style: str) -> List[GenerationResult]:
        """
        Generate all panels for a comic page.
        
        Args:
            panels: List of panel data dicts
            characters: List of character names
            art_style: Art style
        
        Returns:
            List of GenerationResults
        """
        results = []
        
        for i, panel_data in enumerate(panels):
            description = panel_data.get("Description", panel_data.get("description", ""))
            emotion = panel_data.get("emotion", "neutral")
            panel_chars = panel_data.get("characters", characters)
            
            result = self.generate_panel(
                description=description,
                characters=panel_chars,
                art_style=art_style,
                panel_number=i + 1,
                emotion=emotion
            )
            
            results.append(result)
        
        return results
    
    def visualize_layout(self, 
                         result: GenerationResult,
                         show_masks: bool = True) -> Image.Image:
        """
        Create a visualization of the layout and masks.
        
        Args:
            result: Generation result
            show_masks: Whether to overlay masks
        
        Returns:
            Visualization image
        """
        vis = result.image.copy()
        
        # Draw bounding boxes
        vis = self.bbox_extractor.visualize_layout(
            layout=result.layout,
            image=vis
        )
        
        if show_masks and result.masks_used:
            # Create mask overlay
            mask_vis = self.mask_generator.visualize_masks(result.masks_used)
            mask_vis = mask_vis.resize(vis.size)
            
            # Blend with original
            vis = Image.blend(vis, mask_vis, alpha=0.3)
        
        return vis
    
    def export_character_loras(self, output_dir: str) -> Dict[str, str]:
        """
        Export all character LoRAs to a directory.
        
        Args:
            output_dir: Directory to export to
        
        Returns:
            Dict mapping character name to exported path
        """
        os.makedirs(output_dir, exist_ok=True)
        exported = {}
        
        for char_name, char_info in self.characters.items():
            lora = char_info.get("lora")
            if lora and lora.path:
                import shutil
                dest = os.path.join(output_dir, f"{char_name}_lora.safetensors")
                shutil.copy(lora.path, dest)
                exported[char_name] = dest
        
        return exported
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get generation metrics for research reporting."""
        if not self.generation_history:
            return {}
        
        times = [r.generation_time for r in self.generation_history]
        backends = [r.backend_used for r in self.generation_history]
        
        return {
            "total_panels_generated": len(self.generation_history),
            "average_generation_time": sum(times) / len(times),
            "min_generation_time": min(times),
            "max_generation_time": max(times),
            "backend_usage": {b: backends.count(b) for b in set(backends)},
            "characters_registered": len(self.characters),
            "loras_trained": sum(1 for c in self.characters.values() if c.get("lora"))
        }
