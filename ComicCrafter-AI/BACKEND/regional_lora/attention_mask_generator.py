"""
Attention Mask Generator
========================
Creates gradient masks for regional LoRA application during diffusion.

Research Innovation:
Instead of hard binary masks that cause visible seams, we use:
1. Gaussian gradient edges for smooth blending
2. Overlap handling with weighted averaging
3. Depth-aware masking (z-order support)
4. Latent-space aware resizing (matches diffusion spatial dims)

The masks are applied during the cross-attention phase to control
which LoRA adapter influences which spatial regions.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MaskType(Enum):
    """Types of attention masks."""
    BINARY = "binary"           # Hard 0/1 mask
    GAUSSIAN = "gaussian"       # Soft gaussian edges
    COSINE = "cosine"          # Cosine falloff at edges
    LINEAR = "linear"          # Linear gradient at edges
    ADAPTIVE = "adaptive"       # Adapts based on overlap


@dataclass
class RegionalMask:
    """
    A regional mask for LoRA application.
    
    Attributes:
        character_name: Which character this mask is for
        mask: The actual mask array (HxW, values 0-1)
        lora_id: Identifier for the LoRA to apply
        weight: Overall weight for this LoRA (0-1)
        z_order: Depth order for overlap resolution
    """
    character_name: str
    mask: np.ndarray
    lora_id: str
    weight: float = 1.0
    z_order: int = 0
    
    def to_torch(self, device: str = "cuda"):
        """Convert mask to PyTorch tensor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        return torch.from_numpy(self.mask).float().to(device)
    
    def resize_for_latent(self, latent_height: int, latent_width: int) -> 'RegionalMask':
        """
        Resize mask to match latent space dimensions.
        
        SD latent space is typically 1/8 of image size.
        """
        from PIL import Image
        
        # Convert to PIL for smooth resizing
        mask_pil = Image.fromarray((self.mask * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((latent_width, latent_height), Image.BILINEAR)
        new_mask = np.array(mask_resized).astype(np.float32) / 255.0
        
        return RegionalMask(
            character_name=self.character_name,
            mask=new_mask,
            lora_id=self.lora_id,
            weight=self.weight,
            z_order=self.z_order
        )


class AttentionMaskGenerator:
    """
    Generates attention masks for regional LoRA application.
    
    The masks control how much each LoRA adapter contributes to
    different spatial regions during cross-attention computation.
    """
    
    def __init__(self, 
                 default_mask_type: MaskType = MaskType.GAUSSIAN,
                 edge_softness: float = 0.1,
                 min_mask_value: float = 0.0):
        """
        Initialize the mask generator.
        
        Args:
            default_mask_type: Default type of mask to generate
            edge_softness: How soft the edges should be (0=hard, 1=very soft)
            min_mask_value: Minimum mask value (can help with stability)
        """
        self.default_mask_type = default_mask_type
        self.edge_softness = edge_softness
        self.min_mask_value = min_mask_value
    
    def generate_mask(self,
                      bbox: Tuple[float, float, float, float],
                      image_size: Tuple[int, int],
                      mask_type: Optional[MaskType] = None,
                      character_name: str = "unknown",
                      lora_id: str = "",
                      z_order: int = 0) -> RegionalMask:
        """
        Generate a mask for a single bounding box.
        
        Args:
            bbox: Normalized bounding box (x1, y1, x2, y2) in 0-1 range
            image_size: (width, height) of target image
            mask_type: Type of mask to generate
            character_name: Name for the mask
            lora_id: LoRA identifier to associate
            z_order: Depth order
        
        Returns:
            RegionalMask with generated mask array
        """
        mask_type = mask_type or self.default_mask_type
        width, height = image_size
        
        # Create coordinate grids
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        
        x1, y1, x2, y2 = bbox
        
        if mask_type == MaskType.BINARY:
            mask = self._generate_binary_mask(xx, yy, x1, y1, x2, y2)
        elif mask_type == MaskType.GAUSSIAN:
            mask = self._generate_gaussian_mask(xx, yy, x1, y1, x2, y2)
        elif mask_type == MaskType.COSINE:
            mask = self._generate_cosine_mask(xx, yy, x1, y1, x2, y2)
        elif mask_type == MaskType.LINEAR:
            mask = self._generate_linear_mask(xx, yy, x1, y1, x2, y2)
        else:
            mask = self._generate_gaussian_mask(xx, yy, x1, y1, x2, y2)
        
        # Apply minimum value
        mask = np.maximum(mask, self.min_mask_value)
        
        return RegionalMask(
            character_name=character_name,
            mask=mask.astype(np.float32),
            lora_id=lora_id,
            z_order=z_order
        )
    
    def _generate_binary_mask(self, xx: np.ndarray, yy: np.ndarray,
                               x1: float, y1: float, 
                               x2: float, y2: float) -> np.ndarray:
        """Generate a hard binary mask."""
        mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).astype(np.float32)
        return mask
    
    def _generate_gaussian_mask(self, xx: np.ndarray, yy: np.ndarray,
                                 x1: float, y1: float,
                                 x2: float, y2: float) -> np.ndarray:
        """Generate a mask with Gaussian-blurred edges."""
        # First create binary mask
        binary = self._generate_binary_mask(xx, yy, x1, y1, x2, y2)
        
        # Calculate edge softness in pixels
        edge_pixels = int(self.edge_softness * min(xx.shape))
        edge_pixels = max(1, edge_pixels)
        
        # Apply Gaussian blur using scipy
        try:
            from scipy.ndimage import gaussian_filter
            mask = gaussian_filter(binary, sigma=edge_pixels)
        except ImportError:
            # Fallback: approximate with distance-based falloff
            mask = self._approximate_gaussian_falloff(xx, yy, x1, y1, x2, y2)
        
        return mask
    
    def _generate_cosine_mask(self, xx: np.ndarray, yy: np.ndarray,
                               x1: float, y1: float,
                               x2: float, y2: float) -> np.ndarray:
        """Generate a mask with cosine falloff at edges."""
        # Distance from nearest edge
        margin = self.edge_softness
        
        # Calculate smooth transitions for each edge
        def smooth_step(x, edge, margin, inside=True):
            if inside:
                d = (x - edge) / margin
            else:
                d = (edge - x) / margin
            d = np.clip(d, 0, 1)
            return 0.5 - 0.5 * np.cos(np.pi * d)
        
        left = smooth_step(xx, x1, margin, inside=True)
        right = smooth_step(xx, x2, margin, inside=False)
        top = smooth_step(yy, y1, margin, inside=True)
        bottom = smooth_step(yy, y2, margin, inside=False)
        
        mask = left * right * top * bottom
        return mask
    
    def _generate_linear_mask(self, xx: np.ndarray, yy: np.ndarray,
                               x1: float, y1: float,
                               x2: float, y2: float) -> np.ndarray:
        """Generate a mask with linear gradient edges."""
        margin = self.edge_softness
        
        def linear_ramp(x, edge, margin, inside=True):
            if inside:
                d = (x - edge) / margin
            else:
                d = (edge - x) / margin
            return np.clip(d, 0, 1)
        
        left = linear_ramp(xx, x1, margin, inside=True)
        right = linear_ramp(xx, x2, margin, inside=False)
        top = linear_ramp(yy, y1, margin, inside=True)
        bottom = linear_ramp(yy, y2, margin, inside=False)
        
        mask = np.minimum(left, right) * np.minimum(top, bottom)
        return mask
    
    def _approximate_gaussian_falloff(self, xx: np.ndarray, yy: np.ndarray,
                                       x1: float, y1: float,
                                       x2: float, y2: float) -> np.ndarray:
        """Approximate Gaussian falloff without scipy."""
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        sx = (x2 - x1) / 2 + self.edge_softness
        sy = (y2 - y1) / 2 + self.edge_softness
        
        # Gaussian-like falloff
        dist_x = ((xx - cx) / sx) ** 2
        dist_y = ((yy - cy) / sy) ** 2
        
        # Use rectangular distance (product of 1D Gaussians)
        mask = np.exp(-0.5 * (dist_x + dist_y))
        
        # Normalize to have 1.0 inside the box
        mask = mask / mask.max()
        
        return mask
    
    def generate_multi_character_masks(self,
                                        character_boxes: List[Dict],
                                        image_size: Tuple[int, int],
                                        overlap_mode: str = "z_order") -> List[RegionalMask]:
        """
        Generate masks for multiple characters with overlap handling.
        
        Args:
            character_boxes: List of {bbox, character_name, lora_id, z_order}
            image_size: Target image size
            overlap_mode: How to handle overlaps:
                - "z_order": Higher z wins
                - "average": Average overlapping masks
                - "blend": Weighted blend based on mask values
        
        Returns:
            List of RegionalMasks
        """
        masks = []
        
        # Generate individual masks
        for char_data in character_boxes:
            mask = self.generate_mask(
                bbox=tuple(char_data.get("bbox", (0, 0, 1, 1))),
                image_size=image_size,
                character_name=char_data.get("character_name", "unknown"),
                lora_id=char_data.get("lora_id", ""),
                z_order=char_data.get("z_order", 0)
            )
            masks.append(mask)
        
        # Handle overlaps
        if overlap_mode == "z_order":
            masks = self._resolve_overlaps_z_order(masks)
        elif overlap_mode == "average":
            masks = self._resolve_overlaps_average(masks)
        elif overlap_mode == "blend":
            masks = self._resolve_overlaps_blend(masks)
        
        return masks
    
    def _resolve_overlaps_z_order(self, masks: List[RegionalMask]) -> List[RegionalMask]:
        """Resolve overlaps using z-order (higher z-order wins)."""
        if not masks:
            return masks
        
        # Sort by z-order (lower first, will be overwritten by higher)
        sorted_masks = sorted(masks, key=lambda m: m.z_order)
        
        # Create cumulative mask to track claimed regions
        height, width = sorted_masks[0].mask.shape
        claimed = np.zeros((height, width), dtype=np.float32)
        
        result_masks = []
        
        # Process from highest z-order to lowest
        for mask in reversed(sorted_masks):
            # Reduce mask where already claimed
            adjusted_mask = mask.mask * (1.0 - claimed)
            
            # Update claimed regions
            claimed = np.maximum(claimed, mask.mask)
            
            result_masks.append(RegionalMask(
                character_name=mask.character_name,
                mask=adjusted_mask,
                lora_id=mask.lora_id,
                weight=mask.weight,
                z_order=mask.z_order
            ))
        
        return list(reversed(result_masks))
    
    def _resolve_overlaps_average(self, masks: List[RegionalMask]) -> List[RegionalMask]:
        """Resolve overlaps by averaging contributions."""
        if not masks:
            return masks
        
        # Stack all masks
        all_masks = np.stack([m.mask for m in masks], axis=0)
        
        # Sum for normalization
        total = np.sum(all_masks, axis=0, keepdims=True)
        total = np.maximum(total, 1e-8)  # Avoid division by zero
        
        # Normalize each mask
        normalized = all_masks / total
        
        result_masks = []
        for i, mask in enumerate(masks):
            result_masks.append(RegionalMask(
                character_name=mask.character_name,
                mask=normalized[i],
                lora_id=mask.lora_id,
                weight=mask.weight,
                z_order=mask.z_order
            ))
        
        return result_masks
    
    def _resolve_overlaps_blend(self, masks: List[RegionalMask]) -> List[RegionalMask]:
        """Resolve overlaps using weighted blending based on mask values."""
        # Same as average but weighted by original mask values squared
        if not masks:
            return masks
        
        all_masks = np.stack([m.mask for m in masks], axis=0)
        
        # Weight by squared values (emphasizes strong regions)
        weights = all_masks ** 2
        total_weight = np.sum(weights, axis=0, keepdims=True)
        total_weight = np.maximum(total_weight, 1e-8)
        
        normalized = weights / total_weight
        
        result_masks = []
        for i, mask in enumerate(masks):
            result_masks.append(RegionalMask(
                character_name=mask.character_name,
                mask=normalized[i],
                lora_id=mask.lora_id,
                weight=mask.weight,
                z_order=mask.z_order
            ))
        
        return result_masks
    
    def generate_background_mask(self, 
                                  character_masks: List[RegionalMask],
                                  image_size: Tuple[int, int]) -> np.ndarray:
        """
        Generate a background mask (inverse of all character masks).
        
        Useful for applying a base/background LoRA to non-character regions.
        
        Args:
            character_masks: List of character masks
            image_size: Target size
        
        Returns:
            Background mask array
        """
        width, height = image_size
        
        if not character_masks:
            return np.ones((height, width), dtype=np.float32)
        
        # Combine all character masks
        combined = np.zeros((height, width), dtype=np.float32)
        for mask in character_masks:
            # Resize mask if needed
            if mask.mask.shape != (height, width):
                from PIL import Image
                mask_pil = Image.fromarray((mask.mask * 255).astype(np.uint8))
                mask_resized = mask_pil.resize((width, height), Image.BILINEAR)
                resized_arr = np.array(mask_resized).astype(np.float32) / 255.0
                combined = np.maximum(combined, resized_arr)
            else:
                combined = np.maximum(combined, mask.mask)
        
        # Invert for background
        background = 1.0 - combined
        
        return background
    
    def visualize_masks(self, masks: List[RegionalMask],
                        background_mask: Optional[np.ndarray] = None) -> Image.Image:
        """
        Create a color visualization of all masks.
        
        Each character gets a different color, overlapping regions
        show blended colors.
        
        Args:
            masks: List of RegionalMasks to visualize
            background_mask: Optional background mask
        
        Returns:
            PIL Image visualization
        """
        if not masks:
            return Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        height, width = masks[0].mask.shape
        
        # Colors for each character
        colors = [
            np.array([255, 100, 100]),   # Red
            np.array([100, 255, 100]),   # Green
            np.array([100, 100, 255]),   # Blue
            np.array([255, 200, 100]),   # Orange
            np.array([200, 100, 255]),   # Purple
            np.array([100, 255, 255]),   # Cyan
        ]
        
        # Background color
        bg_color = np.array([50, 50, 50])
        
        # Initialize with background
        output = np.ones((height, width, 3)) * bg_color
        
        if background_mask is not None:
            output = output * background_mask[:, :, np.newaxis]
        
        # Blend in each mask
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            mask_3d = mask.mask[:, :, np.newaxis]
            output = output * (1 - mask_3d) + color * mask_3d
        
        # Convert to PIL
        output = np.clip(output, 0, 255).astype(np.uint8)
        return Image.fromarray(output)
    
    def masks_to_attention_tensors(self, 
                                    masks: List[RegionalMask],
                                    latent_size: Tuple[int, int],
                                    device: str = "cuda") -> Dict[str, 'torch.Tensor']:
        """
        Convert masks to attention tensors for use in cross-attention.
        
        Args:
            masks: List of RegionalMasks
            latent_size: (height, width) of latent space
            device: PyTorch device
        
        Returns:
            Dict mapping character_name to attention tensor
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for attention tensors")
        
        attention_tensors = {}
        
        for mask in masks:
            # Resize to latent dimensions
            resized = mask.resize_for_latent(latent_size[0], latent_size[1])
            tensor = resized.to_torch(device)
            
            # Add batch and channel dimensions: [1, 1, H, W]
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            
            attention_tensors[mask.lora_id or mask.character_name] = tensor
        
        return attention_tensors
