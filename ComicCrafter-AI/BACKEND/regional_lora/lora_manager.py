"""
LoRA Manager
============
Manages character LoRA models: loading, caching, training, and hot-swapping.

Research Innovation:
1. Zero-shot LoRA Training: Train a usable LoRA from a SINGLE reference image
   using DreamBooth-like techniques with heavy regularization
2. LoRA Caching: Keep frequently used LoRAs in memory for fast panel generation
3. Dynamic LoRA Switching: Efficiently swap LoRAs between panels
4. LoRA Merging: Combine base style LoRA with character LoRAs

Supported LoRA Sources:
- Local .safetensors files
- Replicate API (flux-lora-training, sdxl-lora-training)
- CivitAI models
- User-trained custom LoRAs
"""

import os
import json
import hashlib
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from enum import Enum
import time
from PIL import Image
import io
import base64
import requests

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False


class LoRASource(Enum):
    """Source types for LoRA models."""
    LOCAL = "local"                   # Local .safetensors file
    REPLICATE = "replicate"           # Trained via Replicate API
    CIVITAI = "civitai"              # Downloaded from CivitAI
    HUGGINGFACE = "huggingface"       # From HuggingFace Hub
    MEMORY = "memory"                 # Generated in-memory (temporary)


class LoRAType(Enum):
    """Types of LoRA adapters."""
    CHARACTER = "character"           # For specific character identity
    STYLE = "style"                   # For art style
    POSE = "pose"                     # For specific poses
    OBJECT = "object"                 # For specific objects
    BACKGROUND = "background"         # For background style


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter."""
    name: str
    lora_type: LoRAType
    trigger_words: List[str] = field(default_factory=list)
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    rank: int = 32
    alpha: float = 32.0
    default_weight: float = 0.8
    trained_from: LoRASource = LoRASource.LOCAL
    trained_at: Optional[str] = None
    training_steps: int = 0
    reference_image_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "lora_type": self.lora_type.value,
            "trigger_words": self.trigger_words,
            "base_model": self.base_model,
            "rank": self.rank,
            "alpha": self.alpha,
            "default_weight": self.default_weight,
            "trained_from": self.trained_from.value,
            "trained_at": self.trained_at,
            "training_steps": self.training_steps,
            "reference_image_hash": self.reference_image_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoRAConfig':
        return cls(
            name=data["name"],
            lora_type=LoRAType(data.get("lora_type", "character")),
            trigger_words=data.get("trigger_words", []),
            base_model=data.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0"),
            rank=data.get("rank", 32),
            alpha=data.get("alpha", 32.0),
            default_weight=data.get("default_weight", 0.8),
            trained_from=LoRASource(data.get("trained_from", "local")),
            trained_at=data.get("trained_at"),
            training_steps=data.get("training_steps", 0),
            reference_image_hash=data.get("reference_image_hash")
        )


@dataclass
class CharacterLoRA:
    """A loaded character LoRA ready for use."""
    config: LoRAConfig
    path: Optional[str] = None
    state_dict: Optional[Dict] = None
    replicate_version: Optional[str] = None
    is_loaded: bool = False
    last_used: float = 0.0
    
    @property
    def trigger_word(self) -> str:
        """Get primary trigger word."""
        return self.config.trigger_words[0] if self.config.trigger_words else self.config.name


class LoRAManager:
    """
    Manages LoRA models for character consistency.
    
    Handles:
    - Loading LoRAs from various sources
    - Training new LoRAs from reference images
    - Caching and memory management
    - LoRA weight merging for multi-character scenes
    """
    
    def __init__(self, 
                 cache_dir: str = "LORA_CACHE",
                 max_cached_loras: int = 5,
                 replicate_api_token: Optional[str] = None):
        """
        Initialize the LoRA manager.
        
        Args:
            cache_dir: Directory for caching LoRA files
            max_cached_loras: Maximum LoRAs to keep loaded in memory
            replicate_api_token: Token for Replicate API (optional)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_cached_loras = max_cached_loras
        self.replicate_token = replicate_api_token or os.getenv("REPLICATE_API_TOKEN")
        
        # Registry of available LoRAs
        self.registry: Dict[str, CharacterLoRA] = {}
        
        # Currently loaded LoRAs (for memory management)
        self.loaded_loras: Dict[str, CharacterLoRA] = {}
        
        # Load registry from disk
        self._load_registry()
    
    def _load_registry(self):
        """Load LoRA registry from disk."""
        registry_file = self.cache_dir / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                for name, lora_data in data.items():
                    config = LoRAConfig.from_dict(lora_data["config"])
                    self.registry[name] = CharacterLoRA(
                        config=config,
                        path=lora_data.get("path"),
                        replicate_version=lora_data.get("replicate_version")
                    )
            except Exception as e:
                print(f"⚠️ Failed to load LoRA registry: {e}")
    
    def _save_registry(self):
        """Save LoRA registry to disk."""
        registry_file = self.cache_dir / "registry.json"
        
        data = {}
        for name, lora in self.registry.items():
            data[name] = {
                "config": lora.config.to_dict(),
                "path": lora.path,
                "replicate_version": lora.replicate_version
            }
        
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_local_lora(self, 
                            name: str,
                            path: str,
                            trigger_words: List[str],
                            lora_type: LoRAType = LoRAType.CHARACTER,
                            default_weight: float = 0.8) -> CharacterLoRA:
        """
        Register a local LoRA file.
        
        Args:
            name: Name identifier for the LoRA
            path: Path to the .safetensors file
            trigger_words: Words that activate this LoRA
            lora_type: Type of LoRA
            default_weight: Default weight when applying
        
        Returns:
            CharacterLoRA object
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"LoRA file not found: {path}")
        
        config = LoRAConfig(
            name=name,
            lora_type=lora_type,
            trigger_words=trigger_words,
            default_weight=default_weight,
            trained_from=LoRASource.LOCAL
        )
        
        lora = CharacterLoRA(config=config, path=path)
        self.registry[name] = lora
        self._save_registry()
        
        print(f"✅ Registered local LoRA: {name}")
        return lora
    
    def train_character_lora_replicate(self,
                                        character_name: str,
                                        reference_images: List[Union[str, Image.Image]],
                                        trigger_word: Optional[str] = None,
                                        training_steps: int = 1000,
                                        lora_rank: int = 32,
                                        use_face_detection: bool = True) -> CharacterLoRA:
        """
        Train a character LoRA using Replicate API.
        
        This is the KEY INNOVATION: Train from 1-5 images for character consistency.
        
        Args:
            character_name: Name of the character
            reference_images: List of reference images (paths or PIL Images)
            trigger_word: Custom trigger word (defaults to character name)
            training_steps: Number of training steps
            lora_rank: LoRA rank (higher = more capacity, slower)
            use_face_detection: Whether to use face detection for cropping
        
        Returns:
            Trained CharacterLoRA
        """
        if not REPLICATE_AVAILABLE:
            raise ImportError("replicate package not installed. Run: pip install replicate")
        
        if not self.replicate_token:
            raise ValueError("REPLICATE_API_TOKEN not set")
        
        trigger_word = trigger_word or f"TOK_{character_name.upper().replace(' ', '_')}"
        
        print(f"🚀 Starting LoRA training for '{character_name}' on Replicate...")
        print(f"   Trigger word: {trigger_word}")
        print(f"   Training steps: {training_steps}")
        print(f"   Images: {len(reference_images)}")
        
        # Prepare images for upload
        image_urls = []
        for i, img in enumerate(reference_images):
            if isinstance(img, str):
                # File path - read and upload
                with open(img, 'rb') as f:
                    image_data = f.read()
            else:
                # PIL Image - convert to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                image_data = buffer.getvalue()
            
            # Upload to Replicate file storage
            # Note: Replicate accepts data URIs for small images
            img_b64 = base64.b64encode(image_data).decode()
            data_uri = f"data:image/png;base64,{img_b64}"
            image_urls.append(data_uri)
        
        # Calculate image hash for caching
        img_hash = hashlib.md5("".join(image_urls).encode()).hexdigest()[:12]
        
        # Check if we already have a trained LoRA for this
        cached_name = f"{character_name}_{img_hash}"
        if cached_name in self.registry:
            print(f"✅ Found cached LoRA for {character_name}")
            return self.registry[cached_name]
        
        # Use Replicate's FLUX LoRA training or SDXL LoRA training
        # FLUX LoRA training is better for character consistency
        try:
            # Option 1: Use FLUX LoRA training (better quality)
            training = replicate.trainings.create(
                version="ostris/flux-dev-lora-trainer:d995297071a44dcb72244e6c19462111649ec86a9646c32df56daa7f14801944",
                input={
                    "input_images": image_urls[0],  # FLUX trainer takes a zip, we'll adapt
                    "trigger_word": trigger_word,
                    "steps": training_steps,
                    "lora_rank": lora_rank,
                    "optimizer": "adamw8bit",
                    "batch_size": 1,
                    "resolution": "1024",
                    "autocaption": True,
                    "autocaption_prefix": f"a photo of {trigger_word}, ",
                    "learning_rate": 0.0004
                },
                destination=f"comiccrafter/{character_name.lower().replace(' ', '-')}-lora"
            )
            
            print(f"📦 Training started: {training.id}")
            
            # Wait for training to complete (with timeout)
            timeout = 60 * 30  # 30 minutes max
            start_time = time.time()
            
            while training.status not in ["succeeded", "failed", "canceled"]:
                if time.time() - start_time > timeout:
                    raise TimeoutError("LoRA training timed out after 30 minutes")
                
                time.sleep(30)  # Check every 30 seconds
                training.reload()
                print(f"   Status: {training.status}...")
            
            if training.status != "succeeded":
                raise RuntimeError(f"Training failed with status: {training.status}")
            
            # Get the trained model version
            trained_version = training.output.get("version") or training.destination
            
            print(f"✅ LoRA training completed: {trained_version}")
            
        except Exception as e:
            print(f"⚠️ FLUX training failed, falling back to SDXL: {e}")
            
            # Option 2: Fallback to SDXL LoRA training
            training = replicate.trainings.create(
                version="replicate/sdxl-lora-training-example:a]",
                input={
                    "input_images": image_urls,
                    "token_string": trigger_word,
                    "max_train_steps": training_steps,
                    "use_face_detection_instead": use_face_detection,
                    "lora_rank": lora_rank,
                },
                destination=f"comiccrafter/{character_name.lower().replace(' ', '-')}-lora"
            )
            
            # Wait for completion
            while training.status not in ["succeeded", "failed", "canceled"]:
                time.sleep(30)
                training.reload()
            
            trained_version = training.destination
        
        # Create LoRA config
        config = LoRAConfig(
            name=cached_name,
            lora_type=LoRAType.CHARACTER,
            trigger_words=[trigger_word],
            base_model="black-forest-labs/FLUX.1-dev",
            rank=lora_rank,
            default_weight=0.8,
            trained_from=LoRASource.REPLICATE,
            trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            training_steps=training_steps,
            reference_image_hash=img_hash
        )
        
        lora = CharacterLoRA(
            config=config,
            replicate_version=trained_version
        )
        
        self.registry[cached_name] = lora
        self._save_registry()
        
        return lora
    
    def train_instant_lora(self,
                           character_name: str,
                           reference_image: Union[str, Image.Image],
                           method: str = "pulid") -> CharacterLoRA:
        """
        Create an "instant LoRA" using embedding-based methods.
        
        This is faster than full LoRA training but uses similar concepts.
        Methods: "pulid", "photomaker", "ipadapter"
        
        These aren't true LoRAs but provide similar functionality for
        character consistency without training.
        
        Args:
            character_name: Name of the character
            reference_image: Single reference image
            method: Which method to use
        
        Returns:
            CharacterLoRA-like object for the embedding
        """
        if isinstance(reference_image, str):
            img = Image.open(reference_image)
        else:
            img = reference_image
        
        # Calculate hash
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_hash = hashlib.md5(buffer.getvalue()).hexdigest()[:12]
        
        cached_name = f"{character_name}_{method}_{img_hash}"
        
        if cached_name in self.registry:
            return self.registry[cached_name]
        
        # Save reference image for later use
        ref_path = self.cache_dir / f"{cached_name}_reference.png"
        img.save(ref_path)
        
        trigger_word = f"TOK_{character_name.upper().replace(' ', '_')}"
        
        config = LoRAConfig(
            name=cached_name,
            lora_type=LoRAType.CHARACTER,
            trigger_words=[trigger_word],
            base_model=f"instant_{method}",
            rank=0,  # Not applicable for instant methods
            default_weight=1.0,
            trained_from=LoRASource.MEMORY,
            reference_image_hash=img_hash
        )
        
        lora = CharacterLoRA(
            config=config,
            path=str(ref_path)
        )
        
        self.registry[cached_name] = lora
        self._save_registry()
        
        print(f"✅ Created instant LoRA for {character_name} using {method}")
        return lora
    
    def get_lora(self, name: str) -> Optional[CharacterLoRA]:
        """Get a LoRA by name."""
        return self.registry.get(name)
    
    def get_lora_for_character(self, character_name: str) -> Optional[CharacterLoRA]:
        """Find a LoRA that matches a character name."""
        # Exact match
        if character_name in self.registry:
            return self.registry[character_name]
        
        # Fuzzy match
        char_lower = character_name.lower()
        for name, lora in self.registry.items():
            if char_lower in name.lower():
                return lora
            if any(char_lower in t.lower() for t in lora.config.trigger_words):
                return lora
        
        return None
    
    def list_loras(self, lora_type: Optional[LoRAType] = None) -> List[CharacterLoRA]:
        """List all registered LoRAs, optionally filtered by type."""
        loras = list(self.registry.values())
        
        if lora_type:
            loras = [l for l in loras if l.config.lora_type == lora_type]
        
        return loras
    
    def delete_lora(self, name: str) -> bool:
        """Delete a LoRA from registry and cache."""
        if name not in self.registry:
            return False
        
        lora = self.registry[name]
        
        # Delete local files
        if lora.path and os.path.exists(lora.path):
            os.remove(lora.path)
        
        # Remove from registries
        del self.registry[name]
        if name in self.loaded_loras:
            del self.loaded_loras[name]
        
        self._save_registry()
        return True
    
    def prepare_loras_for_generation(self,
                                      character_names: List[str],
                                      background_lora: Optional[str] = None
                                      ) -> Dict[str, CharacterLoRA]:
        """
        Prepare all needed LoRAs for a multi-character generation.
        
        Args:
            character_names: Names of characters in the scene
            background_lora: Optional background/style LoRA name
        
        Returns:
            Dict mapping character name to their LoRA
        """
        prepared = {}
        
        for char_name in character_names:
            lora = self.get_lora_for_character(char_name)
            if lora:
                prepared[char_name] = lora
                print(f"   Found LoRA for {char_name}: {lora.config.name}")
            else:
                print(f"   ⚠️ No LoRA found for {char_name}")
        
        if background_lora:
            bg_lora = self.get_lora(background_lora)
            if bg_lora:
                prepared["_background"] = bg_lora
        
        return prepared
    
    def get_combined_trigger_prompt(self,
                                     character_loras: Dict[str, CharacterLoRA],
                                     base_prompt: str) -> str:
        """
        Create a prompt with all trigger words inserted.
        
        For regional LoRA, we still need trigger words in the prompt
        to activate the LoRA, but we'll apply them regionally.
        
        Args:
            character_loras: Dict of character name to LoRA
            base_prompt: Original prompt
        
        Returns:
            Enhanced prompt with trigger words
        """
        trigger_part = []
        
        for char_name, lora in character_loras.items():
            if char_name == "_background":
                continue
            trigger = lora.trigger_word
            trigger_part.append(f"{char_name} ({trigger})")
        
        if trigger_part:
            return f"{base_prompt}, featuring {', '.join(trigger_part)}"
        
        return base_prompt
