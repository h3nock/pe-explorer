"""
Base Evaluator

Provides common utilities for all evaluation tasks:
- Model loading from checkpoints
- Device handling
- Result logging
"""

import json
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from src.model.config import ModelConfig
from src.model.transformer import Transformer


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = device
        self.dtype = dtype
        self.model = None
        self.config = None
        
    def load_model(self) -> Transformer:
        """Load model from checkpoint."""
        print(f"Loading checkpoint from {self.checkpoint_path}...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        
        # Try to get config from checkpoint first
        if "config" in checkpoint:
            config_dict = checkpoint["config"]
            self.config = ModelConfig(**config_dict)
        else:
            # Infer from checkpoint directory name: e.g. tiny_rope_s42 -> size=tiny, pe=rope
            ckpt_dir_name = self.checkpoint_path.parent.name
            self.config = self._infer_config_from_dirname(ckpt_dir_name)
        
        # Create model
        self.model = Transformer(self.config)
        
        # Load weights
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Checkpoint does not contain model weights")
        
        # Handle DDP prefix if present
        state_dict = {
            k.replace("module.", ""): v 
            for k, v in state_dict.items()
        }
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        
        print(f"Loaded model: {self.config.pe_type} PE, {sum(p.numel() for p in self.model.parameters()):,} params")
        return self.model
    
    def _infer_config_from_dirname(self, dirname: str) -> ModelConfig:
        """Infer model config from checkpoint directory name.
        
        Expected format: {size}_{pe_type}_s{seed} or {size}_{pe_type}_{variant}_s{seed}
        Examples: tiny_rope_s42, tiny_decimal_norm_s42, small_sinusoidal_s42
        """
        # Parse directory name
        parts = dirname.split("_")
        if len(parts) < 3:
            raise ValueError(
                f"Cannot infer config from '{dirname}'. Expected format: size_petype_s42. "
                f"Provide --config explicitly."
            )
        
        size = parts[0]  # tiny, small, medium, large
        
        # pe_type might be 1 or 2 parts (e.g. "rope" or "decimal_norm")
        # seed part always starts with 's', find it
        seed_idx = next((i for i, p in enumerate(parts) if p.startswith("s") and p[1:].isdigit()), -1)
        if seed_idx == -1:
            seed_idx = len(parts)
        
        pe_parts = parts[1:seed_idx]
        pe_type = "_".join(pe_parts)  # e.g. "rope", "decimal_norm", "sinusoidal"
        
        # Normalize pe_type variations
        pe_type_map = {
            "sinonly": "sinusoidal",  # map variant names
            "decimal_norm": "decimal",
        }
        base_pe_type = pe_type_map.get(pe_type, pe_type)
        
        # Load size config from configs/config.yaml
        config_candidates = [
            self.checkpoint_path.parent.parent / "configs" / "config.yaml",
            Path("configs/config.yaml"),
            self.checkpoint_path.parent / "config.yaml",
        ]
        
        if self.config_path:
            config_candidates.insert(0, self.config_path)
        
        for cfg_path in config_candidates:
            if cfg_path.exists():
                with open(cfg_path) as f:
                    full_config = yaml.safe_load(f)
                break
        else:
            raise ValueError(f"No config.yaml found. Tried: {config_candidates}")
        
        if size not in full_config:
            raise ValueError(f"Size '{size}' not found in config. Available: {list(full_config.keys())}")
        
        size_config = full_config[size]
        
        # Build ModelConfig with inferred pe_type
        # Handle pe_params for special variants
        pe_params = {}
        if pe_type == "decimal_norm":
            pe_params = {"normalize": True}
        elif pe_type == "sinonly":
            pe_params = {"sin_only": True}
        
        print(f"Inferred config: size={size}, pe_type={base_pe_type}, pe_params={pe_params}")
        
        return ModelConfig(
            d_model=size_config["d_model"],
            n_layers=size_config["n_layers"],
            n_heads=size_config["n_heads"],
            d_ff=size_config["d_ff"],
            max_seq_len=size_config["max_seq_len"],
            pe_type=base_pe_type,
            pe_params=pe_params,
        )
    
    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """Run evaluation. Returns dict of metrics."""
        pass
    
    def log_results(
        self,
        results: dict[str, Any],
        output_path: str | Path | None = None,
        wandb_run = None,
    ) -> None:
        """Log results to console, file, and optionally WandB."""
        # Console output
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 60)
        
        # Save to JSON
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_path}")
        
        # Log to WandB
        if wandb_run:
            wandb_run.log(results)
            print("Results logged to WandB")


def load_tokenizer():
    """Load the tokenizer (tiktoken GPT-2)."""
    import tiktoken
    return tiktoken.get_encoding("gpt2")
