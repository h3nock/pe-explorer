import json
import torch
import wandb
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.model.transformer import Transformer
from src.data.tokenization import Tokenizer
from src.model.config import ModelConfig


@dataclass
class EvalConfig:
    """Centralized configuration for evaluation."""
    checkpoint_path: Path
    device: str = "cuda"
    output_dir: Path = field(default_factory=lambda: Path("eval_results"))
    num_samples: int = 100
    seed: int = 42
    batch_size: int = 4

    # algorithmic task configs 
    context_len: int = 1024
    eval_context_len: int = 2048
    filler_pool_path: Optional[Path] = None
    tasks: dict = field(default_factory=dict)
    content_sampling: dict = field(default_factory=lambda: {"digit": 0.5, "char": 0.5})

    # PPL eval configs
    fineweb_val_path: Path = field(default_factory=lambda: Path("data/fineweb_bin/val"))
    max_seq_len: int = 1024

    @classmethod
    def from_yaml(cls, checkpoint_path: Path, config_path: Path = None, **overrides) -> "EvalConfig":
        """Load config from YAML files with overrides.

        Loads from both data_generation.yaml (algorithmic tasks) and
        config.yaml (model/data settings) for complete configuration.
        """
        config_path = config_path or Path("configs/data_generation.yaml")
        main_config_path = Path("configs/config.yaml")

        kwargs = {
            "checkpoint_path": Path(checkpoint_path),
        }

        # load algorithmic task config
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}

            filler_cfg = cfg.get("filler", {})
            gen_cfg = cfg.get("generation", {})
            kwargs["context_len"] = filler_cfg.get("context_len", 1024)
            kwargs["eval_context_len"] = filler_cfg.get("eval_context_len", 2048)

            pool_path = filler_cfg.get("pool_path")
            if pool_path and Path(pool_path).exists():
                kwargs["filler_pool_path"] = Path(pool_path)

            kwargs["tasks"] = cfg.get("tasks", {})
            kwargs["content_sampling"] = cfg.get("content_sampling", {"digit": 0.5, "char": 0.5})
            kwargs["num_samples"] = gen_cfg.get("num_eval_per_task", kwargs.get("num_samples", 100))
            kwargs["seed"] = gen_cfg.get("eval_seed", kwargs.get("seed", 42))

        # load main config for fineweb/model settings
        if main_config_path.exists():
            with open(main_config_path) as f:
                main_cfg = yaml.safe_load(f) or {}

            fineweb_cfg = main_cfg.get("data", {}).get("fineweb", {})
            val_path = fineweb_cfg.get("val_tokenized_path", "data/fineweb_bin/val")
            kwargs["fineweb_val_path"] = Path(val_path)

            model_cfg = main_cfg.get("model", {})
            kwargs["max_seq_len"] = model_cfg.get("max_seq_len", 1024)

        kwargs.update(overrides)
        return cls(**kwargs)


class BaseEvaluator(ABC):
    def __init__(self, checkpoint_path: str, device: str = "cuda", output_dir: str = "eval_results"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.device = self._resolve_device(device)
        self.tokenizer = Tokenizer()

        # load eval config
        self.eval_config = EvalConfig.from_yaml(
            checkpoint_path=self.checkpoint_path,
            device=device,
            output_dir=self.output_dir,
        )

        # load model
        self.model, self.config, self.metadata = self._load_model()
        self._validate_tokenizer()
        self.model.eval()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_device(self, requested: str) -> torch.device:
        """Resolve device with clear fallback logic and warnings."""
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")

        if requested in ("cuda", "mps"):
            fallback = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Warning: {requested} unavailable, using {fallback}")
            return torch.device(fallback)

        return torch.device("cpu")

    def _validate_tokenizer(self):
        """Validate tokenizer vocab matches model config."""
        if hasattr(self.config, "vocab_size"):
            tokenizer_vocab = self.tokenizer.vocab_size
            model_vocab = self.config.vocab_size
            if tokenizer_vocab != model_vocab:
                raise ValueError(
                    f"Tokenizer vocab size ({tokenizer_vocab}) != "
                    f"model vocab size ({model_vocab}). "
                    "Check that tokenizer matches the checkpoint."
                )

    def _load_model(self):
        """Load model from checkpoint with validation."""
        print(f"Loading checkpoint: {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # extract and validate config
        config_dict = checkpoint.get("config")
        if not config_dict:
            raise ValueError(
                f"Checkpoint missing 'config' key. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        config = ModelConfig(**config_dict)

        model = Transformer(config).to(self.device)

        # load weights (handle DDP prefix)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {
            k.removeprefix("module."): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(state_dict)

        metadata = checkpoint.get("run_metadata", {})
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {n_params / 1e6:.1f}M params on {self.device}")

        return model, config, metadata

    def log_results(self, task_name: str, metrics: dict):
        """Log results to local JSON and WandB."""

        result_file = self.output_dir / f"{task_name}.json"
        with open(result_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Results saved to {result_file}")

        # WandB Log (if active) with flatten nested dicts
        if wandb.run is not None:
            flat = {}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, (int, float)):
                            flat[f"eval/{task_name}/{k}/{k2}"] = v2
                elif isinstance(v, (int, float)):
                    flat[f"eval/{task_name}/{k}"] = v
            if flat:
                wandb.log(flat)

    @abstractmethod
    def run(self) -> dict:
        """Run evaluation and return results dict."""
        pass
