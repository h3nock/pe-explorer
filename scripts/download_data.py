#!/usr/bin/env python3
"""Download FineWeb-Edu dataset. 

Usage:
    python scripts/download_data.py  # download 10BT (default)
    python scripts/download_data.py --variant 100BT   # download 100BT
"""
import argparse
from pathlib import Path
from datasets import load_dataset

VARIANTS = {
    "10BT": "sample-10BT",
    "100BT": "sample-100BT",
}

def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu dataset")
    parser.add_argument(
        "--variant", 
        default="10BT", 
        choices=["10BT", "100BT"],
        help="Dataset variant to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/fineweb-edu-{variant})"
    )
    args = parser.parse_args()
    
    # default output path
    output_dir = args.output_dir or f"data/fineweb-edu-{args.variant}"
    output_path = Path(output_dir)
    
    if output_path.exists():
        print(f"Dataset already exists at {output_path}")
        return
    
    print(f"Downloading FineWeb-Edu {args.variant}...")
    print(f"This may take a while (10BT ~20GB, 100BT ~200GB)")
    
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=VARIANTS[args.variant],
        split="train",
    )
    
    print(f"Saving to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    
    print(f"Done! Dataset saved to {output_path}")
    print("Training will automatically use local data now.")

if __name__ == "__main__":
    main()
