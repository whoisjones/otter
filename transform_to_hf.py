"""
Script to convert old model checkpoints to the new safetensors format.

Old format (Bi-Encoder):
- config.json (span model config)
- token_encoder/ (directory with encoder model)
- type_encoder/ (directory with encoder model)
- span_model_weights.pt or model.pt (custom weights only)

Old format (Cross-Encoder):
- config.json (span model config)
- tokenizer/ (directory with tokenizer)
- model.bin (all model weights)

New format (Bi-Encoder):
- config.json (span model config)
- token_encoder_config.json (token encoder config)
- type_encoder_config.json (type encoder config)
- model.safetensors (entire model weights including encoders)

New format (Cross-Encoder):
- config.json (span model config)
- token_encoder_config.json (token encoder config)
- tokenizer/ (directory with tokenizer)
- model.safetensors (entire model weights including encoder)

This script:
1. Loads the old checkpoint using from_pretrained (legacy method)
2. Saves it using the new save_pretrained method (safetensors format)
"""

import torch
from pathlib import Path
from src.model import BiEncoderModel, ContrastiveBiEncoderModel, CrossEncoderModel, ContrastiveCrossEncoderModel
from src.config import SpanModelConfig
from transformers import AutoConfig, AutoTokenizer
import argparse
import sys
import shutil


def convert_checkpoint_to_new_format(checkpoint_path: str, model_class=None, dry_run: bool = False, backup: bool = True):
    """
    Convert a single checkpoint from old format to new safetensors format.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        model_class: Model class to use. If None, will try to detect from config.
        dry_run: If True, only print what would be done without actually converting
        backup: If True, create a backup of the old checkpoint (deprecated, output goes to new directory)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        return False
    
    # Create output path by appending _new_format
    output_path = Path("bi_hf")
    
    # Check if output already exists (different checks for bi-encoder vs cross-encoder)
    model_safetensors_path = output_path / "model.safetensors"
    token_encoder_config_path = output_path / "token_encoder_config.json"
    type_encoder_config_path = output_path / "type_encoder_config.json"
    
    is_bi_encoder = type_encoder_config_path.exists()
    # Check for tokenizer files directly in output_path (vocab.json or tokenizer_config.json)
    has_tokenizer = (output_path / "vocab.json").exists() or (output_path / "tokenizer_config.json").exists()
    is_cross_encoder = (model_safetensors_path.exists() and token_encoder_config_path.exists() and has_tokenizer)
    
    if (is_bi_encoder and model_safetensors_path.exists()) or is_cross_encoder:
        print(f"✓ Converted checkpoint already exists: {output_path}")
        return True
    
    print(f"\n{'='*60}")
    print(f"Converting checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load config to determine model type
    try:
        config = SpanModelConfig.from_pretrained(str(checkpoint_path))
    except Exception as e:
        print(f"Error loading config: {e}")
        return False
    
    # Determine model class if not provided
    if model_class is None:
        # Detect model type from checkpoint structure
        token_encoder_dir = checkpoint_path / "token_encoder"
        type_encoder_dir = checkpoint_path / "type_encoder"
        model_bin = checkpoint_path / "model.bin"
        
        if model_bin.exists():
            # Cross encoder format
            if config.loss_fn == "contrastive":
                model_class = ContrastiveCrossEncoderModel
            else:
                model_class = CrossEncoderModel
        elif token_encoder_dir.exists() and type_encoder_dir.exists():
            # Bi encoder format
            if config.loss_fn == "contrastive":
                model_class = ContrastiveBiEncoderModel
            else:
                model_class = BiEncoderModel
        else:
            # Default to BiEncoderModel
            model_class = BiEncoderModel
        
        print(f"Detected model class: {model_class.__name__}")
    
    # Check if old format files exist
    token_encoder_dir = checkpoint_path / "token_encoder"
    type_encoder_dir = checkpoint_path / "type_encoder"
    model_bin = checkpoint_path / "model.bin"
    old_weights_file = checkpoint_path / "span_model_weights.pt"
    if not old_weights_file.exists():
        old_weights_file = checkpoint_path / "model.pt"
    
    is_cross_encoder = model_bin.exists()
    is_bi_encoder = token_encoder_dir.exists() and type_encoder_dir.exists()
    
    if not is_bi_encoder and not is_cross_encoder:
        print(f"Warning: Could not determine checkpoint format (bi-encoder or cross-encoder).")
    
    if is_cross_encoder and not (checkpoint_path / "tokenizer").exists():
        print(f"Warning: Cross encoder checkpoint found but tokenizer directory not found.")
    
    if not model_bin.exists() and not old_weights_file.exists():
        print(f"Warning: No weights file found (model.bin, span_model_weights.pt, or model.pt).")
    
    if dry_run:
        print("\n[DRY RUN] Would:")
        print(f"  1. Load model from {checkpoint_path} using legacy format")
        print(f"  2. Save model to {output_path} using new safetensors format")
        return True
    
    # Load model using legacy from_pretrained (handles old format)
    print("Loading model from old format...")
    try:
        model = model_class.from_pretrained(str(checkpoint_path))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load tokenizers
    tokenizer = None
    token_tokenizer = None
    type_tokenizer = None
    
    is_cross_encoder_type = model_class in (CrossEncoderModel, ContrastiveCrossEncoderModel)
    is_bi_encoder_type = model_class in (BiEncoderModel, ContrastiveBiEncoderModel)
    
    if is_cross_encoder_type:
        # Try loading from checkpoint directory (new format) or tokenizer subdirectory (old format)
        tokenizer_subdir = checkpoint_path / "tokenizer"
        
        if tokenizer_subdir.exists():
            # Old format: tokenizer in subdirectory
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_subdir))
                print("✓ Tokenizer loaded from subdirectory")
            except Exception as e:
                print(f"Warning: Could not load tokenizer from subdirectory: {e}")
        else:
            # Try loading from checkpoint directory directly (new format)
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
                print("✓ Tokenizer loaded from checkpoint directory")
            except Exception as e:
                # If tokenizer doesn't exist, create one with the tokens
                print("Creating tokenizer with required tokens...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
                    tokenizer.add_tokens(["[LABEL]"], special_tokens=True)
                    tokenizer.add_tokens(["[SPAN_THRESHOLD]"], special_tokens=True)
                    print("✓ Tokenizer created successfully")
                except Exception as e2:
                    print(f"Warning: Could not create tokenizer: {e2}")
    
    elif is_bi_encoder_type:
        # Load tokenizers for bi-encoder models
        token_tokenizer_path = checkpoint_path / "token_tokenizer"
        type_tokenizer_path = checkpoint_path / "type_tokenizer"
        
        # Try loading from new format subdirectories
        if token_tokenizer_path.exists():
            try:
                token_tokenizer = AutoTokenizer.from_pretrained(str(token_tokenizer_path))
                print("✓ Token tokenizer loaded")
            except Exception as e:
                print(f"Warning: Could not load token tokenizer: {e}")
        else:
            # Try loading from old format token_encoder directory
            token_encoder_dir = checkpoint_path / "token_encoder"
            if token_encoder_dir.exists():
                try:
                    token_tokenizer = AutoTokenizer.from_pretrained(str(token_encoder_dir))
                    print("✓ Token tokenizer loaded from token_encoder directory")
                except Exception:
                    pass
        
        if type_tokenizer_path.exists():
            try:
                type_tokenizer = AutoTokenizer.from_pretrained(str(type_tokenizer_path))
                print("✓ Type tokenizer loaded")
            except Exception as e:
                print(f"Warning: Could not load type tokenizer: {e}")
        else:
            # Try loading from old format type_encoder directory
            type_encoder_dir = checkpoint_path / "type_encoder"
            if type_encoder_dir.exists():
                try:
                    type_tokenizer = AutoTokenizer.from_pretrained(str(type_encoder_dir))
                    print("✓ Type tokenizer loaded from type_encoder directory")
                except Exception:
                    pass
        
        # Create tokenizers if they don't exist
        if token_tokenizer is None:
            try:
                token_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
                print("✓ Token tokenizer created")
            except Exception as e:
                print(f"Warning: Could not create token tokenizer: {e}")
        
        if type_tokenizer is None:
            try:
                type_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)
                print("✓ Type tokenizer created")
            except Exception as e:
                print(f"Warning: Could not create type tokenizer: {e}")
    
    # Save using new format to output path
    print(f"Saving model in new format to {output_path}...")
    try:
        if is_cross_encoder_type and tokenizer is not None:
            model.save_pretrained(str(output_path), tokenizer=tokenizer)
        elif is_bi_encoder_type:
            if token_tokenizer is not None and type_tokenizer is not None:
                model.save_pretrained(str(output_path), token_tokenizer=token_tokenizer, type_tokenizer=type_tokenizer)
            else:
                model.save_pretrained(str(output_path))
        else:
            model.save_pretrained(str(output_path))
        print("✓ Model saved in new format")
        
        # Verify tokenizers were saved correctly
        if is_cross_encoder_type and tokenizer is not None:
            try:
                test_tokenizer = AutoTokenizer.from_pretrained(str(output_path))
                print(f"✓ Tokenizer saved and can be loaded with from_pretrained: {output_path}")
            except Exception as e:
                print(f"Warning: Tokenizer saved but could not be loaded: {e}")
        elif is_bi_encoder_type:
            if token_tokenizer is not None:
                try:
                    test_token_tokenizer = AutoTokenizer.from_pretrained(str(output_path / "token_tokenizer"))
                    print(f"✓ Token tokenizer saved and can be loaded: {output_path / 'token_tokenizer'}")
                except Exception as e:
                    print(f"Warning: Token tokenizer saved but could not be loaded: {e}")
            if type_tokenizer is not None:
                try:
                    test_type_tokenizer = AutoTokenizer.from_pretrained(str(output_path / "type_tokenizer"))
                    print(f"✓ Type tokenizer saved and can be loaded: {output_path / 'type_tokenizer'}")
                except Exception as e:
                    print(f"Warning: Type tokenizer saved but could not be loaded: {e}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Conversion complete!")
    print(f"  - Original checkpoint: {checkpoint_path}")
    print(f"  - New format checkpoint: {output_path}")
    print(f"  - New format files:")
    print(f"    - config.json")
    print(f"    - token_encoder_config.json")
    if model_class in (BiEncoderModel, ContrastiveBiEncoderModel):
        print(f"    - type_encoder_config.json")
        if token_tokenizer is not None:
            print(f"    - token_tokenizer/")
        if type_tokenizer is not None:
            print(f"    - type_tokenizer/")
    elif tokenizer is not None:
        print(f"    - tokenizer files (vocab.json, tokenizer_config.json, etc.)")
    print(f"    - model.safetensors")
    
    return True


def convert_directory(directory_path: str, model_class=None, recursive: bool = False, dry_run: bool = False, backup: bool = True):
    """
    Convert all checkpoints in a directory.
    
    Args:
        directory_path: Path to directory containing checkpoints
        model_class: Model class to use
        recursive: If True, search recursively for checkpoints
        dry_run: If True, only print what would be done
        backup: If True, create backups
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"Error: Directory does not exist: {directory_path}")
        return
    
    # Find all checkpoint directories
    checkpoints = []
    if recursive:
        # Find all directories containing config.json
        for d in directory_path.rglob("config.json"):
            checkpoint_dir = d.parent
            # Check if it's a checkpoint (bi-encoder or cross-encoder format)
            if ((checkpoint_dir / "token_encoder").exists() or 
                (checkpoint_dir / "model.pt").exists() or 
                (checkpoint_dir / "span_model_weights.pt").exists() or
                (checkpoint_dir / "model.bin").exists() or
                (checkpoint_dir / "model.safetensors").exists()):
                if checkpoint_dir not in checkpoints:
                    checkpoints.append(checkpoint_dir)
    else:
        # Only check direct subdirectories
        for subdir in directory_path.iterdir():
            if subdir.is_dir():
                # Check if it's a checkpoint
                if ((subdir / "config.json").exists() and 
                    ((subdir / "token_encoder").exists() or 
                     (subdir / "model.pt").exists() or 
                     (subdir / "span_model_weights.pt").exists() or
                     (subdir / "model.bin").exists() or
                     (subdir / "model.safetensors").exists())):
                    checkpoints.append(subdir)
    
    if not checkpoints:
        print(f"No checkpoints found in {directory_path}")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s) to convert")
    
    success_count = 0
    for checkpoint in checkpoints:
        if convert_checkpoint_to_new_format(checkpoint, model_class=model_class, dry_run=dry_run, backup=backup):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {success_count}/{len(checkpoints)} checkpoints converted")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Convert old model checkpoints to new safetensors format")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to checkpoint directory or parent directory containing checkpoints"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bi_encoder", "contrastive_bi_encoder", "cross_encoder", "contrastive_cross_encoder"],
        default=None,
        help="Model type. If not specified, will try to detect from checkpoint structure."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If path is a directory, search recursively for checkpoints"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done without actually converting"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backups of old checkpoints"
    )
    
    args = parser.parse_args()
    
    # Map model type to class
    model_class_map = {
        "bi_encoder": BiEncoderModel,
        "contrastive_bi_encoder": ContrastiveBiEncoderModel,
        "cross_encoder": CrossEncoderModel,
        "contrastive_cross_encoder": ContrastiveCrossEncoderModel,
    }
    model_class = model_class_map.get(args.model_type) if args.model_type else None
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    if path.is_dir():
        # Check if it's a single checkpoint or directory of checkpoints
        if (path / "config.json").exists():
            # Single checkpoint directory
            convert_checkpoint_to_new_format(path, model_class=model_class, dry_run=args.dry_run, backup=not args.no_backup)
        else:
            # Directory containing multiple checkpoints
            convert_directory(path, model_class=model_class, recursive=args.recursive, dry_run=args.dry_run, backup=not args.no_backup)
    else:
        print(f"Error: Path must be a directory: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
