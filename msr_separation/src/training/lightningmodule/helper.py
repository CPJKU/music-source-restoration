import torch
import re

def map_4stem_to_9stem_checkpoint(model_ckpt, model_state, init_new_stems_from_other=False, add_noise_to_new_stems=False):
    """
    Map checkpoint weights from a 4-stem model to a 9-stem model.
    
    Stem mapping:
    - 4-stem[0] (vocals) -> 9-stem[0] (vocals)
    - 4-stem[1] (bass) -> 9-stem[1] (bass)
    - 4-stem[2] (drums) -> 9-stem[2] (drums)
    - 4-stem[3] (other) -> 9-stem[8] (other)
    
    Args:
        model_ckpt: Checkpoint state dict from 4-stem model
        model_state: Current model state dict (9-stem)
        init_new_stems_from_other: If True, initialize new stems (indices 3-7) from "other" stem weights (4-stem[3]).
                                   Default: False (new stems use random initialization)
        add_noise_to_new_stems: If True and init_new_stems_from_other=True, add small random noise scaled by
                                the variance of pretrained stems. Default: False
    
    Returns:
        Mapped checkpoint state dict
    """
    mapped_ckpt = {}
    
    # Stem mapping: 4-stem index -> 9-stem index
    direct_mapping = {
        0: 0,  # vocals -> vocals
        1: 1,  # bass -> bass
        2: 2,  # drums -> drums
        3: 8,  # other -> other (last stem, 9th)
    }
    
    # Pattern to match mask_estimator keys: mask_estimators.<index>.<rest>
    # Handle both 'roformer.mask_estimators.X' and 'mask_estimators.X' patterns
    mask_estimator_pattern = re.compile(r'(?:^roformer\.)?mask_estimators\.(\d+)(.*)')
    
    # Collect "other" stem weights for initializing new stems (if enabled)
    other_stem_weights = {} if init_new_stems_from_other else None
    
    # First pass: map existing stems and collect "other" stem weights
    for ckpt_key, ckpt_value in model_ckpt.items():
        match = mask_estimator_pattern.match(ckpt_key)
        
        if match:
            # This is a mask_estimator key
            ckpt_stem_idx = int(match.group(1))
            rest_of_key = match.group(2)
            
            # Determine if key has 'roformer.' prefix
            has_roformer_prefix = ckpt_key.startswith('roformer.mask_estimators')
            prefix = 'roformer.' if has_roformer_prefix else ''
            
            if ckpt_stem_idx in direct_mapping:
                # Map directly: 4-stem[0-3] -> 9-stem[0,1,2,8]
                target_stem_idx = direct_mapping[ckpt_stem_idx]
                model_key = f'{prefix}mask_estimators.{target_stem_idx}{rest_of_key}'
                
                if model_key in model_state:
                    # Check shape compatibility
                    if ckpt_value.shape == model_state[model_key].shape:
                        mapped_ckpt[model_key] = ckpt_value
                        
                        # Store "other" stem weights for initializing new stems
                        if init_new_stems_from_other and ckpt_stem_idx == 3:  # "other" stem
                            other_stem_weights[rest_of_key] = ckpt_value
                    else:
                        print(f"Warning: Shape mismatch for {model_key}: checkpoint {ckpt_value.shape} vs model {model_state[model_key].shape}")
                else:
                    print(f"Warning: Key {model_key} not found in model state")
            # Note: New stems (indices 3-7) are left uninitialized - they will be randomly initialized
            # unless init_new_stems_from_other=True
        else:
            # Not a mask_estimator key - copy directly if it exists in model
            # This includes roformer layers, band_split, final_norm, etc.
            if ckpt_key in model_state:
                # Check shape compatibility
                if ckpt_value.shape == model_state[ckpt_key].shape:
                    mapped_ckpt[ckpt_key] = ckpt_value
                else:
                    print(f"Warning: Shape mismatch for {ckpt_key}: checkpoint {ckpt_value.shape} vs model {model_state[ckpt_key].shape}")
            # If key doesn't exist in model, skip it (will be handled as missing key later)
    
    # Second pass: Initialize new stems (3-7) from "other" stem if enabled
    if init_new_stems_from_other and other_stem_weights:
        # Determine which stems are "new" (not in direct_mapping)
        # For 9-stem model: new stems are 3, 4, 5, 6, 7 (guitars, keyboards, synthesizers, percussions, orchestral)
        # We'll detect which stems need initialization by checking model_state
        
        # Find all mask estimator indices in the model
        model_mask_estimator_indices = set()
        for key in model_state.keys():
            match = mask_estimator_pattern.match(key)
            if match:
                model_mask_estimator_indices.add(int(match.group(1)))
        
        # New stems are those not in the direct mapping target indices
        target_indices = set(direct_mapping.values())
        new_stem_indices = sorted([idx for idx in model_mask_estimator_indices if idx not in target_indices])
        
        if new_stem_indices:
            print(f"  Initializing {len(new_stem_indices)} new stems (indices {new_stem_indices}) from 'other' stem weights")
            
            # Compute variance of pretrained stems for noise scaling (if enabled)
            pretrained_variances = {}
            if add_noise_to_new_stems:
                # Collect all pretrained stem weights to compute variance
                pretrained_weights_by_key = {}
                for key in model_ckpt.keys():
                    match = mask_estimator_pattern.match(key)
                    if match:
                        ckpt_stem_idx = int(match.group(1))
                        rest_of_key = match.group(2)
                        if ckpt_stem_idx in direct_mapping:
                            if rest_of_key not in pretrained_weights_by_key:
                                pretrained_weights_by_key[rest_of_key] = []
                            pretrained_weights_by_key[rest_of_key].append(model_ckpt[key])
                
                # Compute variance for each key
                for rest_of_key, weights_list in pretrained_weights_by_key.items():
                    if len(weights_list) > 0:
                        # Stack all pretrained weights and compute variance
                        stacked = torch.stack(weights_list)
                        pretrained_variances[rest_of_key] = torch.var(stacked, dim=0)
            
            # Determine prefix (check if model uses 'roformer.' prefix) - do this once outside the loop
            prefix = ''
            for key in model_state.keys():
                if key.startswith('roformer.mask_estimators'):
                    prefix = 'roformer.'
                    break
                elif key.startswith('mask_estimators'):
                    prefix = ''
                    break
            
            # Initialize each new stem
            for new_stem_idx in new_stem_indices:
                for rest_of_key, other_weight in other_stem_weights.items():
                    model_key = f'{prefix}mask_estimators.{new_stem_idx}{rest_of_key}'
                    
                    if model_key in model_state:
                        if other_weight.shape == model_state[model_key].shape:
                            # Copy weights from "other" stem
                            new_weight = other_weight.clone()
                            
                            # Add noise if enabled
                            if add_noise_to_new_stems and rest_of_key in pretrained_variances:
                                noise_scale = torch.sqrt(pretrained_variances[rest_of_key] + 1e-8)  # Add small epsilon for stability
                                noise = torch.randn_like(new_weight) * noise_scale * 0.1  # Scale noise by 0.1
                                new_weight = new_weight + noise
                            
                            mapped_ckpt[model_key] = new_weight
                        else:
                            print(f"Warning: Shape mismatch for {model_key}: other stem {other_weight.shape} vs model {model_state[model_key].shape}")
                    else:
                        print(f"Warning: Key {model_key} not found in model state")
    
    return mapped_ckpt

def load_ckpt(path, model, map_4stem_to_9stem=None, init_new_stems_from_other=False, add_noise_to_new_stems=False):
    """
    Load checkpoint into model.
    
    Args:
        path: Path to checkpoint file
        model: Model or LightningModule to load checkpoint into
        map_4stem_to_9stem: If True, force mapping. If None, auto-detect. If False, disable.
        init_new_stems_from_other: If True, initialize new stems (indices 3-7) from "other" stem weights.
                                   Only used when mapping 4-stem to 9-stem. Default: False
        add_noise_to_new_stems: If True and init_new_stems_from_other=True, add small random noise scaled by
                                the variance of pretrained stems. Default: False
    
    Returns:
        dict: Full checkpoint dictionary (includes EMA state if present)
    """
    full_ckpt = torch.load(path, weights_only=False, map_location='cpu')
    model_ckpt = full_ckpt['state_dict']
    if set(model.state_dict().keys()) != set(model_ckpt.keys()): # remove prefix, incase the ckpt is of lightning module
        one_model_key = next(iter(model.state_dict().keys()))
        ckpt_corresponding_key = [k for k in model_ckpt.keys() if k.endswith(one_model_key)]
        if ckpt_corresponding_key:
            prefix = ckpt_corresponding_key[0][:-len(one_model_key)]
            model_ckpt = {k[len(prefix):]: v for k, v in model_ckpt.items() if k.startswith(prefix)  }# remove prefix
    
    model_state = model.state_dict()
    
    # Auto-detect if we need to map 4-stem to 9-stem
    should_map = False
    if map_4stem_to_9stem is None:
        # Auto-detect: check if checkpoint has 4 mask_estimators and model has 9
        ckpt_mask_estimators = set()
        model_mask_estimators = set()
        
        for key in model_ckpt.keys():
            match = re.match(r'mask_estimators\.(\d+)', key)
            if match:
                ckpt_mask_estimators.add(int(match.group(1)))
        
        for key in model_state.keys():
            match = re.match(r'mask_estimators\.(\d+)', key)
            if match:
                model_mask_estimators.add(int(match.group(1)))
        
        if len(ckpt_mask_estimators) == 4 and len(model_mask_estimators) == 9:
            should_map = True
    elif map_4stem_to_9stem:
        should_map = True
    
    if should_map:
        print("Detected 4-stem checkpoint loading into 9-stem model. Applying stem mapping...")
        print("  Mapping: vocals->vocals, bass->bass, drums->drums, other->other")
        if init_new_stems_from_other:
            print("  Initializing new stems (guitars, keyboards, synthesizers, percussions, orchestral elements) from 'other' stem")
            if add_noise_to_new_stems:
                print("  Adding small random noise scaled by pretrained stem variance to new stems")
        else:
            print("  New stems (indices 3-7) will use random initialization (default)")
        model_ckpt = map_4stem_to_9stem_checkpoint(model_ckpt, model_state, init_new_stems_from_other, add_noise_to_new_stems)
    
    # Filter out LoRA parameters if they don't exist in checkpoint
    # This allows loading checkpoints that were trained without LoRA
    filtered_ckpt = {}
    missing_keys = []
    unexpected_keys = []
    
    for key in model_state.keys():
        if key in model_ckpt:
            filtered_ckpt[key] = model_ckpt[key]
        else:
            # This key is in the model but not in the checkpoint
            # If it's a LoRA parameter, we'll skip it (initialized to zero anyway)
            if 'lora_A' in key or 'lora_B' in key:
                missing_keys.append(key)
                continue
            else:
                missing_keys.append(key)
    
    for key in model_ckpt.keys():
        if key not in model_state:
            unexpected_keys.append(key)
    
    if missing_keys:
        lora_count = sum(1 for k in missing_keys if 'lora' in k.lower())
        if lora_count > 0:
            print(f"Info: {lora_count} LoRA parameters missing from checkpoint (will use zero initialization)")
        non_lora_missing = len(missing_keys) - lora_count
        if non_lora_missing > 0:
            print(f"Warning: {non_lora_missing} non-LoRA keys missing from checkpoint")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint (will be ignored)")
    
    # Load with strict=False to allow missing LoRA parameters
    model.load_state_dict(filtered_ckpt, strict=False)
    
    return full_ckpt