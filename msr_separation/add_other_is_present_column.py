#!/usr/bin/env python3
"""
Add the 'other_is_present' column to the mss8s dataset.

This script loads the dataset, adds the 'other_is_present' column with value True
for all samples (since 'other' always has a path), and saves the updated dataset
back to disk.
"""

import os
from pathlib import Path
from datasets import load_from_disk, DatasetDict, Dataset


def add_other_is_present_column(dataset_path: str, output_path: str = None):
    """
    Add 'other_is_present' column to all splits of the dataset.
    
    Args:
        dataset_path: Path to the saved HuggingFace dataset
        output_path: Path to save the updated dataset. If None, overwrites the original.
    """
    print("=" * 80)
    print("ADDING 'other_is_present' COLUMN TO MSS8S DATASET")
    print("=" * 80)
    print(f"\nLoading dataset from: {dataset_path}\n")
    
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    ds = load_from_disk(dataset_path)
    print(f"✓ Dataset loaded successfully!")
    print(f"  Available splits: {list(ds.keys())}\n")
    
    # Check if 'other_is_present' already exists
    sample_split = list(ds.keys())[0]
    sample_dataset = ds[sample_split]
    if len(sample_dataset) > 0:
        sample = sample_dataset[0]
        if "other_is_present" in sample:
            print("⚠️  Warning: 'other_is_present' column already exists in the dataset!")
            response = input("Do you want to overwrite it? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Aborting.")
                return
    
    # Process each split
    updated_splits = {}
    
    for split_name in ds.keys():
        split_ds = ds[split_name]
        print(f"Processing split: {split_name} ({len(split_ds)} samples)")
        
        # Check for empty 'other' paths before processing
        empty_other_count = sum(
            1 for ex in split_ds 
            if not (ex.get("other", "") and str(ex.get("other", "")).strip())
        )
        if empty_other_count > 0:
            print(f"  ⚠️  Warning: {empty_other_count} samples have empty 'other' path")
        
        # Add 'other_is_present' column with value True for all samples
        # Since 'other' always contains a path in the 8-stem dataset, set to True
        def add_other_is_present(example):
            """Add other_is_present column. Set to True since 'other' always has a path."""
            # Set to True for all samples (as per user requirement)
            example["other_is_present"] = True
            return example
        
        # Apply the function to add the column
        updated_split = split_ds.map(add_other_is_present, desc=f"Adding other_is_present to {split_name}")
        updated_splits[split_name] = updated_split
        
        # Verify the column was added
        if len(updated_split) > 0:
            sample = updated_split[0]
            if "other_is_present" in sample:
                # Count how many have it set to True
                true_count = sum(1 for ex in updated_split if ex.get("other_is_present", False))
                print(f"  ✓ Added 'other_is_present' column")
                print(f"    Samples with other_is_present=True: {true_count}/{len(updated_split)}")
            else:
                print(f"  ✗ Failed to add column!")
        else:
            print(f"  ⚠️  Split is empty, skipping verification")
    
    # Create updated DatasetDict
    updated_dataset = DatasetDict(updated_splits)
    
    # Determine output path
    if output_path is None:
        output_path = dataset_path
        print(f"\n⚠️  Will overwrite original dataset at: {output_path}")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborting. No changes made.")
            return
    
    # Save the updated dataset
    print(f"\nSaving updated dataset to: {output_path}")
    updated_dataset.save_to_disk(output_path)
    print(f"✓ Dataset saved successfully!")
    
    # Verify the saved dataset
    print(f"\nVerifying saved dataset...")
    verify_ds = load_from_disk(output_path)
    for split_name in verify_ds.keys():
        split_ds = verify_ds[split_name]
        if len(split_ds) > 0:
            sample = split_ds[0]
            if "other_is_present" in sample:
                true_count = sum(1 for ex in split_ds if ex.get("other_is_present", False))
                print(f"  {split_name}: {true_count}/{len(split_ds)} samples have other_is_present=True")
            else:
                print(f"  {split_name}: ✗ Column not found!")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add 'other_is_present' column to mss8s dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/opt/datasets/HF_datasets/saved/mss8s",
        help="Path to the dataset to update"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/opt/datasets/HF_datasets/saved/mss8s_other_is_present",
        help="Path to save the updated dataset"
    )
    
    args = parser.parse_args()
    
    add_other_is_present_column(args.dataset_path, args.output_path)

