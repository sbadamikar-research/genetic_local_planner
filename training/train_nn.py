#!/usr/bin/env python3
"""
Neural network training script for GA-based ROS local planner.

This script trains a PyTorch model to distill GA-generated navigation trajectories
into a fast neural network, then exports to ONNX for C++ deployment.

Usage:
    python training/train_nn.py \\
        --data models/checkpoints/ga_trajectories.pkl \\
        --config training/config/nn_config.yaml \\
        --output models/planner_policy.onnx \\
        --checkpoint models/checkpoints/best_model.pth

The script implements:
- Training loop with validation
- Early stopping and learning rate scheduling
- Model checkpointing (save best model)
- ONNX export with verification
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add training directory to Python path
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir))

from neural_network.model import PlannerPolicy, create_model
from neural_network.dataset import TrajectoryDataset


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        config: Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(data_path: str, config: dict) -> tuple:
    """
    Create training and validation dataloaders.

    Args:
        data_path: Path to GA trajectory pickle file
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load dataset with filtering
    filter_percentile = config['training']['filter_percentile']
    dataset = TrajectoryDataset(data_path, filter_percentile=filter_percentile)

    # Print dataset statistics
    dataset.print_statistics()

    # Split into train and validation
    train_ratio = config['training']['train_split']
    train_dataset, val_dataset = dataset.train_test_split(train_ratio=train_ratio, random_seed=42)

    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = 4  # Fixed for core implementation

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device) -> float:
    """
    Train model for one epoch.

    Args:
        model: PlannerPolicy model
        dataloader: Training data loader
        optimizer: Optimizer (Adam)
        criterion: Loss function (MSE)
        device: Device to run on (cpu/cuda)

    Returns:
        avg_loss: Average training loss for the epoch
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    # Progress bar
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        # Unpack batch
        costmap, robot_state, goal_relative, costmap_metadata, control_target = batch

        # Move to device
        costmap = costmap.to(device)
        robot_state = robot_state.to(device)
        goal_relative = goal_relative.to(device)
        costmap_metadata = costmap_metadata.to(device)
        control_target = control_target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        control_pred = model(costmap, robot_state, goal_relative, costmap_metadata)

        # Compute loss
        loss = criterion(control_pred, control_target)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> float:
    """
    Validate model on validation set.

    Args:
        model: PlannerPolicy model
        dataloader: Validation data loader
        criterion: Loss function (MSE)
        device: Device to run on (cpu/cuda)

    Returns:
        avg_loss: Average validation loss
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    # No gradient computation
    with torch.no_grad():
        # Progress bar
        pbar = tqdm(dataloader, desc="Validation", leave=False)

        for batch in pbar:
            # Unpack batch
            costmap, robot_state, goal_relative, costmap_metadata, control_target = batch

            # Move to device
            costmap = costmap.to(device)
            robot_state = robot_state.to(device)
            goal_relative = goal_relative.to(device)
            costmap_metadata = costmap_metadata.to(device)
            control_target = control_target.to(device)

            # Forward pass
            control_pred = model(costmap, robot_state, goal_relative, costmap_metadata)

            # Compute loss
            loss = criterion(control_pred, control_target)

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model: nn.Module, path: str, epoch: int, val_loss: float):
    """
    Save model checkpoint.

    Args:
        model: PlannerPolicy model
        path: Path to save checkpoint
        epoch: Current epoch number
        val_loss: Current validation loss
    """
    # Create parent directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save checkpoint with metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, path)


def export_to_onnx(model: nn.Module, output_path: str, config: dict):
    """
    Export trained model to ONNX format with verification.

    Args:
        model: Trained PlannerPolicy model
        output_path: Path to save ONNX model
        config: Configuration dictionary

    Raises:
        AssertionError: If ONNX output doesn't match PyTorch output
    """
    # Set model to eval mode
    model.eval()

    # Create dummy inputs matching expected shapes
    dummy_costmap = torch.randn(1, 1, 50, 50)
    dummy_robot_state = torch.randn(1, 9)
    dummy_goal_relative = torch.randn(1, 3)
    dummy_costmap_metadata = torch.randn(1, 2)

    # Create parent directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get ONNX configuration
    onnx_config = config.get('onnx', {})
    opset_version = onnx_config.get('opset_version', 14)
    dynamic_batch = onnx_config.get('dynamic_batch', True)

    print(f"\nExporting model to ONNX (opset version {opset_version})...")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_costmap, dummy_robot_state, dummy_goal_relative, dummy_costmap_metadata),
        output_path,
        input_names=['costmap', 'robot_state', 'goal_relative', 'costmap_metadata'],
        output_names=['control_sequence'],
        dynamic_axes={
            'costmap': {0: 'batch'},
            'robot_state': {0: 'batch'},
            'goal_relative': {0: 'batch'},
            'costmap_metadata': {0: 'batch'},
            'control_sequence': {0: 'batch'}
        } if dynamic_batch else None,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )

    print(f"✓ ONNX export complete: {output_path}")

    # Verify export if requested
    if onnx_config.get('verify_export', True):
        print("\nVerifying ONNX export...")
        verify_onnx_export(model, output_path, dummy_costmap, dummy_robot_state,
                          dummy_goal_relative, dummy_costmap_metadata)


def verify_onnx_export(model: nn.Module,
                       onnx_path: str,
                       dummy_costmap: torch.Tensor,
                       dummy_robot_state: torch.Tensor,
                       dummy_goal_relative: torch.Tensor,
                       dummy_costmap_metadata: torch.Tensor):
    """
    Verify ONNX export by comparing PyTorch and ONNX Runtime outputs.

    Args:
        model: Original PyTorch model
        onnx_path: Path to exported ONNX model
        dummy_costmap: Dummy costmap input
        dummy_robot_state: Dummy robot state input
        dummy_goal_relative: Dummy goal input
        dummy_costmap_metadata: Dummy metadata input

    Raises:
        AssertionError: If outputs don't match within tolerance
    """
    import onnxruntime as ort

    # Run PyTorch inference
    model.eval()
    with torch.no_grad():
        torch_output = model(dummy_costmap, dummy_robot_state,
                            dummy_goal_relative, dummy_costmap_metadata)
    torch_output_np = torch_output.numpy()

    # Run ONNX Runtime inference
    ort_session = ort.InferenceSession(onnx_path)

    ort_inputs = {
        'costmap': dummy_costmap.numpy(),
        'robot_state': dummy_robot_state.numpy(),
        'goal_relative': dummy_goal_relative.numpy(),
        'costmap_metadata': dummy_costmap_metadata.numpy()
    }

    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    max_diff = np.abs(torch_output_np - ort_output).max()
    mean_diff = np.abs(torch_output_np - ort_output).mean()

    print(f"  PyTorch vs ONNX Runtime comparison:")
    print(f"    Max absolute difference: {max_diff:.8f}")
    print(f"    Mean absolute difference: {mean_diff:.8f}")

    # Check if difference is within tolerance
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"  ✓ ONNX verification passed (max_diff={max_diff:.8f} < {tolerance})")
    else:
        raise AssertionError(
            f"ONNX output mismatch: max_diff={max_diff:.8f} >= {tolerance}\n"
            "The exported ONNX model produces different results than PyTorch model."
        )


def main():
    """
    Main training function.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train neural network for GA-based ROS local planner"
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to GA trajectory pickle file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='training/config/nn_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/planner_policy.onnx',
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/checkpoints/best_model.pth',
        help='Path to save best model checkpoint'
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("Neural Network Training for GA-Based ROS Local Planner")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Config: {args.config}")
    print(f"Output ONNX: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 70 + "\n")

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    print(f"✓ Configuration loaded\n")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    print(model.get_model_summary())
    print()

    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(args.data, config)
    print()

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create loss function
    criterion = nn.MSELoss()

    # Create learning rate scheduler
    scheduler_config = config['training']['scheduler']
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config['mode'],
        factor=scheduler_config['factor'],
        patience=scheduler_config['patience'],
        min_lr=scheduler_config['min_lr'],
        verbose=True
    )

    # Early stopping parameters
    early_stopping_config = config['training']['early_stopping']
    early_stopping_patience = early_stopping_config['patience']
    min_delta = early_stopping_config['min_delta']

    # Training loop
    num_epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience_counter = 0

    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping and checkpointing
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, args.checkpoint, epoch, val_loss)
            print(f"  → New best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  → No improvement (patience: {patience_counter}/{early_stopping_patience})")

            if patience_counter >= early_stopping_patience:
                print(f"\n{'=' * 70}")
                print(f"Early stopping triggered at epoch {epoch + 1}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"{'=' * 70}")
                break

    # Training complete
    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'=' * 70}\n")

    # Load best checkpoint for export
    print("Loading best model checkpoint for ONNX export...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.6f})")

    # Export to ONNX
    export_to_onnx(model, args.output, config)

    # Final summary
    print(f"\n{'=' * 70}")
    print("Training pipeline complete!")
    print(f"{'=' * 70}")
    print(f"Best model checkpoint: {args.checkpoint}")
    print(f"ONNX model: {args.output}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
