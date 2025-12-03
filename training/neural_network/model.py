"""
Neural network model for the GA-based ROS local planner.

This module implements a PyTorch model that distills GA-generated navigation
trajectories into a fast neural network. The architecture consists of:
- CostmapEncoder: CNN for processing 50x50 costmap images
- StateEncoder: MLP for robot state, goal, and metadata
- PolicyHead: Output decoder for control sequences

The model takes 4 separate inputs to match the C++ ONNX Runtime interface:
- costmap: [batch, 1, 50, 50]
- robot_state: [batch, 9] - [x, y, theta, v_x, v_y, omega, a_x, a_y, alpha]
- goal_relative: [batch, 3] - [dx, dy, dtheta] in robot frame
- costmap_metadata: [batch, 2] - [inflation_decay, resolution]

Output:
- control_sequence: [batch, 60] - 20 steps × 3 controls (v_x, v_y, omega)
"""

import torch
import torch.nn as nn


class CostmapEncoder(nn.Module):
    """
    CNN encoder for processing 50x50 costmap images.

    Architecture:
    - Conv2d layers: channels [1→32→64→128], kernels [5,3,3], strides [2,2,2]
    - ReLU activations after each conv
    - Flatten and project to hidden_dim (256)

    Input shape: [batch, 1, 50, 50]
    Output shape: [batch, 256]
    """

    def __init__(self, config):
        """
        Initialize CostmapEncoder.

        Args:
            config: Configuration dict with:
                - cnn.channels: List of channel sizes [1, 32, 64, 128]
                - cnn.kernel_sizes: List of kernel sizes [5, 3, 3]
                - cnn.strides: List of strides [2, 2, 2]
                - model.hidden_dim: Output dimension (256)
        """
        super(CostmapEncoder, self).__init__()

        cnn_config = config['model']['cnn']
        channels = cnn_config['channels']
        kernel_sizes = cnn_config['kernel_sizes']
        strides = cnn_config['strides']
        hidden_dim = config['model']['hidden_dim']

        # Build convolutional layers
        conv_layers = []
        for i in range(len(kernel_sizes)):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=kernel_sizes[i] // 2
                )
            )
            conv_layers.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flattened size after convolutions
        # Input: 50x50
        # After conv1 (k=5, s=2, p=2): 25x25
        # After conv2 (k=3, s=2, p=1): 13x13
        # After conv3 (k=3, s=2, p=1): 7x7
        # Actually: with padding=k//2, we get:
        # 50 -> (50 + 2*2 - 5)/2 + 1 = 25
        # 25 -> (25 + 2*1 - 3)/2 + 1 = 13
        # 13 -> (13 + 2*1 - 3)/2 + 1 = 7
        # Final size: 128 * 7 * 7 = 6272
        flattened_size = channels[-1] * 7 * 7

        # Linear projection to hidden_dim
        self.fc = nn.Linear(flattened_size, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, costmap):
        """
        Forward pass through costmap encoder.

        Args:
            costmap: Tensor of shape [batch, 1, 50, 50]

        Returns:
            features: Tensor of shape [batch, 256]
        """
        x = self.conv_layers(costmap)  # [batch, 128, 7, 7]
        x = x.view(x.size(0), -1)      # [batch, 6272]
        x = self.fc(x)                  # [batch, 256]
        x = self.relu(x)                # [batch, 256]
        return x


class StateEncoder(nn.Module):
    """
    MLP encoder for robot state, goal, and costmap metadata.

    Architecture:
    - Concatenate inputs: robot_state(9) + goal_relative(3) + costmap_metadata(2) = 14 dims
    - Linear layers: [14 → 128 → 256]
    - ReLU activations between layers
    - Optional dropout (default 0.0)

    Input shapes:
    - robot_state: [batch, 9]
    - goal_relative: [batch, 3]
    - costmap_metadata: [batch, 2]

    Output shape: [batch, 256]
    """

    def __init__(self, config):
        """
        Initialize StateEncoder.

        Args:
            config: Configuration dict with:
                - mlp.input_dim: Input dimension (14)
                - mlp.hidden_dims: List of hidden dimensions [128, 256]
                - training.dropout: Dropout rate (default 0.0)
        """
        super(StateEncoder, self).__init__()

        mlp_config = config['model']['mlp']
        input_dim = mlp_config['input_dim']
        hidden_dims = mlp_config['hidden_dims']
        dropout = config.get('training', {}).get('dropout', 0.0)

        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, robot_state, goal_relative, costmap_metadata):
        """
        Forward pass through state encoder.

        Args:
            robot_state: Tensor of shape [batch, 9]
            goal_relative: Tensor of shape [batch, 3]
            costmap_metadata: Tensor of shape [batch, 2]

        Returns:
            features: Tensor of shape [batch, 256]
        """
        # Concatenate all inputs
        x = torch.cat([robot_state, goal_relative, costmap_metadata], dim=1)  # [batch, 14]
        x = self.mlp(x)  # [batch, 256]
        return x


class PolicyHead(nn.Module):
    """
    Output decoder for control sequences.

    Architecture:
    - Linear layers: [512 → 256 → 256 → 60]
    - ReLU on intermediate layers only
    - No final activation (regression task, clamping in C++)

    Input shape: [batch, 512] (concatenated costmap + state features)
    Output shape: [batch, 60] (20 steps × 3 controls)
    """

    def __init__(self, config):
        """
        Initialize PolicyHead.

        Args:
            config: Configuration dict with:
                - policy_head.hidden_dims: List of hidden dimensions [256, 256]
                - policy_head.output_dim: Output dimension (60)
                - model.hidden_dim: Input feature dimension (256 * 2 = 512)
        """
        super(PolicyHead, self).__init__()

        policy_config = config['model']['policy_head']
        hidden_dims = policy_config['hidden_dims']
        output_dim = policy_config['output_dim']

        # Input is concatenated costmap + state features
        input_dim = config['model']['hidden_dim'] * 2  # 256 * 2 = 512

        # Build policy head layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        # Final output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.policy = nn.Sequential(*layers)

    def forward(self, fused_features):
        """
        Forward pass through policy head.

        Args:
            fused_features: Tensor of shape [batch, 512]

        Returns:
            control_sequence: Tensor of shape [batch, 60]
        """
        return self.policy(fused_features)


class PlannerPolicy(nn.Module):
    """
    Complete planner policy model combining all components.

    This model orchestrates the CostmapEncoder, StateEncoder, and PolicyHead
    to produce control sequences from multi-modal inputs.

    Forward signature matches C++ ONNX Runtime interface with 4 separate inputs.
    """

    def __init__(self, config):
        """
        Initialize PlannerPolicy.

        Args:
            config: Configuration dict loaded from nn_config.yaml
        """
        super(PlannerPolicy, self).__init__()

        self.config = config

        # Initialize sub-modules
        self.costmap_encoder = CostmapEncoder(config)
        self.state_encoder = StateEncoder(config)
        self.policy_head = PolicyHead(config)

    def forward(self, costmap, robot_state, goal_relative, costmap_metadata):
        """
        Forward pass through complete model.

        Args:
            costmap: Tensor of shape [batch, 1, 50, 50] - normalized costmap
            robot_state: Tensor of shape [batch, 9] - [x, y, theta, v_x, v_y, omega, a_x, a_y, alpha]
            goal_relative: Tensor of shape [batch, 3] - [dx, dy, dtheta] in robot frame
            costmap_metadata: Tensor of shape [batch, 2] - [inflation_decay, resolution]

        Returns:
            control_sequence: Tensor of shape [batch, 60] - flattened control sequence
                              (20 steps × 3 controls: v_x, v_y, omega)
        """
        # Encode costmap
        costmap_features = self.costmap_encoder(costmap)  # [batch, 256]

        # Encode state + goal + metadata
        state_features = self.state_encoder(robot_state, goal_relative, costmap_metadata)  # [batch, 256]

        # Fuse features
        fused = torch.cat([costmap_features, state_features], dim=1)  # [batch, 512]

        # Generate control sequence
        control_sequence = self.policy_head(fused)  # [batch, 60]

        return control_sequence

    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.

        Returns:
            total_params: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self):
        """
        Get a summary of the model architecture.

        Returns:
            summary: String describing the model structure and parameter count
        """
        total_params = self.count_parameters()

        summary = []
        summary.append("=" * 60)
        summary.append("PlannerPolicy Model Summary")
        summary.append("=" * 60)
        summary.append(f"Total trainable parameters: {total_params:,}")
        summary.append("")
        summary.append("Architecture:")
        summary.append(f"  CostmapEncoder: {sum(p.numel() for p in self.costmap_encoder.parameters() if p.requires_grad):,} params")
        summary.append(f"  StateEncoder: {sum(p.numel() for p in self.state_encoder.parameters() if p.requires_grad):,} params")
        summary.append(f"  PolicyHead: {sum(p.numel() for p in self.policy_head.parameters() if p.requires_grad):,} params")
        summary.append("=" * 60)

        return "\n".join(summary)


def create_model(config):
    """
    Factory function to create a PlannerPolicy model from configuration.

    Args:
        config: Configuration dict loaded from nn_config.yaml

    Returns:
        model: PlannerPolicy instance
    """
    model = PlannerPolicy(config)
    return model


if __name__ == "__main__":
    # Quick test of model architecture
    print("Testing PlannerPolicy model...")

    # Create minimal test config
    config = {
        'model': {
            'costmap_size': 50,
            'num_control_steps': 20,
            'hidden_dim': 256,
            'cnn': {
                'channels': [1, 32, 64, 128],
                'kernel_sizes': [5, 3, 3],
                'strides': [2, 2, 2]
            },
            'mlp': {
                'input_dim': 14,
                'hidden_dims': [128, 256]
            },
            'policy_head': {
                'hidden_dims': [256, 256],
                'output_dim': 60
            }
        },
        'training': {
            'dropout': 0.0
        }
    }

    # Create model
    model = create_model(config)
    print(model.get_model_summary())

    # Test forward pass
    batch_size = 2
    costmap = torch.randn(batch_size, 1, 50, 50)
    robot_state = torch.randn(batch_size, 9)
    goal_relative = torch.randn(batch_size, 3)
    costmap_metadata = torch.randn(batch_size, 2)

    output = model(costmap, robot_state, goal_relative, costmap_metadata)

    print(f"\nTest forward pass:")
    print(f"  Input shapes:")
    print(f"    costmap: {costmap.shape}")
    print(f"    robot_state: {robot_state.shape}")
    print(f"    goal_relative: {goal_relative.shape}")
    print(f"    costmap_metadata: {costmap_metadata.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == (batch_size, 60), f"Expected output shape ({batch_size}, 60), got {output.shape}"
    print("\n✓ Model architecture test passed!")
