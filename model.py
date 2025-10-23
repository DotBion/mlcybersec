"""
Simple Multi-Layer Perceptron (MLP) for MNIST classification using PyTorch.

Architecture:
    Input: 784 (28x28 flattened MNIST images)
    Hidden Layer: 128 neurons with ReLU activation
    Output: 10 classes (digits 0-9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLP(nn.Module):
    """
    A simple 2-layer MLP for MNIST digit classification.

    Architecture:
        fc1: Linear(784, 128) + ReLU
        fc2: Linear(128, 10)

    Args:
        input_dim (int): Input dimension (default: 784 for 28x28 images)
        hidden (int): Hidden layer size (default: 128)
        nclass (int): Number of output classes (default: 10)
    """

    def __init__(self, input_dim=784, hidden=128, nclass=10):
        super(TinyMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.nclass = nclass

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, nclass)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Logits of shape (batch_size, nclass)
        """
        # First layer: linear transformation + ReLU activation
        x = F.relu(self.fc1(x))

        # Second layer: linear transformation (no activation - raw logits)
        logits = self.fc2(x)

        return logits

    def predict(self, x):
        """
        Make predictions for input data.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Predicted class labels of shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def load_weights(self, path):
        """
        Load pre-trained weights from a file.

        Args:
            path (str): Path to .pth file containing model state dict
        """
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()


def get_model(pretrained_path=None, device='cpu'):
    """
    Get or create the model instance.

    Args:
        pretrained_path (str, optional): Path to pre-trained weights (.pth file)
        device (str): Device to place model on ('cpu' or 'cuda')

    Returns:
        TinyMLP: The model instance
    """
    model = TinyMLP()

    if pretrained_path is not None:
        model.load_weights(pretrained_path)

    model = model.to(device)
    model.eval()

    return model
