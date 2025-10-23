"""
Utility functions for evaluating adversarial attacks using PyTorch.
"""

import torch
import json
import os

# Try to import matplotlib, but make it optional
try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Image saving will be skipped.")


def accuracy(model, x, y, batch_size=256):
    """
    Compute classification accuracy.

    Args:
        model (nn.Module): PyTorch model
        x (torch.Tensor): Input images, shape (N, input_dim)
        y (torch.Tensor): True labels, shape (N,)
        batch_size (int): Batch size for evaluation

    Returns:
        float: Accuracy as a value between 0 and 1
    """
    model.eval()
    N = x.shape[0]
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_x = x[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            preds = model.predict(batch_x)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total


def evaluate_and_export(model, x_test, y_test, x_adv, out_prefix="results"):
    """
    Evaluate adversarial attack and export results.

    Args:
        model (nn.Module): PyTorch model
        x_test (torch.Tensor): Clean test images
        y_test (torch.Tensor): True labels
        x_adv (torch.Tensor): Adversarial examples
        out_prefix (str): Prefix for output files

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Compute metrics
    clean_acc = accuracy(model, x_test, y_test)
    adv_acc = accuracy(model, x_adv, y_test)

    # Compute average L-infinity perturbation
    linf = torch.max(torch.abs(x_adv - x_test), dim=1)[0].mean().item()

    metrics = {
        "clean_accuracy": float(clean_acc),
        "adv_accuracy": float(adv_acc),
        "avg_linf": float(linf),
        "n_samples": int(x_test.shape[0])
    }

    # Save metrics to JSON
    with open(out_prefix + ".json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Get predictions
    model.eval()
    with torch.no_grad():
        preds_clean = model.predict(x_test)
        preds_adv = model.predict(x_adv)

    # Only save images if matplotlib is available
    if HAS_MATPLOTLIB:
        try:
            os.makedirs(out_prefix + "_imgs", exist_ok=True)

            def save_img(arr, path):
                """Save a flattened image tensor as PNG."""
                # Move to CPU and convert to numpy
                img = arr.cpu().numpy().reshape(28, 28)
                plt.imsave(path, img, cmap='gray')

            # Save success example (where attack changed prediction)
            success_idxs = (preds_clean != preds_adv).nonzero(as_tuple=True)[0]
            if len(success_idxs) > 0:
                idx = success_idxs[0].item()
                save_img(x_adv[idx], os.path.join(out_prefix + "_imgs", "success_example.png"))

            # Save failure example (where attack didn't change prediction)
            failure_idxs = (preds_clean == preds_adv).nonzero(as_tuple=True)[0]
            if len(failure_idxs) > 0:
                idx = failure_idxs[0].item()
                save_img(x_adv[idx], os.path.join(out_prefix + "_imgs", "failure_example.png"))

        except Exception as e:
            print(f"Warning: Could not save images: {e}")
    else:
        print("Image saving skipped (matplotlib not available)")

    return metrics
