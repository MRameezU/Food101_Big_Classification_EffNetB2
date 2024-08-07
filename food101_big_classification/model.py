import torch
import torchvision

from torch import nn

def create_effnetb2_model(num_classes: int = 3, seed: int = 42):
    """
    Creates an EfficientNetB2 model with a custom classifier for the given number of classes.

    Args:
    num_classes (int): Number of output classes for the classifier.
    seed (int): Random seed for reproducibility.

    Returns:
    model (torchvision.models): Modified EfficientNetB2 model with custom classifier.
    transforms (torchvision.transforms): Preprocessing transforms for the model.
    """
    # Load pre-trained EfficientNetB2 weights and transforms
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()

    # Initialize the EfficientNetB2 model with pre-trained weights
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Freeze the base parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Replace the classifier with a custom one for the specified number of classes
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),  # Add dropout for regularization
        nn.Linear(in_features=1408, out_features=num_classes)  # Linear layer for classification
    )

    return model, transforms
