import torchxrayvision as xrv
import skimage
import torch
import torchvision
from captum.attr import IntegratedGradients
import numpy as np
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model outside the function to avoid redundant loading
try:
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(DEVICE)
    model.eval()  # Set model to evaluation mode
    logger.info("Model loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {str(e)}")
    raise
except torch.cuda.OutOfMemoryError as e:
    logger.error(f"CUDA out of memory: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Error loading model: {str(e)}", exc_info=True)
    raise


def load_and_preprocess_image(image_bytes: io.BytesIO) -> torch.Tensor:
    """Load and preprocess the image for the model."""
    try:
        img = skimage.io.imread(image_bytes)
        img = xrv.datasets.normalize(img, 255)
        img = img.mean(2)[None, ...]
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        img = transform(img)
        img = torch.from_numpy(img).float().to(DEVICE)
        logger.info("Image loaded and preprocessed successfully")
        return img
    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        raise


def get_predictions(img: torch.Tensor) -> dict:
    """Get predictions from the model."""
    try:
        with torch.no_grad():
            outputs = model(img[None, ...])
        predictions = dict(zip(model.pathologies, outputs[0].cpu().detach().numpy()))
        logger.info("Predictions generated successfully")
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


def compute_attributions(img: torch.Tensor, target_class: int) -> torch.Tensor:
    """Compute attributions using Integrated Gradients."""
    try:
        ig = IntegratedGradients(model)
        attributions = ig.attribute(img[None, ...], target=target_class)
        logger.info("Attributions computed successfully")
        return attributions
    except Exception as e:
        logger.error(f"Error during attribution computation: {str(e)}")
        raise


def visualize_attribution(img: torch.Tensor, attributions: torch.Tensor, top_pathology: str) -> str:
    """Visualize and return the attributions as a base64 string."""
    try:
        img_np = img.cpu().numpy().squeeze()
        attributions_np = attributions.cpu().numpy().squeeze()

        if img_np.shape != (224, 224):
            raise ValueError(f"Expected image shape (224, 224), but got {img_np.shape}")
        if attributions_np.shape != (224, 224):
            raise ValueError(f"Expected attribution shape (224, 224), but got {attributions_np.shape}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_np, cmap="gray")
        ax.imshow(attributions_np, cmap="jet", alpha=0.5)
        ax.set_title(f"Attribution for {top_pathology}")
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        logger.info("Attribution visualized successfully")
        return base64_image
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        raise