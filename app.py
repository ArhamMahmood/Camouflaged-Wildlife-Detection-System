import os
import numpy as np
import spaces
import torch
import gradio as gr

from PIL import Image
from models.unet import ResNet34UNet


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pth"
)

IMAGE_SIZE = 352

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
).reshape(1, 1, 3)

STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
).reshape(1, 1, 3)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading ResNet34 U-Net...")

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

model = ResNet34UNet(pretrained=False)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print("Configured device:", DEVICE)
print("Checkpoint epoch:", checkpoint.get("epoch", "Unknown"))


# ==========================================
# IMAGE PREDICTION
# ==========================================

@spaces.GPU(duration=30)
def predict_image(image, threshold):
    """Generate probability map, binary mask, and detection overlay."""

    if image is None:
        raise gr.Error("Please upload an image first.")

    try:
        # Convert input to PIL RGB
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        original_width, original_height = image.size

        # Preserve original image for overlay
        original_image = np.asarray(image, dtype=np.uint8)

        # Resize for model input
        resized_image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # Convert to NumPy
        image_array = np.asarray(
            resized_image,
            dtype=np.float32
        ) / 255.0

        # ImageNet normalization
        image_array = (image_array - MEAN) / STD

        # HWC -> CHW and add batch dimension
        input_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        ).unsqueeze(0).float()

        input_tensor = input_tensor.to(DEVICE)

        # Model inference
        with torch.no_grad():
            prediction = model(input_tensor)

            # Handle model output
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]

            probability = torch.sigmoid(prediction)

        # Convert probability to NumPy
        probability = probability.squeeze().float().cpu().numpy()

        # Resize probability map to original dimensions
        probability_image = Image.fromarray(
            np.clip(probability * 255, 0, 255).astype(np.uint8)
        )

        probability_image = probability_image.resize(
            (original_width, original_height),
            Image.Resampling.BILINEAR
        )

        probability_array = np.asarray(
            probability_image,
            dtype=np.uint8
        )

        # Create RGB probability map
        probability_map = np.stack(
            [probability_array] * 3,
            axis=-1
        )

        # Generate binary mask
        binary_mask = (
            probability_array >= int(threshold * 255)
        ).astype(np.uint8) * 255

        binary_mask_rgb = np.stack(
            [binary_mask] * 3,
            axis=-1
        )

        # Create red detection overlay
        overlay = original_image.astype(np.float32).copy()

        detected_pixels = binary_mask > 0

        red_color = np.array(
            [255, 0, 0],
            dtype=np.float32
        )

        alpha = 0.45

        overlay[detected_pixels] = (
            (1 - alpha) * overlay[detected_pixels]
            + alpha * red_color
        )

        overlay = np.clip(
            overlay, 0, 255
        ).astype(np.uint8)

        # Return three RGB images
        return (
            np.ascontiguousarray(probability_map),
            np.ascontiguousarray(binary_mask_rgb),
            np.ascontiguousarray(overlay)
        )

    except gr.Error:
        raise

    except Exception as error:
        print("Prediction error:", repr(error))
        raise gr.Error(
            f"Prediction failed: {str(error)}"
        )


# ==========================================
# GRADIO INTERFACE
# ==========================================

with gr.Blocks(
    title="Camouflaged Wildlife Detection System"
) as demo:

    gr.Markdown(
        """
        # Camouflaged Wildlife Detection System

        ### AI-powered wildlife detection using ResNet34 U-Net

        Upload an image to detect camouflaged wildlife.
        Adjust the threshold to control detection sensitivity.
        """
    )

    with gr.Row():

        with gr.Column():
            input_image = gr.Image(
                label="Upload Wildlife Image",
                type="pil"
            )

            threshold = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Detection Threshold"
            )

            predict_button = gr.Button(
                "Detect Wildlife",
                variant="primary"
            )

        with gr.Column():
            probability_output = gr.Image(
                label="Probability Map"
            )

            mask_output = gr.Image(
                label="Binary Detection Mask"
            )

            overlay_output = gr.Image(
                label="Detection Overlay"
            )

    predict_button.click(
        fn=predict_image,
        inputs=[
            input_image,
            threshold
        ],
        outputs=[
            probability_output,
            mask_output,
            overlay_output
        ]
    )

    gr.Markdown(
        """
        ---
        **Research Demonstration:** This system is designed
        for research and wildlife detection assistance.
        Predictions should be independently verified.
        """
    )


# ==========================================
# LAUNCH APPLICATION
# ==========================================

demo.queue()

if __name__ == "__main__":
    demo.launch()
