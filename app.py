
import os
import torch
import gradio as gr
import numpy as np

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(1, 1, 3)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(1, 1, 3)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("Loading ResNet34 U-Net...")

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
print("Device:", DEVICE)
print("Checkpoint epoch:", checkpoint["epoch"])


# ==========================================
# IMAGE PREDICTION
# ==========================================

def predict_image(image, threshold):

    if image is None:
        raise gr.Error("Please upload an image first.")

    # Preserve original dimensions
    image = image.convert("RGB")

    original_width, original_height = image.size

    # Resize input
    resized_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    # Convert to NumPy
    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    ) / 255.0

    # Apply ImageNet normalization
    image_array = (image_array - MEAN) / STD

    # HWC to CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    # Convert to tensor
    input_tensor = torch.from_numpy(
        image_array.copy()
    ).unsqueeze(0).to(DEVICE)

    # Model inference
    with torch.inference_mode():

        logits = model(input_tensor)

        probabilities = torch.sigmoid(logits)

    probability_array = (
        probabilities.squeeze().cpu().numpy()
    )

    # Restore original image dimensions
    probability_image = Image.fromarray(
        np.clip(
            probability_array * 255,
            0,
            255
        ).astype(np.uint8)
    )

    probability_image = probability_image.resize(
        (original_width, original_height),
        Image.Resampling.BILINEAR
    )

    probability_array = (
        np.asarray(probability_image, dtype=np.float32) / 255.0
    )

    # Generate binary mask
    binary_array = (
        probability_array >= threshold
    ).astype(np.uint8) * 255

    binary_mask = Image.fromarray(
        binary_array,
        mode="L"
    )

    # Create red overlay
    original_array = np.asarray(
        image,
        dtype=np.uint8
    ).copy()

    overlay_array = original_array.copy()

    foreground = binary_array > 0

    red = np.zeros_like(original_array)
    red[:, :, 0] = 255

    overlay_array[foreground] = (
        0.5 * original_array[foreground]
        + 0.5 * red[foreground]
    ).astype(np.uint8)

    overlay = Image.fromarray(overlay_array)

    return probability_image, binary_mask, overlay


# ==========================================
# GRADIO INTERFACE
# ==========================================

with gr.Blocks(
    title="Camouflaged Wildlife Detection System"
) as demo:

    gr.Markdown(
        """
        # Camouflaged Wildlife Detection System

        Upload an image to generate a predicted camouflaged-object
        segmentation mask using a ResNet34 U-Net model.

        **Note:** This is an experimental research application.
        Predictions may be inaccurate and should not be treated
        as verified wildlife identification or legal evidence.
        """
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="Upload Wildlife Image"
            )

            threshold = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.50,
                step=0.01,
                label="Prediction Threshold"
            )

            predict_button = gr.Button(
                "Generate Prediction",
                variant="primary"
            )

        with gr.Column():

            probability_output = gr.Image(
                label="Prediction Probability Map"
            )

            mask_output = gr.Image(
                label="Predicted Binary Mask"
            )

            overlay_output = gr.Image(
                label="Detection Overlay"
            )

    predict_button.click(
        fn=predict_image,
        inputs=[input_image, threshold],
        outputs=[
            probability_output,
            mask_output,
            overlay_output
        ]
    )

    gr.Markdown(
        """
        ---
        **Model:** ResNet34 U-Net

        **Input resolution:** 352 × 352

        **Preprocessing:** ImageNet normalization

        **Purpose:** Research and experimental visualization only.
        """
    )


if __name__ == "__main__":
    demo.launch()
