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
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

IMAGE_SIZE = 352

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("Loading ResNet34 U-Net...")

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

model = ResNet34UNet(pretrained=False)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print("Device:", DEVICE)
print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))


# ==========================================
# IMAGE PREDICTION
# ==========================================

def predict_image(image, threshold):
    if image is None:
        raise gr.Error("Please upload an image first.")

    try:
        # Gradio input is configured as PIL, but handle NumPy input too.
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("RGB")
        original_width, original_height = image.size

        if original_width < 1 or original_height < 1:
            raise ValueError("The uploaded image has invalid dimensions.")

        # Resize and normalize exactly as in model preprocessing.
        resized_image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
        image_array = (image_array - MEAN) / STD
        image_array = np.transpose(image_array, (2, 0, 1)).copy()

        input_tensor = torch.from_numpy(image_array).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.sigmoid(logits)

        # Expected model output: [1, 1, 352, 352].
        if probabilities.ndim != 4 or probabilities.shape[0] != 1:
            raise ValueError(
                f"Unexpected model output shape: {tuple(probabilities.shape)}"
            )

        probability_array = probabilities[0, 0].detach().float().cpu().numpy()

        if not np.isfinite(probability_array).all():
            raise ValueError("The model produced NaN or infinite predictions.")

        probability_array = np.clip(probability_array, 0.0, 1.0)

        # Resize the probability map back to the uploaded image dimensions.
        probability_pil = Image.fromarray(
            np.round(probability_array * 255).astype(np.uint8)
        )
        probability_pil = probability_pil.resize(
            (original_width, original_height),
            Image.Resampling.BILINEAR
        )

        probability_gray = np.asarray(probability_pil, dtype=np.uint8)

        # Gradio image outputs are returned as explicit RGB uint8 arrays.
        probability_rgb = np.repeat(probability_gray[:, :, None], 3, axis=2)

        # Generate binary segmentation mask.
        binary_array = (
            (probability_gray.astype(np.float32) / 255.0) >= float(threshold)
        ).astype(np.uint8) * 255

        binary_rgb = np.repeat(binary_array[:, :, None], 3, axis=2)

        # Create a red overlay on the original image.
        original_array = np.asarray(image, dtype=np.uint8).copy()
        overlay_array = original_array.copy()
        foreground = binary_array > 0

        overlay_array[foreground] = (
            0.5 * original_array[foreground]
            + 0.5 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)

        # Ensure arrays are contiguous and have supported image dimensions.
        probability_rgb = np.ascontiguousarray(probability_rgb, dtype=np.uint8)
        binary_rgb = np.ascontiguousarray(binary_rgb, dtype=np.uint8)
        overlay_array = np.ascontiguousarray(overlay_array, dtype=np.uint8)

        print(
            "Prediction complete:",
            f"input={original_width}x{original_height},",
            f"outputs={probability_rgb.shape}, {binary_rgb.shape}, {overlay_array.shape}"
        )

        return probability_rgb, binary_rgb, overlay_array

    except gr.Error:
        raise
    except Exception as exc:
        print("Prediction error:", repr(exc))
        raise gr.Error(f"Prediction failed: {exc}") from exc


# ==========================================
# GRADIO INTERFACE
# ==========================================

with gr.Blocks(title="Camouflaged Wildlife Detection System") as demo:

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
                image_mode="RGB",
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
                type="numpy",
                image_mode="RGB",
                label="Prediction Probability Map"
            )

            mask_output = gr.Image(
                type="numpy",
                image_mode="RGB",
                label="Predicted Binary Mask"
            )

            overlay_output = gr.Image(
                type="numpy",
                image_mode="RGB",
                label="Detection Overlay"
            )

    predict_button.click(
        fn=predict_image,
        inputs=[input_image, threshold],
        outputs=[probability_output, mask_output, overlay_output]
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
