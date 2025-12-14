import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from Transformer import CNNTransformerModel, Config

def preprocess_image(image_path: str, cfg: Config) -> tuple[torch.Tensor, np.ndarray]:
    """Preprocess image to match training parameters"""
    transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])  # Adapted for chest X-rays
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image.resize(cfg.IMG_SIZE))
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, original_image

def get_attention_weights(model: CNNTransformerModel, image_tensor: torch.Tensor, cfg: Config) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Extract attention weights with shape debugging"""
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(cfg.DEVICE))
    
    # Get attention weights stored in the model
    attention_weights_list = model.transformer_encoder.attention_weights
    
    # Debug: Print attention weight shapes
    if attention_weights_list:
        print("\nDebug: Attention weights shape information")
        print(f"Total layers: {len(attention_weights_list)}")
        print(f"Last layer attention shape: {attention_weights_list[-1].shape}")
    else:
        print("\nDebug: No attention weights found in model")
    
    return attention_weights_list, output

def draw_attention_heatmap(
    original_image: np.ndarray,
    attention_weights_list: list[torch.Tensor],
    output: torch.Tensor,
    cfg: Config,
    image_path: str
) -> None:
    """Generate attention heatmap overlay (with dimension adaptation)"""
    if not attention_weights_list or output is None:
        print("⚠️ Cannot generate heatmap - missing attention weights or model output")
        return
    
    # Debug: Show complete attention structure
    print("\nDebug: Processing attention weights:")
    print(f"Number of layers in attention list: {len(attention_weights_list)}")
    print(f"Shape of first layer weights: {attention_weights_list[0].shape}")

    # Calculate prediction metrics
    pred_prob = output.item()
    pred_class = "pneumonia" if pred_prob > 0.5 else "normal"
    confidence = pred_prob if pred_class == "pneumonia" else 1 - pred_prob

    # Get last layer attention and debug its shape
    last_layer_attn = attention_weights_list[-1]
    print(f"Debug: Last layer attention shape before processing: {last_layer_attn.shape}")

    # Adapt to different possible dimensions
    if len(last_layer_attn.shape) == 3:
        # Handle shape: [num_heads, seq_len, seq_len]
        avg_attn = last_layer_attn.mean(dim=0)  # Average over heads
    elif len(last_layer_attn.shape) == 4:
        # Handle shape: [batch, num_heads, seq_len, seq_len]
        avg_attn = last_layer_attn.mean(dim=1).squeeze(0)  # Average over heads and remove batch
    else:
        # Handle unexpected shape
        print(f"⚠️ Unexpected attention shape: {last_layer_attn.shape}")
        # Try to flatten heads dimension
        avg_attn = last_layer_attn.view(-1, last_layer_attn.shape[-2], last_layer_attn.shape[-1]).mean(dim=0)

    print(f"Debug: Attention shape after averaging: {avg_attn.shape}")

    # Extract spatial attention (handle 1D/2D cases)
    try:
        if len(avg_attn.shape) == 2:
            # Standard case: [seq_len, seq_len]
            if avg_attn.shape[0] == 50:  # 1 class token + 49 spatial tokens
                class_token_attn = avg_attn[0, 1:]  # Ignore class token self-attention
            else:
                class_token_attn = avg_attn.mean(dim=0)  # Fallback averaging
        elif len(avg_attn.shape) == 1:
            # Handle 1D case: [seq_len]
            if avg_attn.shape[0] == 50:
                class_token_attn = avg_attn[1:]  # Skip class token
            else:
                class_token_attn = avg_attn  # Use as-is
        else:
            raise ValueError(f"Unsupported attention dimension: {len(avg_attn.shape)}")

        # Reshape to spatial dimensions (7x7)
        spatial_attn = class_token_attn.view(7, 7).cpu().numpy()
        print("Debug: Successfully reshaped attention to 7x7")

    except Exception as e:
        print(f"❌ Error processing attention weights: {str(e)}")
        print(f"Current attention shape: {avg_attn.shape}")
        # Create fallback heatmap to avoid complete failure
        spatial_attn = np.ones((7, 7))
        print("⚠️ Using fallback uniform attention map")

    # Upsample to original image size
    resize_attn = transforms.Resize(cfg.IMG_SIZE)(
        torch.tensor(spatial_attn).unsqueeze(0)
    ).squeeze(0).numpy()

    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Original image with prediction
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f"Original Image\nPrediction: {pred_class}\nConfidence: {confidence:.2%}", fontsize=12)
    plt.axis('off')
    
    # Heatmap overlay
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    heatmap = plt.imshow(resize_attn, cmap='jet', alpha=0.5)
    plt.title("Attention Heatmap (Red = High Importance)", fontsize=12)
    plt.axis('off')
    plt.colorbar(heatmap, shrink=0.6)
    
    # Save results
    save_dir = os.path.join(cfg.RESULTS_DIR, "attention_heatmaps")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"heatmap_{os.path.basename(image_path)}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Heatmap generation complete!")
    print(f"📊 Prediction: {pred_class} (Confidence: {confidence:.2%})")
    print(f"🖼️ Heatmap saved to: {save_path}")

def main():
    """Main function to run inference and generate attention heatmap"""
    # Initialize configuration
    cfg = Config()
    print(f"🔧 Configuration loaded: Device={cfg.DEVICE}, Image size={cfg.IMG_SIZE}")
    
    # Load trained model
    model = CNNTransformerModel(cfg).to(cfg.DEVICE)
    model_path = r"D:\AIsclepius\pneumonia_classifier\models\checkpoints\best_model_20251011-140607.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        print(f"📦 Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return
    
    # Target image path
    target_image_path = r"D:\AIsclepius\pneumonia_classifier\try\NORMAL2-IM-1427-0001.jpeg"
    if not os.path.exists(target_image_path):
        print(f"❌ Image not found: {target_image_path}")
        return
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(target_image_path, cfg)
    print(f"🖼️ Image preprocessing completed: Size {cfg.IMG_SIZE}")
    
    # Get attention weights and predictions
    attention_weights, output = get_attention_weights(model, image_tensor, cfg)
    
    # Generate heatmap
    draw_attention_heatmap(original_image, attention_weights, output, cfg, target_image_path)

if __name__ == "__main__":
    main()
