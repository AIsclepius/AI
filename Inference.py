import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import torchvision  # Added missing import
import torchvision.transforms as transforms
from datetime import datetime

# ----------------------
# Model Configuration and Definition
# ----------------------
class Config:
    """Configuration parameters for inference"""
    def __init__(self):
        # Image parameters (must match training settings)
        self.IMG_SIZE = (224, 224)
        self.NUM_CLASSES = 1
        
        # Model parameters (must match training settings)
        self.D_MODEL = 512
        self.NUM_HEADS = 8
        self.FFN_DIM = 1024
        self.NUM_LAYERS = 2
        self.DROPOUT_RATE = 0.2
        
        # Device configuration
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image preprocessing parameters
        self.MEAN = [0.485, 0.485, 0.485]
        self.STD = [0.229, 0.229, 0.229]

# Transformer components (only保留推理所需部分)
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()
        self.attention_weights = None

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src2, attn_weights = self.self_attn(src, src, src)
        self.attention_weights = attn_weights
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, 
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.pos_encoder(src)
        self.attention_weights = []
        for layer in self.layers:
            src = layer(src)
            self.attention_weights.append(layer.attention_weights)
        return src

# Main model (only inference-related parts保留)
class CNNTransformerModel(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        # Load base CNN model (DenseNet121)
        self.base_model = torchvision.models.densenet121(pretrained=False)
        self.cnn_features = self.base_model.features
        
        # Feature projection layer
        self.feature_projection = torch.nn.Sequential(
            torch.nn.Linear(1024, cfg.D_MODEL),
            torch.nn.LayerNorm(cfg.D_MODEL)
        )
        
        # Classification token
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, cfg.D_MODEL))
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            num_layers=cfg.NUM_LAYERS,
            d_model=cfg.D_MODEL,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.FFN_DIM,
            dropout=cfg.DROPOUT_RATE
        )
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * cfg.D_MODEL, cfg.D_MODEL),
            torch.nn.GELU(),
            torch.nn.Dropout(cfg.DROPOUT_RATE),
            torch.nn.Linear(cfg.D_MODEL, cfg.NUM_CLASSES)
        )
        
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        cnn_feat = self.cnn_features(x)  # (batch_size, 1024, 7, 7)
        
        # Convert to sequence format
        batch_size = cnn_feat.shape[0]
        seq_len = cnn_feat.shape[2] * cnn_feat.shape[3]  # 7*7=49
        cnn_seq = cnn_feat.flatten(2).transpose(1, 2)  # (batch_size, 49, 1024)
        
        # Feature projection
        projected_seq = self.feature_projection(cnn_seq)  # (batch_size, 49, d_model)
        
        # Add classification token
        class_tokens = self.class_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        transformer_input = torch.cat([class_tokens, projected_seq], dim=1)  # (batch_size, 50, d_model)
        
        # Transformer processing
        transformer_input = transformer_input.transpose(0, 1)  # (50, batch_size, d_model)
        transformer_feat = self.transformer_encoder(transformer_input)  # (50, batch_size, d_model)
        transformer_feat = transformer_feat.transpose(0, 1)  # (batch_size, 50, d_model)
        
        # Extract classification features
        class_token_output = transformer_feat[:, 0, :]  # (batch_size, d_model)
        
        # Aggregate spatial features
        spatial_feat = transformer_feat[:, 1:, :]  # (batch_size, 49, d_model)
        spatial_feat = spatial_feat.transpose(1, 2).view(batch_size, self.cfg.D_MODEL, 7, 7)
        global_spatial_feat = self.global_avg_pool(spatial_feat).flatten(1)  # (batch_size, d_model)
        
        # Feature fusion and classification
        combined_feat = torch.cat([class_token_output, global_spatial_feat], dim=1)
        output = self.classifier(combined_feat)
        
        return torch.sigmoid(output)

# ----------------------
# Inference Application Class
# ----------------------
class MedicalInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image AI Inference System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize configuration and model
        self.cfg = Config()
        self.model = None
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        
        # Create necessary directories
        self.results_dir = os.path.join(os.getcwd(), "inference_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        # Header
        header = tk.Label(
            self.root, 
            text="Medical Image AI Inference System", 
            font=("Arial", 16, "bold"),
            bg="#2c3e50", 
            fg="white",
            pady=10
        )
        header.pack(fill=tk.X)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding=10)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        
        # Model loading area
        model_frame = ttk.LabelFrame(control_frame, text="Model Settings", padding=10)
        model_frame.pack(pady=10, fill=tk.X)
        
        self.model_path_var = tk.StringVar(value="D:\\AIsclepius\\model.pth")
        ttk.Label(model_frame, text="Model Path:").pack(anchor=tk.W, pady=5)
        model_path_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=30)
        model_path_entry.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(
            model_frame, 
            text="Browse", 
            command=self.browse_model
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            model_frame, 
            text="Load Model", 
            command=self.load_model
        ).pack(pady=10, fill=tk.X)
        
        # Image operation area
        image_frame = ttk.LabelFrame(control_frame, text="Image Operations", padding=10)
        image_frame.pack(pady=10, fill=tk.X)
        
        ttk.Button(
            image_frame, 
            text="Load Image", 
            command=self.load_image
        ).pack(pady=5, fill=tk.X)
        
        self.infer_btn = ttk.Button(
            image_frame, 
            text="Run Inference", 
            command=self.run_inference,
            state=tk.DISABLED
        )
        self.infer_btn.pack(pady=5, fill=tk.X)
        
        self.save_btn = ttk.Button(  # Fixed button reference
            image_frame, 
            text="Save Results", 
            command=self.save_results,
            state=tk.DISABLED
        )
        self.save_btn.pack(pady=5, fill=tk.X)
        
        # Results display area
        result_frame = ttk.LabelFrame(control_frame, text="Inference Results", padding=10)
        result_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(result_frame, height=10, width=30, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Right image display area
        display_frame = ttk.LabelFrame(main_frame, text="Image Display", padding=10)
        display_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Create image display canvas
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize image display
        self.axes[0].set_title("Original Image")
        self.axes[0].axis('off')
        self.axes[1].set_title("Attention Heatmap")
        self.axes[1].axis('off')
        self.canvas.draw()
        
        # Configure grid weights
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

    def browse_model(self):
        """Browse and select model file"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pth;*.pth.tar")]
        )
        if model_path:
            self.model_path_var.set(model_path)

    def load_model(self):
        """Load trained model"""
        model_path = self.model_path_var.get()
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file does not exist: {model_path}")
            return
            
        try:
            # Initialize model
            self.model = CNNTransformerModel(self.cfg).to(self.cfg.DEVICE)
            # Load model weights
            checkpoint = torch.load(model_path, map_location=self.cfg.DEVICE)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            # Update status
            self.update_result(f"Model loaded successfully\nPath: {model_path}\nDevice: {self.cfg.DEVICE}")
            messagebox.showinfo("Success", "Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Model Load Failed", f"Could not load model: {str(e)}")
            self.update_result(f"Model load failed: {str(e)}")

    def load_image(self):
        """Load and display image"""
        image_path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[("Image Files", "*.jpeg;*.jpg;*.png;*.bmp")]
        )
        
        if not image_path:
            return
            
        try:
            # Save image path and original image
            self.image_path = image_path
            self.original_image = Image.open(image_path).convert('RGB')
            self.processed_image = self.preprocess_image()
            
            # Display original image
            self.axes[0].clear()
            self.axes[0].imshow(self.original_image)
            self.axes[0].set_title("Original Image")
            self.axes[0].axis('off')
            
            # Clear right image
            self.axes[1].clear()
            self.axes[1].set_title("Attention Heatmap (shows after inference)")
            self.axes[1].axis('off')
            
            self.canvas.draw()
            
            # Enable inference button
            if self.model is not None:
                self.infer_btn.config(state=tk.NORMAL)
            
            self.update_result(f"Loaded image: {os.path.basename(image_path)}")
            
        except Exception as e:
            messagebox.showerror("Image Load Failed", f"Could not open image: {str(e)}")

    def preprocess_image(self):
        """Preprocess image to match model input requirements"""
        transform = transforms.Compose([
            transforms.Resize(self.cfg.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.MEAN, std=self.cfg.STD)
        ])
        return transform(self.original_image).unsqueeze(0)  # Add batch dimension

    def run_inference(self):
        """Perform model inference and display results"""
        if not hasattr(self, 'processed_image') or self.processed_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
            
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first")
            return
            
        try:
            # Model inference
            with torch.no_grad():
                input_tensor = self.processed_image.to(self.cfg.DEVICE)
                output = self.model(input_tensor)
                pred_prob = output.item()
                pred_class = "Abnormal" if pred_prob > 0.5 else "Normal"
                confidence = pred_prob if pred_class == "Abnormal" else 1 - pred_prob
            
            # Get attention weights and generate heatmap
            attention_weights = self.model.transformer_encoder.attention_weights
            self.generate_attention_heatmap(attention_weights, pred_class, confidence)
            
            # Display results
            result_text = (
                f"Inference Result: {pred_class}\n"
                f"Confidence: {confidence:.2%}\n"
                f"Inference Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Note: This result is for reference only.\n"
                f"Final diagnosis should be confirmed by a professional physician."
            )
            self.update_result(result_text)
            
            # Enable save button
            self.infer_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Inference Failed", f"Error during analysis: {str(e)}")

    def generate_attention_heatmap(self, attention_weights, pred_class, confidence):
        """Generate and display attention heatmap"""
        if not attention_weights:
            return
            
        # Process attention weights
        last_layer_attn = attention_weights[-1]
        avg_attn = last_layer_attn.mean(dim=0)  # Average over all attention heads
        
        # Extract attention related to class token
        if avg_attn.shape[0] == 50:  # 1 class token + 49 spatial tokens
            class_token_attn = avg_attn[0, 1:]  # Exclude attention to class token itself
        else:
            class_token_attn = avg_attn.mean(dim=0)
            
        # Reshape to spatial dimensions and upsample
        spatial_attn = class_token_attn.view(7, 7).cpu().numpy()
        resize_attn = transforms.Resize(self.cfg.IMG_SIZE)(
            torch.tensor(spatial_attn).unsqueeze(0)
        ).squeeze(0).numpy()
        
        # Display heatmap
        self.axes[1].clear()
        self.axes[1].imshow(self.original_image.resize(self.cfg.IMG_SIZE))
        heatmap = self.axes[1].imshow(resize_attn, cmap='jet', alpha=0.5)
        self.axes[1].set_title(f"Attention Heatmap - {pred_class} areas highlighted")
        self.axes[1].axis('off')
        self.fig.colorbar(heatmap, ax=self.axes[1], shrink=0.6)
        
        self.canvas.draw()

    def save_results(self):
        """Save inference results and images"""
        if not hasattr(self, 'original_image') or self.original_image is None:
            return
            
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"result_{timestamp}"
            
            # Save result image
            img_save_path = os.path.join(self.results_dir, f"{base_name}.png")
            self.fig.savefig(img_save_path, dpi=300, bbox_inches='tight')
            
            # Save result text
            result_text = self.result_text.get(1.0, tk.END)
            text_save_path = os.path.join(self.results_dir, f"{base_name}.txt")
            with open(text_save_path, 'w', encoding='utf-8') as f:
                f.write(f"Image Path: {self.image_path}\n")
                f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(result_text)
            
            self.update_result(f"{self.result_text.get(1.0, tk.END)}\n\nResults saved to:\n{img_save_path}")
            messagebox.showinfo("Success", f"Results saved to:\n{self.results_dir}")
            
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save results: {str(e)}")

    def update_result(self, text):
        """Update result display area"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

# ----------------------
# Main Program Entry
# ----------------------
if __name__ == "__main__":
    # Ensure proper font rendering
    plt.rcParams["font.family"] = ["Arial", "sans-serif"]
    root = tk.Tk()
    app = MedicalInferenceApp(root)
    root.mainloop()
