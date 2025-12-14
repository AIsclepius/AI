import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

# ----------------------
# Language Support
# ----------------------
LANGUAGES = {
    'en': {
        'app_title': "Medical Image AI Inference System",
        'control_panel': "Control Panel",
        'model_settings': "Model Settings",
        'model_path': "Model Path:",
        'browse': "Browse",
        'load_model': "Load Model",
        'image_operations': "Image Operations",
        'load_image': "Load Image",
        'run_inference': "Run Inference",
        'save_results': "Save Results",
        'inference_results': "Inference Results",
        'image_display': "Image Display",
        'original_image': "Original Image",
        'attention_heatmap': "Attention Heatmap",
        'heatmap_placeholder': "Attention Heatmap (shows after inference)",
        'model_not_exist': "Model file does not exist: {}",
        'model_loaded': "Model loaded successfully\nPath: {}\nDevice: {}",
        'load_success': "Success",
        'model_load_success': "Model loaded successfully",
        'model_load_failed': "Model Load Failed",
        'load_failed_msg': "Could not load model: {}",
        'image_load_failed': "Image Load Failed",
        'image_open_failed': "Could not open image: {}",
        'image_loaded': "Loaded image: {}",
        'warning': "Warning",
        'load_image_first': "Please load an image first",
        'load_model_first': "Please load a model first",
        'inference_failed': "Inference Failed",
        'analysis_error': "Error during analysis: {}",
        'inference_result': "Inference Result: {}",
        'confidence': "Confidence: {:.2%}",
        'inference_time': "Inference Time: {}",
        'note': "Note: This result is for reference only.\nFinal diagnosis should be confirmed by a professional physician.",
        'save_failed': "Save Failed",
        'could_not_save': "Could not save results: {}",
        'save_success': "Success",
        'results_saved': "Results saved to:\n{}",
        'results_saved_to': "Results saved to:\n{}",
        'image_path': "Image Path: {}",
        'analysis_time': "Analysis Time: {}"
    },
    'zh': {
        'app_title': "医学影像AI推理系统",
        'control_panel': "控制面板",
        'model_settings': "模型设置",
        'model_path': "模型路径：",
        'browse': "浏览",
        'load_model': "加载模型",
        'image_operations': "图像操作",
        'load_image': "加载图像",
        'run_inference': "运行推理",
        'save_results': "保存结果",
        'inference_results': "推理结果",
        'image_display': "图像显示",
        'original_image': "原始图像",
        'attention_heatmap': "注意力热力图",
        'heatmap_placeholder': "注意力热力图（推理后显示）",
        'model_not_exist': "模型文件不存在：{}",
        'model_loaded': "模型加载成功\n路径：{}\n设备：{}",
        'load_success': "成功",
        'model_load_success': "模型加载成功",
        'model_load_failed': "模型加载失败",
        'load_failed_msg': "无法加载模型：{}",
        'image_load_failed': "图像加载失败",
        'image_open_failed': "无法打开图像：{}",
        'image_loaded': "已加载图像：{}",
        'warning': "警告",
        'load_image_first': "请先加载图像",
        'load_model_first': "请先加载模型",
        'inference_failed': "推理失败",
        'analysis_error': "分析过程中出错：{}",
        'inference_result': "推理结果：{}",
        'confidence': "置信度：{:.2%}",
        'inference_time': "推理时间：{}",
        'note': "注意：此结果仅供参考。\n最终诊断应由专业医师确认。",
        'save_failed': "保存失败",
        'could_not_save': "无法保存结果：{}",
        'save_success': "成功",
        'results_saved': "结果已保存至：\n{}",
        'results_saved_to': "结果已保存至：\n{}",
        'image_path': "图像路径：{}",
        'analysis_time': "分析时间：{}"
    }
}

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

# Transformer components
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

# Main model
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
        self.current_lang = 'en'  # Default language
        self.lang = LANGUAGES[self.current_lang]
        
        # Set cross-platform font
        self.system_font = self.get_system_font()
        
        self.root.title(self.lang['app_title'])
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize configuration and model
        self.cfg = Config()
        self.model = None
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.inference_result = None  # Store inference result for language switching
        
        # Create necessary directories
        self.results_dir = os.path.join(os.getcwd(), "inference_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize UI
        self.init_ui()

    def get_system_font(self):
        """Get appropriate font for different operating systems"""
        system = os.name
        if system == 'nt':  # Windows
            return ("SimHei", 10)
        elif system == 'posix':
            # Check if it's macOS or Linux
            if os.path.exists('/System/Library/Fonts'):  # macOS
                return ("Heiti TC", 10)
            else:  # Linux
                return ("WenQuanYi Micro Hei", 10)
        else:
            return ("Arial Unicode MS", 10)

    def init_ui(self):
        """Initialize user interface"""
        # Header with language switch
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X)
        
        header = tk.Label(
            header_frame, 
            text=self.lang['app_title'], 
            font=(self.system_font[0], 16, "bold"),
            bg="#2c3e50", 
            fg="white",
            pady=10
        )
        header.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Language switch button
        self.lang_btn = ttk.Button(
            header_frame,
            text="中文" if self.current_lang == 'en' else "English",
            command=self.switch_language
        )
        self.lang_btn.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left control panel
        self.control_frame = ttk.LabelFrame(main_frame, text=self.lang['control_panel'], padding=10)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        
        # Model loading area
        self.model_frame = ttk.LabelFrame(self.control_frame, text=self.lang['model_settings'], padding=10)
        self.model_frame.pack(pady=10, fill=tk.X)
        
        self.model_path_var = tk.StringVar(value=os.path.join(os.getcwd(), "model.pth"))
        ttk.Label(self.model_frame, text=self.lang['model_path'], font=self.system_font).pack(anchor=tk.W, pady=5)
        
        # 将模型路径框架定义为实例属性（关键修复）
        self.model_path_frame = ttk.Frame(self.model_frame)
        self.model_path_frame.pack(fill=tk.X)
        
        self.model_path_entry = ttk.Entry(self.model_path_frame, textvariable=self.model_path_var, width=30)
        self.model_path_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # 保存浏览按钮的引用（方便后续更新文本）
        self.browse_btn = ttk.Button(
            self.model_path_frame, 
            text=self.lang['browse'], 
            command=self.browse_model
        )
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_model_btn = ttk.Button(
            self.model_frame, 
            text=self.lang['load_model'], 
            command=self.load_model
        )
        self.load_model_btn.pack(pady=10, fill=tk.X)
        
        # Image operation area
        self.image_frame = ttk.LabelFrame(self.control_frame, text=self.lang['image_operations'], padding=10)
        self.image_frame.pack(pady=10, fill=tk.X)
        
        self.load_image_btn = ttk.Button(
            self.image_frame, 
            text=self.lang['load_image'], 
            command=self.load_image
        )
        self.load_image_btn.pack(pady=5, fill=tk.X)
        
        self.infer_btn = ttk.Button(
            self.image_frame, 
            text=self.lang['run_inference'], 
            command=self.run_inference,
            state=tk.DISABLED
        )
        self.infer_btn.pack(pady=5, fill=tk.X)
        
        self.save_btn = ttk.Button(
            self.image_frame, 
            text=self.lang['save_results'], 
            command=self.save_results,
            state=tk.DISABLED
        )
        self.save_btn.pack(pady=5, fill=tk.X)
        
        # Results display area
        self.result_frame = ttk.LabelFrame(self.control_frame, text=self.lang['inference_results'], padding=10)
        self.result_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(self.result_frame, height=10, width=30, state=tk.DISABLED, font=self.system_font)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Right image display area
        self.display_frame = ttk.LabelFrame(main_frame, text=self.lang['image_display'], padding=10)
        self.display_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Create image display canvas with proper font
        plt.rcParams["font.family"] = [self.system_font[0], "sans-serif"]
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize image display
        self.axes[0].set_title(self.lang['original_image'])
        self.axes[0].axis('off')
        self.axes[1].set_title(self.lang['heatmap_placeholder'])
        self.axes[1].axis('off')
        self.canvas.draw()
        
        # Configure grid weights
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

    def switch_language(self):
        """Switch between English and Chinese"""
        self.current_lang = 'zh' if self.current_lang == 'en' else 'en'
        self.lang = LANGUAGES[self.current_lang]
        
        # Update UI elements
        self.root.title(self.lang['app_title'])
        self.lang_btn.config(text="中文" if self.current_lang == 'en' else "English")
        
        # Update frame labels
        self.control_frame.config(text=self.lang['control_panel'])
        self.model_frame.config(text=self.lang['model_settings'])
        self.image_frame.config(text=self.lang['image_operations'])
        self.result_frame.config(text=self.lang['inference_results'])
        self.display_frame.config(text=self.lang['image_display'])
        
        # Update labels
        for widget in self.model_frame.winfo_children():
            if isinstance(widget, ttk.Label) and widget['text'] in [LANGUAGES['en']['model_path'], LANGUAGES['zh']['model_path']]:
                widget.config(text=self.lang['model_path'])
        
        # Update buttons（修复核心：直接通过保存的按钮引用更新文本）
        self.browse_btn.config(text=self.lang['browse'])
        self.load_model_btn.config(text=self.lang['load_model'])
        self.load_image_btn.config(text=self.lang['load_image'])
        self.infer_btn.config(text=self.lang['run_inference'])
        self.save_btn.config(text=self.lang['save_results'])
        
        # Update plot titles
        self.axes[0].set_title(self.lang['original_image'])
        if hasattr(self, 'attention_heatmap_title'):
            self.axes[1].set_title(self.attention_heatmap_title)
        else:
            self.axes[1].set_title(self.lang['heatmap_placeholder'])
        self.canvas.draw()
        
        # Update result text if available
        if self.inference_result:
            pred_class, confidence, timestamp = self.inference_result
            result_text = (
                f"{self.lang['inference_result'].format(pred_class)}\n"
                f"{self.lang['confidence'].format(confidence)}\n"
                f"{self.lang['inference_time'].format(timestamp)}\n\n"
                f"{self.lang['note']}"
            )
            self.update_result(result_text)

    def browse_model(self):
        """Browse and select model file"""
        model_path = filedialog.askopenfilename(
            title=self.lang['model_settings'],
            filetypes=[("Model Files", "*.pth;*.pth.tar")]
        )
        if model_path:
            self.model_path_var.set(model_path)

    def load_model(self):
        """Load trained model"""
        model_path = self.model_path_var.get()
        if not os.path.exists(model_path):
            messagebox.showerror(self.lang['warning'], self.lang['model_not_exist'].format(model_path))
            return
            
        try:
            # Initialize model
            self.model = CNNTransformerModel(self.cfg).to(self.cfg.DEVICE)
            # Load model weights
            checkpoint = torch.load(model_path, map_location=self.cfg.DEVICE)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            # Update status
            self.update_result(self.lang['model_loaded'].format(model_path, self.cfg.DEVICE))
            messagebox.showinfo(self.lang['load_success'], self.lang['model_load_success'])
            
            # Enable inference button if image is loaded
            if self.original_image is not None:
                self.infer_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror(self.lang['model_load_failed'], self.lang['load_failed_msg'].format(str(e)))
            self.update_result(self.lang['load_failed_msg'].format(str(e)))

    def load_image(self):
        """Load and display image"""
        image_path = filedialog.askopenfilename(
            title=self.lang['image_operations'],
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
            self.axes[0].set_title(self.lang['original_image'])
            self.axes[0].axis('off')
            
            # Clear right image
            self.axes[1].clear()
            self.axes[1].set_title(self.lang['heatmap_placeholder'])
            self.axes[1].axis('off')
            
            self.canvas.draw()
            
            # Enable inference button
            if self.model is not None:
                self.infer_btn.config(state=tk.NORMAL)
            
            self.update_result(self.lang['image_loaded'].format(os.path.basename(image_path)))
            
        except Exception as e:
            messagebox.showerror(self.lang['image_load_failed'], self.lang['image_open_failed'].format(str(e)))

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
            messagebox.showwarning(self.lang['warning'], self.lang['load_image_first'])
            return
            
        if self.model is None:
            messagebox.showwarning(self.lang['warning'], self.lang['load_model_first'])
            return
            
        try:
            # Model inference
            with torch.no_grad():
                input_tensor = self.processed_image.to(self.cfg.DEVICE)
                output = self.model(input_tensor)
                pred_prob = output.item()
                # Handle class names for both languages
                if self.current_lang == 'en':
                    pred_class = "Abnormal" if pred_prob > 0.5 else "Normal"
                else:
                    pred_class = "异常" if pred_prob > 0.5 else "正常"
                confidence = pred_prob if pred_class in ["Abnormal", "异常"] else 1 - pred_prob
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Store result for language switching
            self.inference_result = (pred_class, confidence, timestamp)
            
            # Get attention weights and generate heatmap
            attention_weights = self.model.transformer_encoder.attention_weights
            self.generate_attention_heatmap(attention_weights, pred_class, confidence)
            
            # Display results
            result_text = (
                f"{self.lang['inference_result'].format(pred_class)}\n"
                f"{self.lang['confidence'].format(confidence)}\n"
                f"{self.lang['inference_time'].format(timestamp)}\n\n"
                f"{self.lang['note']}"
            )
            self.update_result(result_text)
            
            # Update buttons state
            self.infer_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror(self.lang['inference_failed'], self.lang['analysis_error'].format(str(e)))

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
        
        # Create heatmap title (store for language switching)
        if self.current_lang == 'en':
            self.attention_heatmap_title = f"Attention Heatmap - {pred_class} areas highlighted"
        else:
            self.attention_heatmap_title = f"注意力热力图 - 突出显示{pred_class}区域"
        
        # Display heatmap
        self.axes[1].clear()
        self.axes[1].imshow(self.original_image.resize(self.cfg.IMG_SIZE))
        heatmap = self.axes[1].imshow(resize_attn, cmap='jet', alpha=0.5)
        self.axes[1].set_title(self.attention_heatmap_title)
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
                f.write(self.lang['image_path'].format(self.image_path) + "\n")
                f.write(self.lang['analysis_time'].format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "\n\n")
                f.write(result_text)
            
            self.update_result(f"{self.result_text.get(1.0, tk.END)}\n\n{self.lang['results_saved_to'].format(img_save_path)}")
            messagebox.showinfo(self.lang['save_success'], self.lang['results_saved'].format(self.results_dir))
            
        except Exception as e:
            messagebox.showerror(self.lang['save_failed'], self.lang['could_not_save'].format(str(e)))

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
    root = tk.Tk()
    app = MedicalInferenceApp(root)
    root.mainloop()