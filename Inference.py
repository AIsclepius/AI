# import tkinter as tk
# from tkinter import filedialog, messagebox, ttk
import gradio as gr # web graphics
import torch
import numpy as np
from PIL import Image #, ImageTk
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

CURRENT_MODEL = None
CURRENT_CONFIG = None

HAS_CUDA = torch.cuda.is_available()
HAS_IPEX = False

try:
    import intel_extension_for_pytorch as ipex
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        HAS_IPEX = True
except ImportError:
    pass
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
        'analysis_time': "Analysis Time: {}",
        'error': "Error: {}",
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
        'analysis_time': "分析时间：{}",
        'error': "错误：{}",
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
        if HAS_CUDA:
            self.DEVICE = torch.device('cuda')
        elif HAS_IPEX:
            self.DEVICE = torch.device('xpu')
        else:
            self.DEVICE = torch.device('cpu') # CPU by default
        
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

def load_model_logic(file_obj, lang_code):
    global CURRENT_MODEL, CURRENT_CONFIG
    lang = LANGUAGES[lang_code]
    
    if file_obj is None:
        return lang['error'].format("No file uploaded")

    try:
        cfg = Config()
        model = CNNTransformerModel(cfg).to(cfg.DEVICE)
        
        checkpoint = torch.load(file_obj.name, map_location=cfg.DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        
        CURRENT_MODEL = model
        CURRENT_CONFIG = cfg
        
        return lang['model_loaded'].format(file_obj.name, cfg.DEVICE)
    except Exception as e:
        return lang['load_failed'].format(str(e))

def run_inference_logic(image, lang_code):
    global CURRENT_MODEL, CURRENT_CONFIG
    lang = LANGUAGES[lang_code]
    
    if CURRENT_MODEL is None:
        return lang['error'].format("Please load a model first"), None
    
    if image is None:
        return lang['error'].format("Please upload an image"), None

    try:
        transform = transforms.Compose([
            transforms.Resize(CURRENT_CONFIG.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=CURRENT_CONFIG.MEAN, std=CURRENT_CONFIG.STD)
        ])
        input_tensor = transform(image).unsqueeze(0).to(CURRENT_CONFIG.DEVICE)

        with torch.no_grad():
            output = CURRENT_MODEL(input_tensor)
            pred_prob = output.item()
            
            if lang_code == 'en':
                pred_class = "Abnormal" if pred_prob > 0.5 else "Normal"
            else:
                pred_class = "异常" if pred_prob > 0.5 else "正常"
                
            confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        fig = plt.figure(figsize=(10, 5))
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(image)
        ax1.set_title(LANGUAGES[lang_code]['original_image'])
        ax1.axis('off')

        attention_weights = CURRENT_MODEL.transformer_encoder.attention_weights
        last_layer_attn = attention_weights[-1]
        avg_attn = last_layer_attn.mean(dim=0)
        
        if avg_attn.shape[0] == 50:
            class_token_attn = avg_attn[0, 1:]
        else:
            class_token_attn = avg_attn.mean(dim=0)
            
        spatial_attn = class_token_attn.view(7, 7).cpu().numpy()
        resize_attn = transforms.Resize(CURRENT_CONFIG.IMG_SIZE)(
            torch.tensor(spatial_attn).unsqueeze(0)
        ).squeeze(0).numpy()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(image.resize(CURRENT_CONFIG.IMG_SIZE))
        heatmap = ax2.imshow(resize_attn, cmap='jet', alpha=0.5)
        
        heatmap_title = f"Attention Heatmap - {pred_class}" if lang_code == 'en' else f"注意力热力图 - {pred_class}"
        ax2.set_title(heatmap_title)
        ax2.axis('off')
        fig.colorbar(heatmap, ax=ax2, shrink=0.6)

        result_str = (
            f"{lang['inference_result'].format(pred_class)}\n"
            f"{lang['confidence'].format(confidence)}\n"
            f"{lang['inference_time'].format(timestamp)}\n\n"
            f"{lang['note']}"
        )
        
        results_dir = os.path.join(os.getcwd(), "inference_results")
        os.makedirs(results_dir, exist_ok=True)
        filename_base = datetime.now().strftime("result_%Y%m%d_%H%M%S")
        
        fig.savefig(os.path.join(results_dir, f"{filename_base}.png"))
        with open(os.path.join(results_dir, f"{filename_base}.txt"), 'w', encoding='utf-8') as f:
            f.write(result_str)
            
        # final_msg = result_str + "\n\n" + lang['save_results'].format(results_dir)
        final_msg = '' # results are saved at the host
        
        return final_msg, fig

    except Exception as e:
        return lang['error'].format(str(e)), None

# ----------------------
# Dynamic Language Update Function
# ----------------------
def update_ui_language(lang_code):
    """Updates UI component labels and values based on selected language."""
    lang = LANGUAGES[lang_code]
    return (
        gr.update(value="# " + lang['app_title']),                          # header_md
        gr.update(value="### 1. " + lang['model_settings']),                # model_header_md
        gr.update(label=lang['model_path']),                                # model_file
        gr.update(value=lang['load_model']),                                # load_btn
        gr.update(value="### 2. " + lang['image_operations']),              # image_header_md
        gr.update(label=lang['load_image']),                                # img_input
        gr.update(value=lang['run_inference']),                             # run_btn
        gr.update(value="### " + lang['inference_results']),                # result_header_md
        gr.update(label=lang['inference_results']),                         # result_output (Using same key as header)
        gr.update(label=lang['image_display'])                              # plot_output
    )

# ----------------------
# Gradio Web Interface
# ----------------------
with gr.Blocks(title="Medical Image AI Inference") as demo:
    
    # Header - Initialized with default language (English)
    header_md = gr.Markdown("# " + LANGUAGES['en']['app_title'])
    
    with gr.Row():
        lang_dropdown = gr.Dropdown(choices=["en", "zh"], value="en", label="Language / 语言")
    
    with gr.Row():
        # Left Column: Controls
        with gr.Column(scale=1):
            model_header_md = gr.Markdown("### 1. " + LANGUAGES['en']['model_settings'])
            model_file = gr.File(label=LANGUAGES['en']['model_path'], file_types=[".pth"])
            load_btn = gr.Button(LANGUAGES['en']['load_model'], variant="secondary")
            load_status = gr.Textbox(label="System Status", interactive=False)
            
            image_header_md = gr.Markdown("### 2. " + LANGUAGES['en']['image_operations'])
            img_input = gr.Image(type="pil", label=LANGUAGES['en']['load_image'])
            run_btn = gr.Button(LANGUAGES['en']['run_inference'], variant="primary")
            
        # Right Column: Results
        with gr.Column(scale=2):
            result_header_md = gr.Markdown("### " + LANGUAGES['en']['inference_results'])
            result_output = gr.Textbox(label=LANGUAGES['en']['inference_results'], lines=6)
            plot_output = gr.Plot(label=LANGUAGES['en']['image_display'])

    # Event Handlers
    
    # 1. Language Change Handler - Updates all UI text
    lang_dropdown.change(
        fn=update_ui_language,
        inputs=lang_dropdown,
        outputs=[
            header_md, 
            model_header_md, 
            model_file, 
            load_btn, 
            image_header_md, 
            img_input, 
            run_btn, 
            result_header_md, 
            result_output, 
            plot_output
        ]
    )

    # 2. Logic Handlers
    load_btn.click(
        fn=load_model_logic, 
        inputs=[model_file, lang_dropdown], 
        outputs=load_status
    )
    
    run_btn.click(
        fn=run_inference_logic, 
        inputs=[img_input, lang_dropdown], 
        outputs=[result_output, plot_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7890)