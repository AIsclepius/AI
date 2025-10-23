import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
from copy import deepcopy  # For copying dataset objects to isolate transforms

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ----------------------
# Configuration Class (centralized hyperparameter management)
# ----------------------
class Config:
    """Model configuration parameters optimized for small datasets"""
    def __init__(self):
        # Data parameters
        self.IMG_SIZE = (224, 224)
        self.BATCH_SIZE = 16  # Smaller batch size for limited data
        self.VALIDATION_SPLIT = 0.2
        self.NUM_CLASSES = 1
        
        # Model parameters
        self.D_MODEL = 512
        self.NUM_HEADS = 8
        self.FFN_DIM = 1024
        self.NUM_LAYERS = 2  # Fewer layers to prevent overfitting
        self.DROPOUT_RATE = 0.2  # Slightly increased dropout for regularization
        self.FREEZE_CNN = True  # Critical for small datasets: preserve pre-trained features
        self.FINE_TUNE_LAYERS = 2  # Only unfreeze top 2 modules (optimal for <10k images)
        
        # Training parameters (conservative settings for small data)
        self.INIT_LR = 5e-6  # Lower initial learning rate
        self.EPOCHS = 20
        self.MIN_LR = 1e-7
        self.PATIENCE = 5  # Early stopping to prevent overfitting
        self.CHECKPOINT_DIR = '../models/checkpoints/'
        self.LOG_DIR = '../logs/'
        self.RESULTS_DIR = '../results/'
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # CheXNet weights path (user should download separately)
        self.CHEXNET_WEIGHTS_PATH = '../pretrained/model.pth.tar'
        
        # Create necessary directories
        for dir_path in [self.CHECKPOINT_DIR, self.LOG_DIR, self.RESULTS_DIR, os.path.dirname(self.CHEXNET_WEIGHTS_PATH)]:
            os.makedirs(dir_path, exist_ok=True)

# ----------------------
# Transformer Components
# ----------------------
class PositionalEncoding(nn.Module):
    """Positional encoding layer: adds positional information to sequence features"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)  # Non-trainable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder block: multi-head attention + feedforward network + residual connections"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # GELU activation function
        self.activation = nn.GELU()
        
        # Save attention weights for visualization
        self.attention_weights = None

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head self-attention
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask)
        self.attention_weights = attn_weights
        
        # Residual connection + normalization
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        # Residual connection + normalization
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class TransformerEncoder(nn.Module):
    """Transformer encoder: positional encoding + N encoder blocks"""
    def __init__(self, num_layers: int, d_model: int, nhead: int, 
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Create multiple encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Positional encoding
        src = self.pos_encoder(src)
        
        # Save attention weights from each layer
        self.attention_weights = []
        for layer in self.layers:
            src = layer(src)
            self.attention_weights.append(layer.attention_weights)
            
        return src

# ----------------------
# Data Processing Module
# ----------------------
class CustomImageDataset(datasets.ImageFolder):
    """Custom dataset class for handling potential single-channel chest X-ray images"""
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # Ensure image is 3-channel (required by pre-trained CNN)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
            
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return sample, target

def load_data(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and preprocess data with chest X-ray specific normalization"""
    # Chest X-ray normalization parameters (instead of ImageNet)
    chest_xray_mean = [0.485, 0.485, 0.485]  # Extended to 3 channels
    chest_xray_std = [0.229, 0.229, 0.229]
    
    # Training set data augmentation (moderate for small datasets)
    train_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.RandomHorizontalFlip(),  # Simple flip for augmentation
        transforms.RandomRotation(15),  # Gentle rotation to avoid distorting anatomy
        transforms.ToTensor(),
        transforms.Normalize(mean=chest_xray_mean, std=chest_xray_std)
    ])
    
    # Validation and test sets use only essential preprocessing
    val_test_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=chest_xray_mean, std=chest_xray_std)
    ])
    
    # Dataset paths
    train_dir = '../data/train'
    test_dir = '../data/test'
    
    # Load full training set with training transform
    full_train_dataset = CustomImageDataset(
        root=train_dir,
        transform=train_transform
    )
    
    # Create a deep copy of the full training dataset for validation
    # This ensures training and validation sets have independent transform configurations
    val_base_dataset = deepcopy(full_train_dataset)
    
    # Calculate split sizes
    total_size = len(full_train_dataset)
    val_size = int(cfg.VALIDATION_SPLIT * total_size)
    train_size = total_size - val_size
    
    # Generate and shuffle indices for splitting
    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_indices = indices[val_size:]  # Indices for training set
    val_indices = indices[:val_size]    # Indices for validation set
    
    # Create subsets using the shuffled indices
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)
    
    # Set appropriate transform for validation set (no augmentation)
    val_dataset.dataset.transform = val_test_transform
    
    # Load test set with validation/test transform
    test_dataset = CustomImageDataset(
        root=test_dir,
        transform=val_test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Validate data loading
    print("\n===== Data Loading Validation =====")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Class mapping: {full_train_dataset.class_to_idx}")
    
    return train_loader, val_loader, test_loader

# ----------------------
# CNN+Transformer Hybrid Model (Optimized for Small Datasets)
# ----------------------
class CNNTransformerModel(nn.Module):
    """CNN+Transformer hybrid model using CheXNet's DenseNet121, optimized for small datasets"""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        # 1. Load CheXNet's DenseNet121 as feature extractor
        self.base_model = models.densenet121(pretrained=False)  # Do not load ImageNet weights
        # Load CheXNet pretrained weights (pre-trained on chest X-rays)
        #try:
        #    self.base_model.load_state_dict(torch.load(cfg.CHEXNET_WEIGHTS_PATH, map_location=cfg.DEVICE))
        #    print(f"Successfully loaded CheXNet weights from {cfg.CHEXNET_WEIGHTS_PATH}")
        #except FileNotFoundError:
        #    print(f"Warning: CheXNet weights not found at {cfg.CHEXNET_WEIGHTS_PATH}. Using randomly initialized DenseNet121.")
        try:
            checkpoint = torch.load(cfg.CHEXNET_WEIGHTS_PATH, map_location=cfg.DEVICE)
            state_dict = checkpoint['state_dict']
            new_state_dict = {}

            for k, v in state_dict.items():
                key = k.replace('module.', '').replace('densenet121.', '')
                key = key.replace('norm.1', 'norm1').replace('conv.1', 'conv1')
                key = key.replace('norm.2', 'norm2').replace('conv.2', 'conv2')
                if not key.startswith('classifier'):
                    new_state_dict[key] = v

            self.base_model.load_state_dict(new_state_dict, strict=False)
            print(f"Successfully loaded CheXNet weights from {cfg.CHEXNET_WEIGHTS_PATH}")
        except FileNotFoundError:
            print(f"Warning: CheXNet weights not found at {cfg.CHEXNET_WEIGHTS_PATH}. Using randomly initialized DenseNet121.")
        except KeyError as e:
            print(f"Warning: Mismatched keys when loading weights: {e}. Trying to load with strict=False.")
            self.base_model.load_state_dict(new_state_dict, strict=False)



        # Extract feature layers (remove classification head)
        self.cnn_features = self.base_model.features
        
        # Store CNN layer names for visualization/debugging
        self.cnn_layer_names = [name for name, _ in self.cnn_features.named_children()]
        print(f"DenseNet121 feature layers: {self.cnn_layer_names}")
        
        # 2. Configure fine-tuning strategy (critical for small datasets)
        if cfg.FREEZE_CNN:
            # First freeze all layers to preserve pre-trained knowledge
            for param in self.cnn_features.parameters():
                param.requires_grad = False
            
            # Get list of feature submodules
            features_children = list(self.cnn_features.children())
            total_modules = len(features_children)
            
            # Unfreeze only top N modules (configured in Config)
            if 0 < cfg.FINE_TUNE_LAYERS <= total_modules:
                # Calculate indices for top modules
                unfreeze_start_idx = total_modules - cfg.FINE_TUNE_LAYERS
                unfreeze_indices = range(unfreeze_start_idx, total_modules)
                unfreeze_modules = [features_children[i] for i in unfreeze_indices]
                unfreeze_module_names = [self.cnn_layer_names[i] for i in unfreeze_indices]
                
                # Unfreeze target modules
                for module in unfreeze_modules:
                    for param in module.parameters():
                        param.requires_grad = True
                
                print(f"Unfrozen top {cfg.FINE_TUNE_LAYERS} modules for fine-tuning: {unfreeze_module_names}")
            else:
                print(f"Invalid FINE_TUNE_LAYERS={cfg.FINE_TUNE_LAYERS}, using fully frozen CNN")
        else:
            # Unfreeze all layers (not recommended for small datasets)
            for param in self.cnn_features.parameters():
                param.requires_grad = True
            print("All CNN layers are unfrozen (not recommended for small datasets)")
        
        # 3. Feature projection layer: adapts DenseNet121 output (1024-d) to Transformer dimension
        self.feature_projection = nn.Sequential(
            nn.Linear(1024, cfg.D_MODEL),
            nn.LayerNorm(cfg.D_MODEL)
        )
        
        # 4. Learnable class token for classification
        self.class_token = nn.Parameter(torch.randn(1, 1, cfg.D_MODEL))
        
        # 5. Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            num_layers=cfg.NUM_LAYERS,
            d_model=cfg.D_MODEL,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.FFN_DIM,
            dropout=cfg.DROPOUT_RATE
        )
        
        # 6. Classification head (smaller for limited data)
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.D_MODEL, cfg.D_MODEL),  # Fuse class token and global features
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(cfg.D_MODEL, cfg.NUM_CLASSES)
        )
        
        # Global average pooling for spatial feature aggregation
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. CNN feature extraction
        # Input shape: (batch_size, 3, H, W)
        cnn_feat = self.cnn_features(x)  # Output: (batch_size, 1024, 7, 7) (DenseNet121 output)
        
        # 2. Convert to sequence format
        batch_size = cnn_feat.shape[0]
        seq_len = cnn_feat.shape[2] * cnn_feat.shape[3]  # 7*7=49 spatial positions
        cnn_seq = cnn_feat.flatten(2).transpose(1, 2)  # Shape: (batch_size, 49, 1024)
        
        # 3. Project features to Transformer dimension
        projected_seq = self.feature_projection(cnn_seq)  # Shape: (batch_size, 49, d_model)
        
        # 4. Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, d_model)
        transformer_input = torch.cat([class_tokens, projected_seq], dim=1)  # Shape: (batch_size, 50, d_model)
        
        # 5. Transformer processing (requires seq_len first format)
        transformer_input = transformer_input.transpose(0, 1)  # Shape: (50, batch_size, d_model)
        transformer_feat = self.transformer_encoder(transformer_input)  # Shape: (50, batch_size, d_model)
        transformer_feat = transformer_feat.transpose(0, 1)  # Shape: (batch_size, 50, d_model)
        
        # 6. Extract classification features
        class_token_output = transformer_feat[:, 0, :]  # Shape: (batch_size, d_model)
        
        # 7. Aggregate spatial features
        spatial_feat = transformer_feat[:, 1:, :]  # Shape: (batch_size, 49, d_model)
        spatial_feat = spatial_feat.transpose(1, 2).view(batch_size, self.cfg.D_MODEL, 7, 7)  # Shape: (batch_size, d_model, 7, 7)
        global_spatial_feat = self.global_avg_pool(spatial_feat).flatten(1)  # Shape: (batch_size, d_model)
        
        # 8. Fuse features and classify
        combined_feat = torch.cat([class_token_output, global_spatial_feat], dim=1)  # Shape: (batch_size, 2*d_model)
        output = self.classifier(combined_feat)  # Shape: (batch_size, num_classes)
        
        return torch.sigmoid(output)  # Sigmoid for binary classification

# ----------------------
# Training and Evaluation Functions
# ----------------------
def log_gradient_distributions(model: nn.Module, writer: SummaryWriter, epoch: int):
    """Log gradient distributions of model parameters to TensorBoard"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(
                f'Gradients/{name}', 
                param.grad, 
                epoch
            )

def visualize_feature_maps(model: CNNTransformerModel, image: torch.Tensor, 
                          cfg: Config, writer: SummaryWriter, epoch: int, idx: int = 0):
    """Visualize CNN feature maps and log them to TensorBoard"""
    model.eval()
    img = image.unsqueeze(0).to(cfg.DEVICE)
    cnn_layers = list(model.cnn_features.children())
    total_layers = len(cnn_layers)
    
    # Select key feature layers (adapted to DenseNet structure)
    target_layers = [0, 3, 4, 6]
    target_layers = [idx for idx in target_layers if idx < total_layers]
    
    feature_maps = []
    hooks = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output.cpu().detach())
    
    # Register hooks for target layers
    for layer_idx in target_layers:
        layer = cnn_layers[layer_idx]
        hooks.append(layer.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(img)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Log feature maps
    for layer_idx, fm in zip(target_layers, feature_maps):
        feature_map = fm[0]
        num_maps = min(16, feature_map.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        
        for j in range(num_maps):
            axes[j].imshow(feature_map[j], cmap='viridis')
            axes[j].axis('off')
        
        # Hide unused subplots
        for j in range(num_maps, 16):
            axes[j].axis('off')
        
        plt.tight_layout()
        writer.add_figure(
            f'FeatureMaps/Layer_{layer_idx}/Sample_{idx}',
            fig,
            global_step=epoch
        )
        plt.close(fig)
    
    # Log original image (denormalized)
    img_np = image.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.485, 0.485])
    std = np.array([0.229, 0.229, 0.229])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    writer.add_image(
        f'InputImage/Sample_{idx}',
        img_np.transpose(2, 0, 1),
        global_step=epoch,
        dataformats='CHW'
    )
    model.train()


def train_epoch(model: nn.Module, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, 
                cfg: Config, writer: SummaryWriter, epoch: int) -> Dict[str, float]:
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{cfg.EPOCHS}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Log gradients periodically
        if batch_idx % 10 == 0:
            log_gradient_distributions(model, writer, epoch)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Accuracy": f"{correct/total:.4f}"})
    
    # Visualize feature maps for first image in last batch
    visualize_feature_maps(model, images[0].cpu(), cfg, writer, epoch)
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = correct / total
    
    # Log to TensorBoard
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
    
    return {"loss": epoch_loss, "accuracy": epoch_acc}

def validate(model: nn.Module, val_loader: DataLoader, 
             criterion: nn.Module, cfg: Config, writer: SummaryWriter, epoch: int) -> Dict[str, float]:
    """Validate model performance"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{cfg.EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Update metrics
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Accuracy": f"{correct/total:.4f}"})
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(val_loader)
    epoch_acc = correct / total
    
    # Calculate precision and recall
    from sklearn.metrics import precision_score, recall_score
    epoch_precision = precision_score(all_labels, all_preds)
    epoch_recall = recall_score(all_labels, all_preds)
    
    # Log to TensorBoard
    writer.add_scalar('Val/Loss', epoch_loss, epoch)
    writer.add_scalar('Val/Accuracy', epoch_acc, epoch)
    writer.add_scalar('Val/Precision', epoch_precision, epoch)
    writer.add_scalar('Val/Recall', epoch_recall, epoch)
    
    return {
        "loss": epoch_loss, 
        "accuracy": epoch_acc,
        "precision": epoch_precision,
        "recall": epoch_recall
    }

def visualize_attention_weights(model: CNNTransformerModel, image: torch.Tensor, 
                               cfg: Config, idx: int = 0):
    """Visualize Transformer attention weights for interpretability"""
    model.eval()
    with torch.no_grad():
        img = image.unsqueeze(0).to(cfg.DEVICE)
        _ = model(img)
        attention_weights = model.transformer_encoder.attention_weights
        
        plt.figure(figsize=(16, 10))
        last_layer_attn = attention_weights[-1].cpu().numpy()
        num_heads = last_layer_attn.shape[0]
        
        # Visualize first 8 attention heads
        for i in range(min(8, num_heads)):
            plt.subplot(2, 4, i+1)
            attn_map = last_layer_attn[i, 0, 1:].reshape(7, 7)  # Exclude class token
            plt.imshow(attn_map, cmap='viridis')
            plt.title(f'Attention Head #{i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(cfg.RESULTS_DIR, f'attention_visualization_{idx}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Attention visualization saved to: {save_path}")

def evaluate_model(model: CNNTransformerModel, test_loader: DataLoader, cfg: Config):
    """Comprehensively evaluate model performance on test set"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE).float().unsqueeze(1)
            
            outputs = model(images)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Flatten arrays
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    
    # Calculate performance metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)
    
    print("\n===== Test Set Evaluation Results =====")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Classification report
    print("\n===== Classification Report =====")
    print(classification_report(all_labels, all_preds, target_names=['normal', 'pneumonia']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['normal', 'pneumonia'],
                yticklabels=['normal', 'pneumonia'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(cfg.RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    roc_path = os.path.join(cfg.RESULTS_DIR, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved to: {roc_path}")
    
    # Visualize attention for random samples
    sample_indices = np.random.choice(len(test_loader.dataset), min(3, len(test_loader.dataset)))
    for i, idx in enumerate(sample_indices):
        image, _ = test_loader.dataset[idx]
        visualize_attention_weights(model, image, cfg, i)

# ----------------------
# Main Function: Training and Evaluation Pipeline
# ----------------------
def main():
    cfg = Config()
    print("===== Model Configuration =====")
    for key, value in cfg.__dict__.items():
        print(f"{key}: {value}")
    
    # Load and preprocess data
    train_loader, val_loader, test_loader = load_data(cfg)
    
    # Initialize model
    model = CNNTransformerModel(cfg).to(cfg.DEVICE)
    print("\n===== Model Structure =====")
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=cfg.INIT_LR)  # Lower LR for small data
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3, 
        min_lr=cfg.MIN_LR, verbose=True
    )
    
    # Initialize TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, timestamp))
    
    # Training loop variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None
    
    # Training history tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'lr': []
    }
    
    print("\n===== Starting Training =====")
    for epoch in range(cfg.EPOCHS):
        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, cfg, writer, epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        
        # Validation phase
        val_metrics = validate(model, val_loader, criterion, cfg, writer, epoch)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS}")
        print(f"Training: Loss={train_metrics['loss']:.4f}, Accuracy={train_metrics['accuracy']:.4f}")
        print(f"Validation: Loss={val_metrics['loss']:.4f}, Accuracy={val_metrics['accuracy']:.4f}, "
              f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Early stopping and best model tracking
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_weights = model.state_dict()
            patience_counter = 0
            best_model_path = os.path.join(cfg.CHECKPOINT_DIR, f"best_model_{timestamp}.pth")
            torch.save(best_model_weights, best_model_path)
            print(f"Saved best model to: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"Early stopping triggered! Stopping training at epoch {epoch+1}")
                break
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    # Plot training curves
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='Learning Rate')
    plt.title('Learning Rate Changes')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    curve_path = os.path.join(cfg.RESULTS_DIR, 'training_curves.png')
    plt.savefig(curve_path)
    plt.close()
    print(f"Training curves saved to: {curve_path}")
    
    # Final evaluation on test set
    evaluate_model(model, test_loader, cfg)
    
    # Save final model
    final_model_path = os.path.join(cfg.CHECKPOINT_DIR, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
