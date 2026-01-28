import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ---------- 数据加载 ----------
folder = r"XX"
all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.xlsx')]
dfs = [pd.read_excel(file) for file in all_files]
df = pd.concat(dfs, ignore_index=True)

# ---------- 功率列解析 ----------
def parse_power_column(power_list):
    try:
        power_vals = eval(power_list)
        if isinstance(power_vals, list):
            return np.mean(power_vals) if power_vals else 0
    except:
        return np.nan
    return np.nan

df['dc_power_avg'] = df['dc_power'].apply(parse_power_column)
df['ac_power_avg'] = df['ac_power'].apply(parse_power_column)

# ---------- 时间特征增强 ----------
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek  # 星期几 (0-6, 0是周一)
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # 是否周末
df['day_part'] = pd.cut(df['hour'], 
                       bins=[0, 6, 11, 14, 18, 24], 
                       labels=[0, 1, 2, 3, 4])  # 一天分为凌晨、上午、中午、下午、晚上

# ---------- 数值列转换 ----------
for col in ['cnt', 'env', 'ser', 'price', 'gdp', 'people_density', 'station_density', 
            'pile_density', 'people', 'all_cnt', 'dc_cnt', 'ac_cnt']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ---------- One-hot 编码 ----------
df = df.dropna(subset=['zones', 'scope'])
zone_dummies = pd.get_dummies(df['zones'], prefix='zone')
scope_dummies = pd.get_dummies(df['scope'], prefix='scope')
df = pd.concat([df, zone_dummies, scope_dummies], axis=1)

# ---------- 特征分组 ----------
# 时间特征
time_feature_cols = ['hour', 'day_of_week', 'is_weekend', 'day_part']

# 空间特征
spatial_feature_cols = ['lng', 'lat', 'station_density', 'pile_density']
spatial_feature_cols += list(zone_dummies.columns) + list(scope_dummies.columns)

# 上下文特征
context_feature_cols = [
    'price', 'showtag_num', 'gdp', 'people_density',
    'people', 'all_cnt', 'dc_cnt', 'ac_cnt',
    'dc_power_avg', 'ac_power_avg'
]

# 合并所有特征
all_feature_cols = time_feature_cols + spatial_feature_cols + context_feature_cols

# ---------- 数据准备 ----------
model_data = df.dropna(subset=all_feature_cols + ['use'])
X = model_data[all_feature_cols]
y = model_data['use']

# 记录各组特征的维度，用于后续模型构建
time_dim = len(time_feature_cols)
spatial_dim = len(spatial_feature_cols)
context_dim = len(context_feature_cols)

# ---------- 数据预处理：标准化 ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 重新组织特征，便于后续模型处理
X_time = X_scaled[:, :time_dim]
X_spatial = X_scaled[:, time_dim:time_dim+spatial_dim]
X_context = X_scaled[:, time_dim+spatial_dim:]

# ---------- 数据划分：训练、验证、测试 ----------
indices = range(len(X_scaled))
train_indices, test_indices, _, _ = train_test_split(indices, indices, test_size=0.2, random_state=42)
train_indices, val_indices, _, _ = train_test_split(train_indices, train_indices, test_size=0.25, random_state=42)

X_train, X_val, X_test = X_scaled[train_indices], X_scaled[val_indices], X_scaled[test_indices]
X_train_time, X_val_time, X_test_time = X_time[train_indices], X_time[val_indices], X_time[test_indices]
X_train_spatial, X_val_spatial, X_test_spatial = X_spatial[train_indices], X_spatial[val_indices], X_spatial[test_indices]
X_train_context, X_val_context, X_test_context = X_context[train_indices], X_context[val_indices], X_context[test_indices]

y_train, y_val, y_test = y.values[train_indices], y.values[val_indices], y.values[test_indices]

# ---------- 转换为PyTorch张量 ----------
# 全特征张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# 分组特征张量
X_train_time_tensor = torch.FloatTensor(X_train_time)
X_val_time_tensor = torch.FloatTensor(X_val_time)
X_test_time_tensor = torch.FloatTensor(X_test_time)

X_train_spatial_tensor = torch.FloatTensor(X_train_spatial)
X_val_spatial_tensor = torch.FloatTensor(X_val_spatial)
X_test_spatial_tensor = torch.FloatTensor(X_test_spatial)

X_train_context_tensor = torch.FloatTensor(X_train_context)
X_val_context_tensor = torch.FloatTensor(X_val_context)
X_test_context_tensor = torch.FloatTensor(X_test_context)

# ---------- 数据加载器 ----------
class EnhancedDataset(torch.utils.data.Dataset):
    def __init__(self, X_all, X_time, X_spatial, X_context, y):
        self.X_all = X_all
        self.X_time = X_time
        self.X_spatial = X_spatial
        self.X_context = X_context
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            'X_all': self.X_all[idx],
            'X_time': self.X_time[idx],
            'X_spatial': self.X_spatial[idx],
            'X_context': self.X_context[idx],
            'y': self.y[idx]
        }

batch_size = 256

train_dataset = EnhancedDataset(
    X_train_tensor, X_train_time_tensor, X_train_spatial_tensor, X_train_context_tensor, y_train_tensor
)
val_dataset = EnhancedDataset(
    X_val_tensor, X_val_time_tensor, X_val_spatial_tensor, X_val_context_tensor, y_val_tensor
)
test_dataset = EnhancedDataset(
    X_test_tensor, X_test_time_tensor, X_test_spatial_tensor, X_test_context_tensor, y_test_tensor
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ---------- 模型组件定义 ----------
class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, head_dim=16, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        
        # 线性投影层
        self.q_proj = nn.Linear(input_dim, self.total_dim)
        self.k_proj = nn.Linear(input_dim, self.total_dim)
        self.v_proj = nn.Linear(input_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # 线性投影并重塑为多头格式
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置以便进行注意力计算
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 注意力分数计算
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 重塑并投影回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.total_dim)
        output = self.out_proj(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads=8, head_dim=16, ff_dim=64, dropout=0.1):
        super().__init__()
        self.attention = MultiheadAttention(input_dim, num_heads, head_dim, dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 自注意力层 + 残差连接
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class GroupedFeatureEmbedding(nn.Module):
    def __init__(self, time_dim, spatial_dim, context_dim, embedding_dim, dropout=0.1):
        super().__init__()
        
        # 为每组特征创建专门的嵌入层
        self.time_embedding = nn.Sequential(
            nn.Linear(time_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.spatial_embedding = nn.Sequential(
            nn.Linear(spatial_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.context_embedding = nn.Sequential(
            nn.Linear(context_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 特征交互门控机制
        self.gate_time = nn.Linear(embedding_dim, embedding_dim)
        self.gate_spatial = nn.Linear(embedding_dim, embedding_dim)
        self.gate_context = nn.Linear(embedding_dim, embedding_dim)
        
        # 最终融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x_time, x_spatial, x_context):
        # 嵌入每组特征
        time_emb = self.time_embedding(x_time)
        spatial_emb = self.spatial_embedding(x_spatial)
        context_emb = self.context_embedding(x_context)
        
        # 简单的特征交互 - 门控机制
        gate_t = torch.sigmoid(self.gate_time(time_emb))
        gate_s = torch.sigmoid(self.gate_spatial(spatial_emb))
        gate_c = torch.sigmoid(self.gate_context(context_emb))
        
        time_gated = time_emb * gate_t
        spatial_gated = spatial_emb * gate_s
        context_gated = context_emb * gate_c
        
        # 连接特征并通过融合层
        combined = torch.cat([time_gated, spatial_gated, context_gated], dim=1)
        fused = self.fusion_layer(combined)
        
        # 返回融合后的嵌入
        return fused

class EnhancedFTTransformer(nn.Module):
    def __init__(self, time_dim, spatial_dim, context_dim, num_layers=3, num_heads=8,
                 head_dim=16, ff_dim=256, dropout=0.1, embedding_dim=128):
        super().__init__()
        
        # 分组特征嵌入层
        self.feature_embedding = GroupedFeatureEmbedding(
            time_dim, spatial_dim, context_dim, embedding_dim, dropout
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Transformer 层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, head_dim, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.classifier = nn.Linear(embedding_dim, 1)
        
    def forward(self, x_time, x_spatial, x_context):
        batch_size = x_time.size(0)
        
        # 分组特征嵌入
        feature_emb = self.feature_embedding(x_time, x_spatial, x_context)
        feature_emb = feature_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embedding_dim]
        x = torch.cat([cls_tokens, feature_emb], dim=1)  # [batch_size, 2, embedding_dim]
        
        # 应用Transformer层
        for block in self.transformer_blocks:
            x = block(x)
        
        # 使用CLS token进行预测
        cls_output = x[:, 0]
        output = self.classifier(cls_output)
        
        return output

# ---------- 模型、损失函数和优化器 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = EnhancedFTTransformer(
    time_dim=time_dim,
    spatial_dim=spatial_dim,
    context_dim=context_dim,
    num_layers=3,
    num_heads=8,
    head_dim=16,
    ff_dim=256,
    dropout=0.1,
    embedding_dim=128
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ---------- 训练函数 ----------
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        # 获取数据和标签
        x_time = batch['X_time'].to(device)
        x_spatial = batch['X_spatial'].to(device)
        x_context = batch['X_context'].to(device)
        targets = batch['y'].to(device)
        
        # 前向传播
        outputs = model(x_time, x_spatial, x_context)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * targets.size(0)
    
    return running_loss / len(train_loader.dataset)

# ---------- 评估函数 ----------
def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in data_loader:
            x_time = batch['X_time'].to(device)
            x_spatial = batch['X_spatial'].to(device)
            x_context = batch['X_context'].to(device)
            targets = batch['y'].to(device)
            
            outputs = model(x_time, x_spatial, x_context)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * targets.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    all_targets = np.array(all_targets).reshape(-1)
    all_outputs = np.array(all_outputs).reshape(-1)
    
    return running_loss / len(data_loader.dataset), all_targets, all_outputs

# ---------- 训练循环 ----------
num_epochs = 300
best_val_loss = float('inf')
early_stopping_patience = 30
early_stopping_counter = 0
best_model_state = None

train_losses = []
val_losses = []

print("开始训练增强型FT-Transformer模型...")
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, _, _ = evaluate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # 学习率调整
    scheduler.step(val_loss)
    
    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        best_model_state = model.state_dict().copy()
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} *')
    else:
        early_stopping_counter += 1
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    if early_stopping_counter >= early_stopping_patience:
        print(f'提前停止于 epoch {epoch+1}')
        break

# 加载最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# ---------- 最终评估 ----------
_, y_train_true, y_train_pred = evaluate(model, train_loader, criterion, device)
_, y_val_true, y_val_pred = evaluate(model, val_loader, criterion, device)
_, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, device)

# ---------- 数值指标函数 ----------
def print_metrics(y_true, y_pred, label="Set"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n【{label} 评估指标】") 
    print(f"  R²  : {r2:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  MAE : {mae:.4f}")

print_metrics(y_train_true, y_train_pred, label="训练集")
print_metrics(y_val_true, y_val_pred, label="验证集")
print_metrics(y_test_true, y_test_pred, label="测试集")

# ---------- 可视化：损失曲线 ----------
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('增强型FT-Transformer: 训练与验证损失')

# ---------- 可视化：测试集预测 vs 实际 ----------
plt.subplot(2, 1, 2)
plt.scatter(y_test_true, y_test_pred, c=abs(y_test_true - y_test_pred), cmap='viridis', alpha=0.6)
plt.plot([min(y_test_true), max(y_test_true)], [min(y_test_true), max(y_test_true)], 'r--', lw=2, label='理想预测线')
plt.xlabel('实际使用率')
plt.ylabel('预测使用率')
plt.title('测试集: 预测值 vs 实际值')
plt.colorbar(label='绝对误差')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('enhanced_ft_transformer_results.png')
plt.show()

# ---------- 保存模型 ----------
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': scaler,
    'time_dim': time_dim,
    'spatial_dim': spatial_dim,
    'context_dim': context_dim,
}, 'enhanced_ft_transformer_model.pth')

print("增强型FT-Transformer模型训练完成并已保存！") 