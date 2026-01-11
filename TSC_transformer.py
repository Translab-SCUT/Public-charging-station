import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.trial import TrialState
import warnings
warnings.filterwarnings("ignore")

folder = r""
all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.xlsx')]
dfs = [pd.read_excel(file) for file in all_files]
df = pd.concat(dfs, ignore_index=True)

def parse_power_column(power_list):
    try:
        power_vals = eval(power_list)
        if isinstance(power_vals, list):
            return np.mean(power_vals) if power_vals else 0
        return 0
    except:
        return 0

df['dc_power_avg'] = df['dc_power'].apply(parse_power_column)
df['ac_power_avg'] = df['ac_power'].apply(parse_power_column)

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['day_part'] = pd.cut(df['hour'], bins=[0, 6, 11, 14, 18, 24], labels=[0, 1, 2, 3, 4])

for col in ['cnt', 'env', 'ser', 'price', 'gdp', 'people_density', 'station_density', 
            'pile_density', 'people', 'all_cnt', 'dc_cnt', 'ac_cnt']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['zones', 'scope'])
zone_dummies = pd.get_dummies(df['zones'], prefix='zone')
scope_dummies = pd.get_dummies(df['scope'], prefix='scope')
df = pd.concat([df, zone_dummies, scope_dummies], axis=1)

time_feature_cols = ['hour', 'day_of_week', 'is_weekend', 'day_part']

spatial_feature_cols = ['lng', 'lat', 'station_density', 'pile_density']
spatial_feature_cols += list(zone_dummies.columns) + list(scope_dummies.columns)

context_feature_cols = [
    'price', 'showtag_num', 'gdp', 'people_density',
    'people', 'all_cnt', 'dc_cnt', 'ac_cnt',
    'dc_power_avg', 'ac_power_avg'
]

all_feature_cols = time_feature_cols + spatial_feature_cols + context_feature_cols

model_data = df.dropna(subset=all_feature_cols + ['use'])
X = model_data[all_feature_cols]
y = model_data['use']

time_dim = len(time_feature_cols)
spatial_dim = len(spatial_feature_cols)
context_dim = len(context_feature_cols)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_time = X_scaled[:, :time_dim]
X_spatial = X_scaled[:, time_dim:time_dim+spatial_dim]
X_context = X_scaled[:, time_dim+spatial_dim:]

indices = range(len(X_scaled))
train_indices, test_indices, _, _ = train_test_split(indices, indices, test_size=0.2, random_state=42)
train_indices, val_indices, _, _ = train_test_split(train_indices, train_indices, test_size=0.25, random_state=42)

X_train, X_val, X_test = X_scaled[train_indices], X_scaled[val_indices], X_scaled[test_indices]
X_train_time, X_val_time, X_test_time = X_time[train_indices], X_time[val_indices], X_time[test_indices]
X_train_spatial, X_val_spatial, X_test_spatial = X_spatial[train_indices], X_spatial[val_indices], X_spatial[test_indices]
X_train_context, X_val_context, X_test_context = X_context[train_indices], X_context[val_indices], X_context[test_indices]

y_train, y_val, y_test = y.values[train_indices], y.values[val_indices], y.values[test_indices]

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

X_train_time_tensor = torch.FloatTensor(X_train_time)
X_val_time_tensor = torch.FloatTensor(X_val_time)
X_test_time_tensor = torch.FloatTensor(X_test_time)

X_train_spatial_tensor = torch.FloatTensor(X_train_spatial)
X_val_spatial_tensor = torch.FloatTensor(X_val_spatial)
X_test_spatial_tensor = torch.FloatTensor(X_test_spatial)

X_train_context_tensor = torch.FloatTensor(X_train_context)
X_val_context_tensor = torch.FloatTensor(X_val_context)
X_test_context_tensor = torch.FloatTensor(X_test_context)

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

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, head_dim=16, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        
        self.q_proj = nn.Linear(input_dim, self.total_dim)
        self.k_proj = nn.Linear(input_dim, self.total_dim)
        self.v_proj = nn.Linear(input_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
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
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class GroupedFeatureEmbedding(nn.Module):
    def __init__(self, time_dim, spatial_dim, context_dim, embedding_dim, dropout=0.1):
        super().__init__()
        
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
        
        self.gate_time = nn.Linear(embedding_dim, embedding_dim)
        self.gate_spatial = nn.Linear(embedding_dim, embedding_dim)
        self.gate_context = nn.Linear(embedding_dim, embedding_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x_time, x_spatial, x_context):
        time_emb = self.time_embedding(x_time)
        spatial_emb = self.spatial_embedding(x_spatial)
        context_emb = self.context_embedding(x_context)
        
        gate_t = torch.sigmoid(self.gate_time(time_emb))
        gate_s = torch.sigmoid(self.gate_spatial(spatial_emb))
        gate_c = torch.sigmoid(self.gate_context(context_emb))
        
        time_gated = time_emb * gate_t
        spatial_gated = spatial_emb * gate_s
        context_gated = context_emb * gate_c
        
        combined = torch.cat([time_gated, spatial_gated, context_gated], dim=1)
        fused = self.fusion_layer(combined)
        
        return fused

class TSCTransformer(nn.Module):
    def __init__(self, time_dim, spatial_dim, context_dim, num_layers=3, num_heads=8,
                 head_dim=16, ff_dim=256, dropout=0.1, embedding_dim=128):
        super().__init__()
        
        self.feature_embedding = GroupedFeatureEmbedding(
            time_dim, spatial_dim, context_dim, embedding_dim, dropout
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, head_dim, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(embedding_dim, 1)
        
    def forward(self, x_time, x_spatial, x_context):
        batch_size = x_time.size(0)
        
        feature_emb = self.feature_embedding(x_time, x_spatial, x_context)
        feature_emb = feature_emb.unsqueeze(1)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, feature_emb], dim=1)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        cls_output = x[:, 0]
        output = self.classifier(cls_output)
        
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    num_layers = trial.suggest_int('num_layers', 2, 4)
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    head_dim = trial.suggest_categorical('head_dim', [8, 16, 32])
    ff_dim = trial.suggest_categorical('ff_dim', [128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    
    train_loader_trial = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader_trial = DataLoader(val_dataset, batch_size=batch_size)
    
    model = TSCTransformer(
        time_dim=time_dim,
        spatial_dim=spatial_dim,
        context_dim=context_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        ff_dim=ff_dim,
        dropout=dropout,
        embedding_dim=embedding_dim
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    num_epochs_trial = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    best_val_targets_np = None
    best_val_outputs_np = None
    
    for epoch in range(num_epochs_trial):
        model.train()
        for batch in train_loader_trial:
            x_time = batch['X_time'].to(device)
            x_spatial = batch['X_spatial'].to(device)
            x_context = batch['X_context'].to(device)
            targets = batch['y'].to(device)
            
            outputs = model(x_time, x_spatial, x_context)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        all_val_targets = []
        all_val_outputs = []
        
        with torch.no_grad():
            for batch in val_loader_trial:
                x_time = batch['X_time'].to(device)
                x_spatial = batch['X_spatial'].to(device)
                x_context = batch['X_context'].to(device)
                targets = batch['y'].to(device)
                
                outputs = model(x_time, x_spatial, x_context)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                
                all_val_targets.extend(targets.cpu().numpy())
                all_val_outputs.extend(outputs.cpu().numpy())
        
        val_loss = val_loss / len(val_loader_trial.dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_val_targets_np = np.array(all_val_targets).reshape(-1)
            best_val_outputs_np = np.array(all_val_outputs).reshape(-1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    best_mse = mean_squared_error(best_val_targets_np, best_val_outputs_np)
    best_mae = mean_absolute_error(best_val_targets_np, best_val_outputs_np)
    best_r2 = r2_score(best_val_targets_np, best_val_outputs_np)
    
    return best_val_loss

study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)
study.optimize(objective, n_trials=30, timeout=None, show_progress_bar=True)

best_params = study.best_params

best_batch_size = best_params['batch_size']
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_batch_size)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size)

model = TSCTransformer(
    time_dim=time_dim,
    spatial_dim=spatial_dim,
    context_dim=context_dim,
    num_layers=best_params['num_layers'],
    num_heads=best_params['num_heads'],
    head_dim=best_params['head_dim'],
    ff_dim=best_params['ff_dim'],
    dropout=best_params['dropout'],
    embedding_dim=best_params['embedding_dim']
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        x_time = batch['X_time'].to(device)
        x_spatial = batch['X_spatial'].to(device)
        x_context = batch['X_context'].to(device)
        targets = batch['y'].to(device)
        
        outputs = model(x_time, x_spatial, x_context)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * targets.size(0)
    
    return running_loss / len(train_loader.dataset)

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

num_epochs = 300
best_val_loss = float('inf')
early_stopping_patience = 30
early_stopping_counter = 0
best_model_state = None

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, _, _ = evaluate(model, val_loader, criterion, device)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= early_stopping_patience:
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

_, y_train_true, y_train_pred = evaluate(model, train_loader, criterion, device)
_, y_val_true, y_val_pred = evaluate(model, val_loader, criterion, device)
_, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, device)

train_r2 = r2_score(y_train_true, y_train_pred)
train_mse = mean_squared_error(y_train_true, y_train_pred)
train_mae = mean_absolute_error(y_train_true, y_train_pred)

val_r2 = r2_score(y_val_true, y_val_pred)
val_mse = mean_squared_error(y_val_true, y_val_pred)
val_mae = mean_absolute_error(y_val_true, y_val_pred)

test_r2 = r2_score(y_test_true, y_test_pred)
test_mse = mean_squared_error(y_test_true, y_test_pred)
test_mae = mean_absolute_error(y_test_true, y_test_pred)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': scaler,
    'time_dim': time_dim,
    'spatial_dim': spatial_dim,
    'context_dim': context_dim,
    'best_hyperparameters': best_params,
    'best_val_loss_from_optuna': study.best_value,
    'train_metrics': {'r2': train_r2, 'mse': train_mse, 'mae': train_mae},
    'val_metrics': {'r2': val_r2, 'mse': val_mse, 'mae': val_mae},
    'test_metrics': {'r2': test_r2, 'mse': test_mse, 'mae': test_mae}
}, 'TSC_Transformer_model.pth')