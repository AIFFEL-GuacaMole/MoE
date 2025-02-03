import argparse
import os.path as osp
from tdc.single_pred import ADME
from rdkit import Chem
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
import torch
import torch.nn as nn
import pandas as pd
import dgl
from performer_pytorch import SelfAttention as PerformerAttention
import torch_geometric.transforms as T
from typing import Any, Dict, Optional
from torch.nn import LayerNorm
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, average_precision_score
import numpy as np
import wandb

def smiles_to_graph(smiles: str, label: float) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("transform mol none")
    node_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.long).view(-1, 1)

    edges = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
        bond_type = int(bond.GetBondTypeAsDouble())  # 정수로 변환
        edge_attrs.append(bond_type)
        edge_attrs.append(bond_type)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 1)

    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class SmilesDataset(Dataset):
    def __init__(self, data, transform=None, pre_transform=None):
        super().__init__('.', transform, pre_transform)
        # Preprocess DataFrame into a list of tuples: (smiles, label)
        self.data = [(row['Drug'], row['Y']) for _, row in data.iterrows()]
        self.transform = transform
        self.pre_transform = pre_transform

    def len(self):
        return len(self.data)

    def get(self, idx):
        smiles, label = self.data[idx]
        graph = smiles_to_graph(smiles, label)
        
        if self.pre_transform:
            graph = self.pre_transform(graph)

        # PE가 (N, 1)인 경우 (N, 20)으로 확장
        if graph.pe.shape[1] == 1:
            graph.pe = graph.pe.expand(-1, 20)  

        if self.transform:
            graph = self.transform(graph)

        return graph
        
# Calculate max_node_value from dataset
def calculate_max_node_value(dataset):
    max_value = 0
    for data in dataset:
        max_value = max(max_value, data.x.max().item())
    return max_value

def calculate_accuracy(y_true, y_pred):
    # 이진 분류 기준으로 0.5 임계값 적용
    y_pred_labels = (y_pred > 0.5).astype(int)
    return accuracy_score(y_true, y_pred_labels)

def calculate_auprc(y_true, y_pred):
    # average_precision_score은 AUPRC 계산
    return average_precision_score(y_true, y_pred)

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # 예측 확률
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.redraw_projection.redraw_projections()
        out = model(data.x, data.pe, data.edge_index, data.edge_attr,data.batch)
        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        y_true.append(data.y.cpu().numpy())
        y_pred.append(torch.sigmoid(out).detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    accuracy = calculate_accuracy(y_true, y_pred)
    auprc = calculate_auprc(y_true, y_pred)

    return total_loss / len(loader.dataset), accuracy, auprc

@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            model.redraw_projection.redraw_projections()
            out = model(data.x, data.pe, data.edge_index, data.edge_attr,data.batch)
            loss = criterion(out.squeeze(), data.y)
            total_loss += loss.item() * data.num_graphs
            y_true.append(data.y.cpu().numpy())
            y_pred.append(torch.sigmoid(out).cpu().numpy()) 

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    accuracy = calculate_accuracy(y_true, y_pred)
    auprc = calculate_auprc(y_true, y_pred)

    return total_loss / len(loader.dataset) , accuracy, auprc 

class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any], max_node_value: int):
        super().__init__()

        self.node_emb = Embedding(max_node_value + 1, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(20, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),  # 이진 분류를 위해 1개의 값 출력
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, pe, edge_index, edge_attr, batch):
        try:
            x = x.squeeze(-1).long()
            
            x_pe = self.pe_norm(pe)
            
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
            
            edge_attr = self.edge_emb(edge_attr).squeeze(-2)
            
            for conv in self.convs:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)
            
            x = global_add_pool(x, batch)
            return self.mlp(x)

        except Exception as e:
            print(f"Error during forward pass: {e}")
            print(f"x shape: {x.shape if isinstance(x, torch.Tensor) else 'List of ' + str(len(x))}, "
                f"pe shape: {pe.shape if isinstance(pe, torch.Tensor) else 'List of ' + str(len(pe))}, "
                f"edge_index shape: {edge_index.shape if isinstance(edge_index, torch.Tensor) else 'List of ' + str(len(edge_index))}, "
                f"edge_attr shape: {edge_attr.shape if isinstance(edge_attr, torch.Tensor) else 'List of ' + str(len(edge_attr))}, "
                f"batch shape: {batch.shape if isinstance(batch, torch.Tensor) else 'List of ' + str(len(batch))}")
            raise e

class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            if fast_attentions:
                for fast_attention in fast_attentions:
                    fast_attention.redraw_projection_matrix()
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return 
        self.num_last_redraw += 1

def main():
    sweep_config = {
    "method": "grid",  # Grid Search 방식
    "metric": {
        "name": "val_auprc",  # 최적화 대상 메트릭
        "goal": "maximize"    # 목표: 최대화
    },
    "parameters": {
        "batch_size": {
            "values": [32, 64, 128]  # 배치 크기
        },
        "learning_rate": {
            "values": [1e-2, 5e-3, 1e-3]  # 학습률
        },
        "hidden_dim": {
            "values": [128, 256, 512]  # 은닉층 크기
        },
        "dropout_rate": {
            "values": [0.1, 0.2, 0.3]  # 드롭아웃 비율
        },
        "epochs": {
            "values": [50, 100]  # 에포크 수
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config,'CYP2C9_Veith_graphgps')
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 로드
    split = 'scaffold'
    data = ADME(name='CYP2C9_Veith')
    split_data = data.get_split(method=split)

    train_data, valid_data, test_data = split_data['train'], split_data['valid'], split_data['test']
    
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

    train_dataset = SmilesDataset(train_data, pre_transform=transform)
    valid_dataset = SmilesDataset(valid_data, pre_transform=transform)
    test_dataset = SmilesDataset(test_data, pre_transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--attn_type', default='multihead',
    help="Global attention type such as 'multihead' or 'performer'.")
    args = parser.parse_args()

    attn_kwargs = {'dropout': 0.5}
    max_node_value = calculate_max_node_value(train_dataset)

    model = GPS(channels=64, pe_dim=8, num_layers=10, attn_type=args.attn_type,
            attn_kwargs=attn_kwargs, max_node_value=max_node_value).to(device)
    
    ## pretrain model load 
    checkpoint = torch.load('./148.ckpt')
    state_dict = checkpoint['model_state']
    model.load_state_dict(state_dict, strict = False)
    
    # 모델 가중치 출력
    print("Loaded weights in the model:")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Shape: {param.shape}")

    global criterion
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)

     # Sweep agent
    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config

            # Model initialization
            model = GPS(
                channels=config.hidden_dim,
                pe_dim=8,
                num_layers=10,
                attn_type='multihead',
                attn_kwargs={'dropout': config.dropout_rate},
                max_node_value=max_node_value
            ).to(device)

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

            global criterion
            criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

            for epoch in range(config.epochs):
                train_loss, train_acc, train_auprc = train_one_epoch(model, train_loader, optimizer)
                val_loss, val_acc, val_auprc = validate(model, valid_loader)

                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_auprc": train_auprc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_auprc": val_auprc
                })
                print(f"Epoch {epoch + 1}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUPRC: {train_auprc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUPRC: {val_auprc:.4f}")

    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    main()