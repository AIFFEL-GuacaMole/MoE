import os 
import torch
import torch.nn as nn
from gps_finetuning import GPS
from dgllife.model import load_pretrained
from sklearn.model_selection import train_test_split  
from torch.nn import BCELoss
from torch.optim import Adam
from tdc.benchmark_group import admet_group
from sklearn.metrics import precision_recall_curve, auc
import wandb
from molfeat.trans.pretrained import PretrainedDGLTransformer
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPS_CONFIG = {
    "channels": 64,
    "pe_dim": 20,
    "num_layers": 10,
    "attn_type": "multihead",
    "attn_kwargs": {"dropout": 0.5},
    "max_node_value": 100
}

def load_graph_models():
    global graphgps_model, gin_model, ginfomax_model

    try:
        graphgps_model = GPS(**GPS_CONFIG).to(device)
        checkpoint = torch.load("MoE/MoE/148.ckpt", map_location=device)
        state_dict = checkpoint['model_state']
        graphgps_model.load_state_dict(state_dict, strict=False)
        graphgps_model.eval()
        print("[DEBUG] GPS model loaded successfully.")

    except Exception as e:
        print(f"Warning: Failed to load GPS model. Error: {str(e)}")
        graphgps_model = None

def prepare_con_features(smiles_data):
    transformer = PretrainedDGLTransformer(kind='gin_supervised_contextpred')
    return transformer(smiles_data)

def prepare_info_features(smiles_data):
    transformer = PretrainedDGLTransformer(kind='gin_supervised_infomax')
    return transformer(smiles_data)

def prepare_edge_features(smiles_data):
    transformer = PretrainedDGLTransformer(kind='gin_supervised_edgepred')
    return transformer(smiles_data)

def prepare_mask_features(smiles_data):
    transformer = PretrainedDGLTransformer(kind='gin_supervised_masking')
    return transformer(smiles_data)

def load_or_generate_features(data, batch_size, dataset_name):
    """
    데이터셋 별로 캐시 파일을 생성하고 저장/불러오기
    """
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)  # 캐시 디렉토리 생성

    # 데이터셋 별로 캐시 파일 경로 설정
    con_cache_path = os.path.join(cache_dir, f"{dataset_name}_con_features.pt")
    info_cache_path = os.path.join(cache_dir, f"{dataset_name}_info_features.pt")
    edge_cache_path = os.path.join(cache_dir, f"{dataset_name}_edge_features.pt")
    mask_cache_path = os.path.join(cache_dir, f"{dataset_name}_mask_features.pt")

    #  만약 모든 캐시 파일이 존재하면 불러오기
    if all(os.path.exists(path) for path in [con_cache_path, info_cache_path, edge_cache_path, mask_cache_path]):
        print(f"Loading cached features for {dataset_name}...")

        con_features = torch.load(con_cache_path).to(device)
        info_features = torch.load(info_cache_path).to(device)
        edge_features = torch.load(edge_cache_path).to(device)
        mask_features = torch.load(mask_cache_path).to(device)

        return con_features, info_features, edge_features, mask_features

    #  없으면 새로 생성
    smiles = data["Drug"].tolist()

    # 🔹 GIN Context Features
    print(f"{dataset_name} - con_feature generating...")
    con_features = []
    for i in range(0, len(smiles), batch_size):
        batch_smiles = smiles[i:i + batch_size]
        batch_features = prepare_con_features(batch_smiles)
        batch_features = torch.tensor(batch_features, dtype=torch.float32).to(device)
        con_features.append(batch_features)
    con_features = torch.cat(con_features, dim=0)

    #  GIN Info Features
    print(f"{dataset_name} - info_feature generating...")
    info_features = []
    for i in range(0, len(smiles), batch_size):
        batch_smiles = smiles[i:i + batch_size]
        batch_features = prepare_info_features(batch_smiles)
        batch_features = torch.tensor(batch_features, dtype=torch.float32).to(device)
        info_features.append(batch_features)
    info_features = torch.cat(info_features, dim=0)

    #  Edge Features
    print(f"{dataset_name} - edge_feature generating...")
    edge_features = []
    for i in range(0, len(smiles), batch_size):
        batch_smiles = smiles[i:i + batch_size]
        batch_edges = prepare_edge_features(batch_smiles)
        batch_edges = torch.tensor(batch_edges, dtype=torch.float32).to(device)
        edge_features.append(batch_edges)
    edge_features = torch.cat(edge_features, dim=0)

    #  Mask Features
    print(f"{dataset_name} - mask_feature generating...")
    mask_features = []
    for i in range(0, len(smiles), batch_size):
        batch_smiles = smiles[i:i + batch_size]
        batch_masks = prepare_mask_features(batch_smiles)
        batch_masks = torch.tensor(batch_masks, dtype=torch.float32).to(device)
        mask_features.append(batch_masks)
    mask_features = torch.cat(mask_features, dim=0)

    # ✅ 생성된 feature 저장 (캐시 활용)
    torch.save(con_features, con_cache_path)
    torch.save(info_features, info_cache_path)
    torch.save(edge_features, edge_cache_path)
    torch.save(mask_features, mask_cache_path)

    return con_features, info_features, edge_features, mask_features

def compute_auprc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

class CrossAttention(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_heads=8):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.projection = nn.Linear(sum(input_sizes), hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, *inputs):
        combined = torch.cat(inputs, dim=-1)
        projected = self.projection(combined)
        attn_output, _ = self.attention(projected, projected, projected)
        return self.dropout(self.layer_norm(attn_output))

class SafeBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(SafeBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
    def forward(self, x):
        if x.shape[0] == 1:
            return x
        else:
            return self.bn(x)

class MoEForMultiModel(nn.Module):
    def __init__(self, input_sizes, output_size, hidden_size, num_experts=4, k=2, dropout=0.4):
        super(MoEForMultiModel, self).__init__()
        self.num_experts = num_experts
        self.k = k  

        self.cross_attention = CrossAttention(input_sizes, hidden_size)

        self.gate = nn.Sequential(
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)  
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.GELU(),
                SafeBatchNorm1d(1024),  
                nn.Dropout(dropout),
                nn.Linear(1024, 512),
                nn.GELU(),
                SafeBatchNorm1d(512),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.GELU(),
                SafeBatchNorm1d(256),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 1)  
            ) for _ in range(num_experts)
        ])

    def forward(self, con_output):
        combined_input = self.cross_attention(con_output)  

        routing_weights = self.gate(combined_input)  

        topk_values, topk_indices = torch.topk(routing_weights, self.k, dim=-1)  
        topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True)  

        batch_size = combined_input.shape[0]
        expert_outputs = torch.zeros(batch_size, self.k, 1).to(combined_input.device)

        for i in range(self.k):
            expert_idx = topk_indices[:, i] 
            for j in range(batch_size):
                expert_outputs[j, i] = self.experts[expert_idx[j]](combined_input[j:j+1])

        final_output = torch.sum(expert_outputs * topk_values.unsqueeze(-1), dim=1)  

        return torch.sigmoid(final_output).view(-1) 

dataset_routing_history = {}

def track_routing_over_epochs(dataset_name, routing_assignments, num_experts, epoch):
    """
    각 epoch마다 데이터셋별 expert 분포를 저장하는 함수
    - dataset_name: 데이터셋 이름 
    - routing_assignments: 현재 epoch에서 각 샘플이 routing된 expert 인덱스 리스트
    - num_experts: 사용된 expert 개수
    - epoch: 현재 epoch 번호
    """
    if dataset_name not in dataset_routing_history:
        dataset_routing_history[dataset_name] = {expert: [] for expert in range(num_experts)}

    # 각 expert에 얼마나 많은 샘플이 routing되었는지 count
    expert_counts = np.zeros(num_experts)
    for assign in routing_assignments:
        expert_counts[assign] += 1

    # 비율 계산 (각 expert별 routing된 샘플 비율)
    total_samples = len(routing_assignments)
    expert_ratios = expert_counts / total_samples if total_samples > 0 else expert_counts

    # 저장
    for expert in range(num_experts):
        dataset_routing_history[dataset_name][expert].append(expert_ratios[expert])

    print(f"[Epoch {epoch}] {dataset_name} - Expert Routing Distribution: {expert_ratios}")


def visualize_routing_weights(moe, dataset_name):
    """
    각 epoch별로 router가 각 embedding(con, info, edge, mask)에 부여한 평균 가중치를 시각화 및 저장
    """
    num_epochs = len(moe.routing_weights)  # 총 학습 epoch 수
    num_experts = moe.num_experts
    num_embeddings = len(moe.input_sizes)  # con, info, edge, mask (총 4개)

    # (Epochs, num_experts) 형태를 (Epochs, num_embeddings) 형태로 변환
    routing_weights = np.array(moe.routing_weights)  # (Epochs, batch_size, num_experts)
    avg_weights_per_epoch = np.mean(routing_weights, axis=1)  # (Epochs, num_experts)

    #  Expert 별 embedding 가중치 평균 계산
    embedding_names = ["con", "info", "edge", "mask"]
    expert_means = avg_weights_per_epoch.T  # (num_experts, Epochs)

    plt.figure(figsize=(12, 6))

    for i in range(num_embeddings):
        plt.plot(range(num_epochs), expert_means[i], label=f"{embedding_names[i]}")

    plt.xlabel("Epoch")
    plt.ylabel("Average Routing Weight")
    plt.title(f"Expert Routing Weights per Embedding ({dataset_name})")
    plt.legend()
    plt.grid(True)

    # 🔹 이미지 저장
    save_path = f"./routing_weights_{dataset_name}.png"
    plt.savefig(save_path)
    print(f"[INFO] Routing weight plot saved at: {save_path}")

    plt.show()

if __name__ == "__main__":
    admet_groups = admet_group(path='data/')
    benchmarks_auprc = [
        admet_groups.get('CYP2C9_Veith'),
        admet_groups.get('CYP2D6_Veith'),
        admet_groups.get('CYP3A4_Veith')
    ]
    benchmarks_auroc = [
        admet_groups.get('hERG'),
        admet_groups.get('AMES'),
        admet_groups.get('DILI')
    ]

    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 5e-5
    hidden_size = 1024

    patience = 5
    best_val_auprc = 0.0
    patience_counter = 0

    load_graph_models()

    for benchmarks, metric in [(benchmarks_auprc, 'AUPRC'), (benchmarks_auroc, 'AUROC')]:
        predictions = {}
        for benchmark in benchmarks:
            name = benchmark['name']
            train_val, test = benchmark['train_val'], benchmark['test']
            
            train, val = train_test_split(train_val, test_size=0.2, random_state=42)

            train_con, _, _, _ = load_or_generate_features(train, batch_size, name)
            val_con, _, _, _ = load_or_generate_features(val, batch_size, name)
            test_con, _, _, _ = load_or_generate_features(test, batch_size, name)
            
            input_sizes = [train_con.shape[1]]
            output_size = 1

            moe = MoEForMultiModel(input_sizes, output_size, hidden_size, dropout=0.4).to(device)

            optimizer = Adam(moe.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = BCELoss()

            for epoch in range(200):
                moe.train()
                optimizer.zero_grad()
                train_output = moe(train_con)
                train_output = train_output.view(-1)
                train_labels = torch.tensor(train["Y"].values[:len(train_output)], dtype=torch.float32).to(device)
                train_output = train_output[:len(train_labels)]
                train_labels = train_labels[:len(train_output)]
                train_loss = criterion(train_output, train_labels)
                train_loss.backward()
                optimizer.step()
                train_auprc = compute_auprc(train_labels.cpu().numpy(), train_output[:len(train_labels)].cpu().detach().numpy())
                
                moe.eval()
                with torch.no_grad():
                    val_output = moe(val_con)
                    val_output = val_output.view(-1)
                    val_labels = torch.tensor(val["Y"].values[:len(val_output)], dtype=torch.float32).to(device)
                    val_output = val_output[:len(val_labels)]
                    val_labels = val_labels[:len(val_output)]
                    val_loss = criterion(val_output, val_labels)
                    val_auprc = compute_auprc(val_labels.cpu().numpy(), val_output[:len(val_labels)].cpu().detach().numpy())

                print(f"Epoch [{epoch+1}/200] | Train Loss: {train_loss.item():.4f} | Train AUPRC: {train_auprc:.4f} | "
                    f"Val Loss: {val_loss.item():.4f} | Val AUPRC: {val_auprc:.4f}")
                
                if val_auprc > best_val_auprc:
                    best_val_auprc = val_auprc
                    patience_counter = 0  
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}!")
                    break

            moe.eval()
            test_output_list = []
            for i in range(0, len(test), batch_size):
                batch_indices = slice(i, min(i + batch_size, len(test)))
                batch_con = test_con[batch_indices]
                with torch.no_grad():
                    batch_output = moe(batch_con).squeeze()
                test_output_list.append(batch_output)

            test_output = torch.cat(test_output_list, dim=0)
            y_pred = test_output.cpu().numpy()
            predictions[name] = y_pred

        results = admet_groups.evaluate(predictions)
        print(f"Evaluation Results for {metric}:", results)