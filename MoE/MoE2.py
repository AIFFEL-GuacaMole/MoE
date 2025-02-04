import os 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split  
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.optim import Adam
from tdc.benchmark_group import admet_group
from sklearn.metrics import precision_recall_curve, auc
from molfeat.trans.pretrained import PretrainedDGLTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_or_generate_features(data, batch_size, dataset_name, split):
    cache_dir = "MoE/MoE/cache"
    os.makedirs(cache_dir, exist_ok=True)  
    
    con_cache_path = os.path.join(cache_dir, f"{dataset_name}_con_features.pt")
    info_cache_path = os.path.join(cache_dir, f"{dataset_name}_info_features.pt")
    edge_cache_path = os.path.join(cache_dir, f"{dataset_name}_edge_features.pt")
    mask_cache_path = os.path.join(cache_dir, f"{dataset_name}_mask_features.pt")

    if all(os.path.exists(path) for path in [con_cache_path, info_cache_path, edge_cache_path, mask_cache_path]):
        print(f"Loading cached features for {dataset_name}...")

        con_features = torch.load(con_cache_path).to(device)
        info_features = torch.load(info_cache_path).to(device)
        edge_features = torch.load(edge_cache_path).to(device)
        mask_features = torch.load(mask_cache_path).to(device)

        return con_features, info_features, edge_features, mask_features

    #  없으면 새로 생성
    smiles = data["Drug"].tolist()
    print(f"Dataset {dataset_name} ({split}): {len(smiles)} molecules for feature extraction")

    # GIN Context Features
    print(f"{dataset_name} ({split}) - con_feature generating...")
    con_features = []
    for i in range(0, len(smiles), batch_size):
        batch_smiles = smiles[i:i + batch_size]
        batch_con = prepare_con_features(batch_smiles)
        batch_con = torch.tensor(batch_con, dtype=torch.float32).to(device)
        con_features.append(batch_con)
    con_features = torch.cat(con_features, dim=0)
    print(f"Generated Context Features: {con_features.shape}")

    #  GIN Info Features
    print(f"Generated Context Features: {con_features.shape}")
    info_features = []
    for i in range(0, len(smiles), batch_size):
        batch_smiles = smiles[i:i + batch_size]
        batch_info = prepare_info_features(batch_smiles)
        batch_info = torch.tensor(batch_info, dtype=torch.float32).to(device)
        info_features.append(batch_info)
    info_features = torch.cat(info_features, dim=0)

    #  Edge Features
    print(f"{dataset_name} ({split}) - edge_feature generating...")

    edge_features = []
    for i in range(0, len(smiles), batch_size):
        batch_smiles = smiles[i:i + batch_size]
        batch_edges = prepare_edge_features(batch_smiles)
        batch_edges = torch.tensor(batch_edges, dtype=torch.float32).to(device)
        edge_features.append(batch_edges)
    edge_features = torch.cat(edge_features, dim=0)

    #  Mask Features
    print(f"{dataset_name} ({split}) - mask_feature generating...")
    mask_features = []
    for i in range(0, len(smiles), batch_size):
        batch_smiles = smiles[i:i + batch_size]
        batch_masks = prepare_mask_features(batch_smiles)
        batch_masks = torch.tensor(batch_masks, dtype=torch.float32).to(device)
        mask_features.append(batch_masks)
    mask_features = torch.cat(mask_features, dim=0)

    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            batch_smiles = smiles[i:i + batch_size]
            con_features.append(torch.tensor(prepare_con_features(batch_smiles), dtype=torch.float32).to(device))
            info_features.append(torch.tensor(prepare_info_features(batch_smiles), dtype=torch.float32).to(device))
            edge_features.append(torch.tensor(prepare_edge_features(batch_smiles), dtype=torch.float32).to(device))
            mask_features.append(torch.tensor(prepare_mask_features(batch_smiles), dtype=torch.float32).to(device))
    
    con_features = torch.cat(con_features, dim=0)
    info_features = torch.cat(info_features, dim=0)
    edge_features = torch.cat(edge_features, dim=0)
    mask_features = torch.cat(mask_features, dim=0)

    torch.save(con_features, con_cache_path)
    torch.save(info_features, info_cache_path)
    torch.save(edge_features, edge_cache_path)
    torch.save(mask_features, mask_cache_path)

    return con_features, info_features, edge_features, mask_features

def compute_auprc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

class SparseDispatcher:
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        expected_batch_size = self._gates.size(0)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined[:expected_batch_size]

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)  
        self.dropout1 = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)  
        out = self.relu(out)
        out = self.dropout1(out)  
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([MLP(input_size, output_size, hidden_size) for _ in range(num_experts)])
        self.w_gate = nn.Parameter(torch.randn(input_size, num_experts) * 0.05, requires_grad=True)
        self.w_noise = nn.Parameter(torch.randn(input_size, num_experts) * 0.05, requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def forward(self, x, loss_coef=0.5):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss

if __name__ == "__main__":
    admet_groups = admet_group(path='data/')
    benchmarks = [admet_groups.get('CYP2C9_Veith')]
    batch_size = 32
    input_size = 1200
    output_size = 1
    num_experts = 3
    hidden_size = 512
    learning_rate = 5e-4
    weight_decay = 1e-4
    patience = 10
    best_val_auprc = 0.0
    patience_counter = 0

    for benchmark in benchmarks:
        name = benchmark['name']
        train_val, test = benchmark['train_val'], benchmark['test']
        train, val = train_test_split(train_val, test_size=0.2, stratify=train_val["Y"], random_state=42)
        
        train_con, train_info, train_edge, train_mask = load_or_generate_features(train, batch_size, name, "train")
        val_con, val_info, val_edge, val_mask = load_or_generate_features(val, batch_size, name, "val")
        test_con, test_info, test_edge, test_mask = load_or_generate_features(test, batch_size, name, "test")

        moe = MoE(input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=3).to(device)
        optimizer = Adam(moe.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))

        for epoch in range(200):
            moe.train()
            optimizer.zero_grad()
            train_input = torch.cat([train_con, train_info, train_edge, train_mask], dim=1)
            train_output, loss_entropy_reg = moe(train_input)
            train_labels = torch.tensor(train["Y"].values, dtype=torch.float32).unsqueeze(1).to(device)
            train_loss = criterion(train_output, train_labels) + loss_entropy_reg
            
            train_loss.backward()
            optimizer.step()
            train_auprc = compute_auprc(train_labels.cpu().numpy(), train_output.cpu().detach().numpy())
            
            moe.eval()
            with torch.no_grad():
                val_input = torch.cat([val_con, val_info, val_edge, val_mask], dim=1)
                val_output, _ = moe(val_input)
                val_output = val_output.mean(dim=1) # [7738]
                val_labels = torch.tensor(val["Y"].values, dtype=torch.float32).unsqueeze(1).to(device) # [1935,1]
                val_loss = criterion(val_output, val_labels)
                val_auprc = compute_auprc(val_labels.cpu().numpy(), val_output.cpu().detach().numpy())
            
            scheduler.step(val_loss)
            print(f"Epoch [{epoch+1}/200] | Train Loss: {train_loss.item():.4f} | Train AUPRC: {train_auprc:.4f} | "
                f"Val Loss: {val_loss.item():.4f} | Val AUPRC: {val_auprc:.4f}")
            
            """ if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break """

            moe.eval()
            test_output_list = []
            predictions = {}
            for i in range(0, len(test), batch_size):
                batch_indices = slice(i, min(i + batch_size, len(test)))
                batch_con = test_con[batch_indices]
                batch_info = test_info[batch_indices]
                batch_edge = test_edge[batch_indices]
                batch_mask = test_mask[batch_indices]
                batch_input = torch.cat([batch_con, batch_info, batch_edge, batch_mask], dim=1)
                with torch.no_grad():
                    batch_output = moe(batch_input).squeeze()
                test_output_list.append(batch_output)
            
            test_output = torch.cat(test_output_list, dim=0)
            y_pred = test_output.cpu().numpy()
            predictions[name] = y_pred

            results = admet_groups.evaluate(predictions)
            print(f"Evaluation Results for AUPRC:", results)
