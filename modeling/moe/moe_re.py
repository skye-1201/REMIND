import torch
import torch.nn as nn
import torch.nn.functional as F

class PerturbationAug(nn.Module):
    def __init__(self, mode='shuffle', ratio=0.1, noise_std=0.02):
        super().__init__()
        self.mode = mode
        self.ratio = ratio
        self.noise_std = noise_std

    def forward(self, x):
        B, C = x.shape
        x_aug = x.clone()
        num_dim = int(C * self.ratio)

        for b in range(B):
            idx = torch.randperm(C)[:num_dim]
            if self.mode == 'noise':
                noise = torch.randn(num_dim, device=x.device) * self.noise_std
                x_aug[b, idx] += noise
            elif self.mode == 'mask':
                x_aug[b, idx] = 0
            elif self.mode == 'shuffle':
                perm = idx[torch.randperm(num_dim)]
                x_aug[b, idx] = x[b, perm]
        return x_aug

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)



class Expert(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.fc = nn.Linear(hidden_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.aug = PerturbationAug(mode='noise', ratio=0.1, noise_std=0.02)
        # self.aug = PerturbationAug(mode='noise', ratio=0.2, noise_std=0.1)

    def forward(self, x,aug):
        if aug:
            x = self.aug(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.fc(x)
        # x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ExpertBranch(nn.Module):
    def __init__(self, input_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        mlp_hidden_dim = int(input_dim * 4)
        self.experts = nn.ModuleList([Expert(in_features=input_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x,aug):
        """
        :param x: Tensor [B, C]
        :return: reconstructed Tensor [B, C]
        """
        gate_weights = self.gate(x)  # [B, num_experts]

        expert_outputs = torch.stack([expert(x,aug) for expert in self.experts], dim=1)  # [B, num_experts, C]

        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)  # [B, C]
        return output