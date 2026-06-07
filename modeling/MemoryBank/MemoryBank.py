import torch
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, feature_dim, num_classes, device='cuda'):
        self.device = device
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.features = torch.zeros(num_classes, feature_dim).to(device)  # 每个 ID 存一个特征
        self.labels = torch.arange(num_classes).to(device)
        self.counts = torch.zeros(num_classes).to(device)  # 用于平均更新特征

    @torch.no_grad()
    def update(self, features, labels, momentum=0.2):
        for feat, label in zip(features, labels):
            label = label.item()
            if self.counts[label] == 0:
                self.features[label] = feat
            else:
                self.features[label] = (1 - momentum) * self.features[label] + momentum * feat
            self.counts[label] += 1

    def get_topk_labels(self, query_feat: torch.Tensor, k=10):
        sim = F.cosine_similarity(query_feat.unsqueeze(0), self.features, dim=1)
        # import pdb;pdb.set_trace()
        topk = torch.topk(sim, k=k, dim=0)
        return self.labels[topk.indices], topk.values



    def get_features_by_labels(self, labels: torch.Tensor):
        assert (labels >= 0).all() and (labels < self.num_classes).all(), "Invalid label in get_features_by_labels"
        return self.features[labels]

    def get_all(self):
        return self.features


