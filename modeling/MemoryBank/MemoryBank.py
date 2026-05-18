import torch
import torch.nn.functional as F


class MemoryBank:
    def __init__(self, feature_dim, num_classes, device='cuda'):
        self.device = device
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.features = torch.zeros(self.num_classes, self.feature_dim, device=device)
        self.labels = torch.arange(self.num_classes, device=device, dtype=torch.long)
        self.counts = torch.zeros(self.num_classes, device=device)

    @classmethod
    def from_tensor(cls, features: torch.Tensor, device=None):
        """
        Build a MemoryBank directly from a [N, C] feature matrix.
        Labels are re-indexed to [0, ..., N-1], which is convenient for
        test-time subsetting of the memory size.
        """
        if not torch.is_tensor(features):
            raise TypeError(f"features must be a torch.Tensor, got {type(features)}")
        if features.dim() != 2:
            raise ValueError(f"features must have shape [N, C], got {tuple(features.shape)}")

        if device is None:
            device = features.device
        bank = cls(feature_dim=features.size(1), num_classes=features.size(0), device=device)
        bank.features = features.detach().to(device).clone()
        bank.labels = torch.arange(features.size(0), device=device, dtype=torch.long)
        bank.counts = torch.ones(features.size(0), device=device)
        return bank

    @property
    def num_entries(self):
        return int(self.features.size(0))

    def clone(self):
        bank = MemoryBank(self.feature_dim, self.num_classes, device=self.features.device)
        bank.features = self.features.clone()
        bank.labels = self.labels.clone()
        bank.counts = self.counts.clone()
        return bank

    def subset(self, size=None, indices=None, mode='first', seed=1234):
        """
        Return a new MemoryBank that keeps only a subset of memory entries.
        The returned bank is re-indexed to [0, ..., N-1] so that labels from
        query_memory can still be used to index target_memory safely.
        """
        total = self.num_entries
        if indices is None:
            if size is None or size <= 0 or size >= total:
                indices = torch.arange(total, device=self.features.device)
            else:
                if mode == 'first':
                    indices = torch.arange(size, device=self.features.device)
                elif mode == 'random':
                    g = torch.Generator(device=self.features.device)
                    g.manual_seed(int(seed))
                    indices = torch.randperm(total, generator=g, device=self.features.device)[:size]
                else:
                    raise ValueError(f"Unsupported subset mode: {mode}")
        else:
            if not torch.is_tensor(indices):
                indices = torch.tensor(indices, device=self.features.device, dtype=torch.long)
            else:
                indices = indices.to(self.features.device, dtype=torch.long)

        indices = indices.long()
        return MemoryBank.from_tensor(self.features.index_select(0, indices), device=self.features.device)

    @torch.no_grad()
    def update(self, features, labels, momentum=0.4):
        for feat, label in zip(features, labels):
            label = int(label.item())
            if self.counts[label] == 0:
                self.features[label] = feat
            else:
                self.features[label] = (1 - momentum) * self.features[label] + momentum * feat
            self.counts[label] += 1

    def get_topk_labels(self, query_feat: torch.Tensor, k=10):
        """
        Supports either:
            - query_feat: [C]
            - query_feat: [B, C]
        Returns:
            labels: [K] or [B, K]
            values: [K] or [B, K]
        """
        if query_feat.dim() == 1:
            single = True
            query_feat = query_feat.unsqueeze(0)
        elif query_feat.dim() == 2:
            single = False
        else:
            raise ValueError(f"query_feat must be [C] or [B, C], got {tuple(query_feat.shape)}")

        if self.num_entries == 0:
            raise ValueError("MemoryBank is empty.")

        q = F.normalize(query_feat, dim=1)
        m = F.normalize(self.features, dim=1)
        sim = q @ m.t()  # [B, N]

        k = max(1, min(int(k), sim.size(1)))
        values, indices = torch.topk(sim, k=k, dim=1)
        labels = self.labels.index_select(0, indices.reshape(-1)).view_as(indices)

        if single:
            return labels[0], values[0]
        return labels, values

    def get_features_by_labels(self, labels: torch.Tensor):
        labels = labels.long().to(self.features.device)
        if labels.numel() == 0:
            raise ValueError("labels is empty in get_features_by_labels")
        if not ((labels >= 0).all() and (labels < self.num_classes).all()):
            raise AssertionError("Invalid label in get_features_by_labels")
        return self.features[labels]

    def get_all(self):
        return self.features
