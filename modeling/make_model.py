import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from fvcore.nn import flop_count
from .backbones.vit_pytorch import DropPath
from modeling.make_model_clipreid import load_clip_to_cpu
from .backbones.vit_pytorch import Mlp
import copy
from .moe.moe_re import ExpertBranch
from .clip import clip
from modeling.MemoryBank.MemoryBank import MemoryBank
import torch.nn.functional as F
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
import torch
import random

def generate_miss_list(num_samples, drop_ratio=0.5):
    miss_list = []
    for _ in range(num_samples):
        if random.random() < drop_ratio:
            miss_list.append(random.choice(['r', 'n', 't', 'rn', 'rt', 'nt']))
        else:
            miss_list.append('none')
    return miss_list

class CAP(nn.Module):
    def __init__(self, dim=768):
        super(CAP, self).__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, query, key_value):
        # query: [b, c]
        # key_value: [b, c]

        # 投影
        q = self.q_proj(query).unsqueeze(1)        # [b, 1, dim]
        k = self.k_proj(key_value).unsqueeze(1)    # [b, 1, dim]
        v = self.v_proj(key_value).unsqueeze(1)    # [b, 1, dim]

        # 注意力打分
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)  # [b, 1, 1]
        attn_probs = F.softmax(attn_scores, dim=-1)                             # [b, 1, 1]

        # 加权求和
        output = self.fc(torch.matmul(attn_probs, v).squeeze(1)+query)  # [b, dim]

        return output

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights  # 返回注意力矩阵

class PerturbationAug(nn.Module):
    def __init__(self, mode='noise', ratio=0.1, noise_std=0.02):
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

class SRExtractor(nn.Module):
    def __init__(self, input_dim=768, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        self.encoder_layer = CustomTransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, feat_r, feat_n, feat_t, miss=''):
        B, C = feat_r.shape
        device = feat_r.device

        fused = torch.stack([feat_r, feat_n, feat_t], dim=1)  # [B, 3, C]

        mask_map = {
            'r': [1, 0, 0],
            'n': [0, 1, 0],
            't': [0, 0, 1],
            'rn': [1, 1, 0],
            'rt': [1, 0, 1],
            'nt': [0, 1, 1],
            '': [0, 0, 0],
        }
        miss_bin = mask_map.get(miss, [0, 0, 0])
        attn_mask = torch.tensor(miss_bin, dtype=torch.bool, device=device).unsqueeze(0).repeat(B, 1)

        # transformer 只含1层 encoder layer，所以直接调用该层
        shared_all, attn_weights = self.encoder_layer(fused, src_key_padding_mask=attn_mask)

        shared_r = shared_all[:, 0, :]
        shared_n = shared_all[:, 1, :]
        shared_t = shared_all[:, 2, :]

        # attn_weights 形状 [B * num_heads, 3, 3] 或者 [B, num_heads, 3, 3]
        # 根据实际维度调整
        if attn_weights.dim() == 3:
            attn_weights = attn_weights.unsqueeze(1)  # [B, 1, 3, 3]
        attn_avg = attn_weights.mean(dim=1)  # 平均所有 head，[B, 3, 3]

        # 取每个模态对自身的注意力强度
        attn_r = attn_avg[:, 0, 0].unsqueeze(1)
        attn_n = attn_avg[:, 1, 1].unsqueeze(1)
        attn_t = attn_avg[:, 2, 2].unsqueeze(1)

        return [shared_r, shared_n, shared_t], [attn_r, attn_n, attn_t]

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normy = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(512, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        f1 = x.unsqueeze(1)
        f2 = y.unsqueeze(1)
        f1 = f1.expand(f2.shape[0], -1, -1)
        # pdb.set_trace()
        B, N, C = f2.shape
        q = self.q_(f1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(self.normy(f2)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(f2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.squeeze(1)


class caBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(512)
        self.norm2 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(dim, dim)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x,y):
        # pdb.set_trace()
        fina = self.drop_path(self.attn(self.norm1(self.l1(x)),self.norm2(self.l2(y))))
        # fina = x + self.drop_path(self.mlp(self.norm2(x)))
        return fina

def get_avg_feature_by_topk_labels(memory: MemoryBank, topk_labels: torch.Tensor) -> torch.Tensor:
    """
    给定 top-K 标签，从 memory 中取出特征并求平均。
    Args:
        memory (MemoryBank): 用于查找特征的 memory bank
        topk_labels (Tensor): top-K 标签，形状为 [K]
    Returns:
        Tensor: 平均特征，形状为 [feature_dim]
    """
    feats = memory.get_features_by_labels(topk_labels)  # [K, C]
    avg_feat = feats.mean(dim=0)  # [C]
    return avg_feat


def get_avg_features_batch(query_feats: torch.Tensor,
                           query_memory: MemoryBank,
                           target_memory: MemoryBank,
                           k=5,
                           use_query_memory=True):
    """
    For each query feature in a batch, retrieve top-k labels from a memory bank,
    then average the corresponding target-memory features.

    Vehicle setting (MODEL.DIRECT=0): use query_memory to find top-k labels.
    persontrian setting (MODEL.DIRECT=1): keep the original persontrian behavior,
    using target_memory to find top-k labels.
    """
    avg_feats = []
    search_memory = query_memory if use_query_memory else target_memory
    for i in range(query_feats.size(0)):
        topk_labels, _ = search_memory.get_topk_labels(query_feats[i], k)
        avg_feat = get_avg_feature_by_topk_labels(target_memory, topk_labels)
        avg_feats.append(avg_feat)
    return torch.stack(avg_feats, dim=0)  # [B, C]


class DeMo(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, memory, factory):
        super(DeMo, self).__init__()

        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        else:
            self.feat_dim = 768

        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        try:
            self.direct = int(cfg.MODEL.DIRECT)
        except Exception:
            self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss = cfg.TEST.MISS
        if self.miss is None or self.miss == 'none':
            self.miss = ''
        try:
            self.ratio = cfg.TEST.RATIO
        except Exception:
            self.ratio = None
        self.GLOBAL_LOCAL = cfg.MODEL.GLOBAL_LOCAL
        self.head = cfg.MODEL.HEAD
        self.share_heads = cfg.MODEL.SH_HEAD
        self.num_experts = cfg.MODEL.BLOCK
        self.numk = cfg.MODEL.TOPK
        self.memory = memory
        self.warmup_epochs = 1
        self.model_name = 'ViT-B-16'

        # MODEL.DIRECT=0: vehicle setting, three modality-specific backbones.
        # MODEL.DIRECT=1: persontrian setting, one shared backbone.
        if self.direct == 1:
            self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
  
        else:
            self.RGB = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
            self.NIR = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
            self.TIR = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)

        self.ca1 = caBlock(self.feat_dim, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                           drop=0., attn_drop=0., drop_path=0.1, act_layer=nn.GELU,
                           norm_layer=nn.LayerNorm)
        self.ca2 = copy.deepcopy(self.ca1)
        self.ca3 = copy.deepcopy(self.ca1)
        self.cas = copy.deepcopy(self.ca1)
        self.cat1 = CAP(self.feat_dim)
        self.cat2 = copy.deepcopy(self.cat1)
        self.share = SRExtractor(self.feat_dim, num_heads=self.share_heads, dropout=0.1)
        self.moe1 = ExpertBranch(self.feat_dim, num_experts=self.num_experts)
        self.moe2 = copy.deepcopy(self.moe1)

        self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifiers = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
        self.classifiers.apply(weights_init_classifier)
        self.bottlenecks = nn.BatchNorm1d(3 * self.feat_dim)
        self.bottlenecks.bias.requires_grad_(False)
        self.bottlenecks.apply(weights_init_kaiming)

        self.classifier1 = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier1.apply(weights_init_classifier)
        self.bottleneck1 = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)

        self.classifier2 = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)
        self.bottleneck2 = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)

        # Extra heads used by the persontrian setting.
        if self.direct == 1:
            self.classifierr = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifierr.apply(weights_init_classifier)
            self.bottleneckr = nn.BatchNorm1d(self.feat_dim)
            self.bottleneckr.bias.requires_grad_(False)
            self.bottleneckr.apply(weights_init_kaiming)

            self.classifiern = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifiern.apply(weights_init_classifier)
            self.bottleneckn = nn.BatchNorm1d(self.feat_dim)
            self.bottleneckn.bias.requires_grad_(False)
            self.bottleneckn.apply(weights_init_kaiming)

            self.classifiert = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifiert.apply(weights_init_classifier)
            self.bottleneckt = nn.BatchNorm1d(self.feat_dim)
            self.bottleneckt.bias.requires_grad_(False)
            self.bottleneckt.apply(weights_init_kaiming)

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def process_branch_R(self, features):
        global_feat = features[0]
        fuse_feat = global_feat + self.ca1(features[1], global_feat)

        feat1 = self.bottleneck1(global_feat)
        feat2 = self.bottleneck2(fuse_feat)
        cls_score1 = self.classifier1(feat1)
        cls_score2 = self.classifier2(feat2)

        if self.training:
            return [cls_score1, cls_score2], [global_feat, fuse_feat]
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat1], dim=1)
            else:
                return torch.cat([fuse_feat], dim=1)

    def process_branch_N(self, features):
        global_feat = features[0]
        fuse_feat = global_feat + self.ca1(features[1], global_feat)

        feat1 = self.bottleneck1(global_feat)
        feat2 = self.bottleneck2(fuse_feat)
        cls_score1 = self.classifier1(feat1)
        cls_score2 = self.classifier2(feat2)

        if self.training:
            return [cls_score1, cls_score2], [global_feat, fuse_feat]
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat1], dim=1)
            else:
                return torch.cat([fuse_feat], dim=1)

    def process_branch_T(self, features):
        global_feat = features[0]
        fuse_feat = global_feat + self.ca1(features[1], global_feat)

        feat1 = self.bottleneck1(global_feat)
        feat2 = self.bottleneck2(fuse_feat)
        cls_score1 = self.classifier1(feat1)
        cls_score2 = self.classifier2(feat2)

        if self.training:
            return [cls_score1, cls_score2], [global_feat, fuse_feat]
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat1], dim=1)
            else:
                return torch.cat([fuse_feat], dim=1)

    def memory_R(self, features):
        use_query_memory = (self.direct == 0)
        f1 = get_avg_features_batch(features[0][1], self.memory[3], self.memory[0],
                                    use_query_memory=use_query_memory)
        f2 = get_avg_features_batch(features[0][2], self.memory[3], self.memory[0],
                                    use_query_memory=use_query_memory)
        ff1 = self.cat1(f1, features[0][1])
        ff2 = self.cat2(f2, features[0][2])
        aug = self.training
        feat1 = self.moe1(ff1, aug)
        feat2 = self.moe2(ff2, aug)
        feats1 = self.bottleneck1(feat1)
        feats2 = self.bottleneck2(feat2)

        if self.training:
            cls_score1 = self.classifier1(feats1)
            cls_score2 = self.classifier2(feats2)
            return [cls_score1, cls_score2], [feat1, feat2]
        else:
            return [feat1, feat2]

    def memory_N(self, features):
        use_query_memory = (self.direct == 0)
        f1 = get_avg_features_batch(features[0][0], self.memory[4], self.memory[1],
                                    use_query_memory=use_query_memory)
        f2 = get_avg_features_batch(features[0][2], self.memory[4], self.memory[1],
                                    use_query_memory=use_query_memory)
        ff1 = self.cat1(f1, features[0][0])
        ff2 = self.cat2(f2, features[0][2])
        aug = self.training
        feat1 = self.moe1(ff1, aug)
        feat2 = self.moe2(ff2, aug)
        feats1 = self.bottleneck1(feat1)
        feats2 = self.bottleneck2(feat2)

        if self.training:
            cls_score1 = self.classifier1(feats1)
            cls_score2 = self.classifier2(feats2)
            return [cls_score1, cls_score2], [feat1, feat2]
        else:
            return [feat1, feat2]

    def memory_T(self, features):
        use_query_memory = (self.direct == 0)
        f1 = get_avg_features_batch(features[0][0], self.memory[5], self.memory[2],
                                    use_query_memory=use_query_memory)
        f2 = get_avg_features_batch(features[0][1], self.memory[5], self.memory[2],
                                    use_query_memory=use_query_memory)
        ff1 = self.cat1(f1, features[0][0])
        ff2 = self.cat2(f2, features[0][1])
        aug = self.training
        feat1 = self.moe1(ff1, aug)
        feat2 = self.moe2(ff2, aug)
        feats1 = self.bottleneck1(feat1)
        feats2 = self.bottleneck2(feat2)

        if self.training:
            cls_score1 = self.classifier1(feats1)
            cls_score2 = self.classifier2(feats2)
            return [cls_score1, cls_score2], [feat1, feat2]
        else:
            return [feat1, feat2]

    def apply_fourier_transform(self, x):
        """
        Apply Fourier transform on the feature dimension.
        """
        return torch.fft.fft(x, dim=-1)

    def calculate_frequency_contribution(self, freq_data):
        """
        Calculate frequency-domain contribution.
        freq_data: [batch_size, feature_dim]
        """
        energy = torch.sum(torch.abs(freq_data) ** 2, dim=-1)
        return energy

    def weighted_fusion(self, R, N, T):
        """
        Variance-based weighted fusion.
        """
        var_r = torch.var(R, dim=1)
        var_n = torch.var(N, dim=1)
        var_t = torch.var(T, dim=1)

        total_contrib = var_r + var_n + var_t
        R_weight = var_r / total_contrib
        N_weight = var_n / total_contrib
        T_weight = var_t / total_contrib
        Tw = 3 - R_weight - N_weight - T_weight
        rw = (1 - R_weight) / Tw
        nw = (1 - N_weight) / Tw
        tw = (1 - T_weight) / Tw
        fused_freq = (R * rw.unsqueeze(1) +
                      N * nw.unsqueeze(1) +
                      T * tw.unsqueeze(1))
        return fused_freq, [rw, nw, tw]

    def post_process(self, featR, featN, featT):
        resR = self.process_branch_R(featR)
        resN = self.process_branch_N(featN)
        resT = self.process_branch_T(featT)
        return resR, resN, resT

    def post_memory(self, featR, featN, featT):
        resR = self.memory_R(featR)
        resN = self.memory_N(featN)
        resT = self.memory_T(featT)
        return resR, resN, resT

    def forward(self, x, label=None, cam_label=None, view_label=None,
                return_pattern=3, img_path=None, epoch=None, miss=None):
        if self.direct == 1:
            return self._forward_persontrian(x, label=label, cam_label=cam_label,
                                            view_label=view_label, return_pattern=return_pattern,
                                            img_path=img_path, epoch=epoch)
        return self._forward_vehicle(x, label=label, cam_label=cam_label,
                                     view_label=view_label, return_pattern=return_pattern,
                                     img_path=img_path, epoch=epoch)

    def _forward_vehicle(self, x, label=None, cam_label=None, view_label=None,
                         return_pattern=3, img_path=None, epoch=None):

        RGB = x['RGB']
        NI = x['NI']
        TI = x['TI']

        RGB_cash, RGB_global, textR = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
        NI_cash, NI_global, textN = self.NIR(NI, cam_label=cam_label, view_label=view_label)
        TI_cash, TI_global, textT = self.TIR(TI, cam_label=cam_label, view_label=view_label)

        model1, model2, model3 = self.post_process([RGB_global, textR], [NI_global, textN], [TI_global, textT])

        share, attn = self.share(RGB_global, NI_global, TI_global, self.miss)
        shareR = share[0] + self.cas(textR, share[0])
        shareN = share[1] + self.cas(textN, share[1])
        shareT = share[2] + self.cas(textT, share[2])

        if self.training:
            spr = model1[1][1]
            spn = model2[1][1]
            spt = model3[1][1]
        else:
            spr = model1
            spn = model2
            spt = model3

        sh = [shareR, shareN, shareT]
        sp = [spr, spn, spt]
        ori = [RGB_global, NI_global, TI_global]

        if self.training:
            self.memory[0].update(spr, label)
            self.memory[1].update(spn, label)
            self.memory[2].update(spt, label)
            self.memory[3].update(shareR, label)
            self.memory[4].update(shareN, label)
            self.memory[5].update(shareT, label)
            if epoch is not None and epoch > self.warmup_epochs:
                memoryr, memoryn, memoryt = self.post_memory([sh, attn], [sh, attn], [sh, attn])
        else:
            memoryr, memoryn, memoryt = self.post_memory([sh, attn], [sh, attn], [sh, attn])

        if self.training:
            fused_features, weight = self.weighted_fusion(sp[0], sp[1], sp[2])
            if epoch is not None and epoch > self.warmup_epochs:
                return model1, model2, model3, fused_features, weight, memoryr, memoryn, memoryt, sp, sh, ori
            return model1, model2, model3, fused_features, weight, sp, sh

        if self.miss == 'r':
            return torch.cat([(memoryr[0] + memoryr[1]) / 2, spn, spt], dim=1)
        elif self.miss == 'n':
            return torch.cat([spr, (memoryn[0] + memoryn[1]) / 2, spt], dim=1)
        elif self.miss == 't':
            return torch.cat([spr, spn, (memoryt[0] + memoryt[1]) / 2], dim=1)
        elif self.miss == 'rn':
            return torch.cat([memoryr[1], memoryn[1], spt], dim=1)
        elif self.miss == 'rt':
            return torch.cat([memoryr[0], spn, memoryt[1]], dim=1)
        elif self.miss == 'nt':
            return torch.cat([spr, memoryn[0], memoryt[0]], dim=1)
        else:
            return torch.cat([spr, spn, spt], dim=1)

    def _forward_persontrian(self, x, label=None, cam_label=None, view_label=None,
                            return_pattern=3, img_path=None, epoch=None):

        RGB = x['RGB']
        NI = x['NI']
        TI = x['TI']

        RGB_cash, RGB_global, textR = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
        NI_cash, NI_global, textN = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
        TI_cash, TI_global, textT = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)

        share, attn = self.share(RGB_global, NI_global, TI_global, self.miss)
        shareR = share[0] + self.cas(textR, share[0])
        shareN = share[1] + self.cas(textN, share[1])
        shareT = share[2] + self.cas(textT, share[2])

        spr = RGB_global + self.ca1(textR, RGB_global)
        spn = NI_global + self.ca2(textN, NI_global)
        spt = TI_global + self.ca3(textT, TI_global)

        sh = [shareR, shareN, shareT]
        sp = [spr, spn, spt]

        if self.training:
            if epoch is not None and epoch > self.warmup_epochs:
                memoryr, memoryn, memoryt = self.post_memory([sh, attn], [sh, attn], [sh, attn])
            self.memory[0].update(spr, label)
            self.memory[1].update(spn, label)
            self.memory[2].update(spt, label)
            self.memory[3].update(shareR, label)
            self.memory[4].update(shareN, label)
            self.memory[5].update(shareT, label)
        else:
            memoryr, memoryn, memoryt = self.post_memory([sh, attn], [sh, attn], [sh, attn])

        ori1 = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
        ori_global1 = self.bottleneck(ori1)
        ori_score1 = self.classifier(ori_global1)
        ori = [RGB_global, NI_global, TI_global]

        # Keep the original persontrian branch behavior for compatibility.
        ori_globalr = self.bottleneckr(shareR)
        ori_scorer = self.classifierr(ori_globalr)
        ori_globaln = self.bottleneckr(shareN)
        ori_scoren = self.classifierr(ori_globaln)
        ori_globalt = self.bottleneckr(shareT)
        ori_scoret = self.classifierr(ori_globalt)

        ori2 = torch.cat(sp, dim=-1)

        if self.training:
            fused_features, weight = self.weighted_fusion(sp[0], sp[1], sp[2])
            if epoch is not None and epoch > self.warmup_epochs:
                return (ori_score1, ori1, [ori_scorer, ori_scoren, ori_scoret],
                        [ori_globalr, ori_globaln, ori_globalt],
                        fused_features, weight, memoryr, memoryn, memoryt, sp, sh, ori)
            return (ori_score1, ori1, [ori_scorer, ori_scoren, ori_scoret],
                    [ori_globalr, ori_globaln, ori_globalt],
                    fused_features, weight, sp, sh, ori)

        if self.miss == 'r':
            return torch.cat([(memoryr[0] + memoryr[1]) / 2, NI_global, TI_global], dim=1)
        elif self.miss == 'n':
            return torch.cat([RGB_global, (memoryn[0] + memoryn[1]) / 2, TI_global], dim=1)
        elif self.miss == 't':
            return torch.cat([RGB_global, NI_global, (memoryt[0] + memoryt[1]) / 2], dim=1)
        elif self.miss == 'rn':
            return torch.cat([memoryr[1], memoryn[1], TI_global], dim=1)
        elif self.miss == 'rt':
            return torch.cat([memoryr[0], NI_global, memoryt[1]], dim=1)
        elif self.miss == 'nt':
            return torch.cat([RGB_global, memoryn[0], memoryt[0]], dim=1)
        else:
            return (ori2 + ori1) / 2


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0, memory=None):
    model = DeMo(num_class, cfg, camera_num, view_num,memory, __factory_T_type)
    print('===========Building DeMo===========')
    return model


