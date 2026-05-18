import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.MemoryBank.MemoryBank import MemoryBank
from modeling.backbones.vit_pytorch import (
    deit_small_patch16_224,
    vit_base_patch16_224,
    vit_small_patch16_224,
)
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24

from .backbones.vit_pytorch import DropPath
from .meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
from .moe.moe_re import ExpertBranch


class MemoryPriorFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key_value):
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(key_value).unsqueeze(1)
        v = self.v_proj(key_value).unsqueeze(1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        return self.out_proj(torch.matmul(attn, v).squeeze(1) + query)

class AttentionReturningEncoderLayer(nn.TransformerEncoderLayer):
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


class ModalityCommonExtractor(nn.Module):
    def __init__(self, input_dim=768, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        self.encoder_layer = AttentionReturningEncoderLayer(
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

class TextCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normy = nn.LayerNorm(512)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(512, dim, bias=qkv_bias)
        self.v_ = nn.Linear(512, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        f1 = x.unsqueeze(1)
        f2 = y.unsqueeze(1)
        f2 = f2.expand(f1.shape[0], -1, -1)
        # pdb.set_trace()
        B, N, C = f1.shape
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


class TextGuidedCrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(512)
        self.norm2 = norm_layer(dim)
        self.attn = TextCrossAttention(
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
        fina = self.drop_path(self.attn(self.norm2(self.l2(y)),self.norm1(self.l1(x))))
        # fina = x + self.drop_path(self.mlp(self.norm2(x)))
        return fina

def average_memory_features_by_labels(memory: MemoryBank, topk_labels: torch.Tensor) -> torch.Tensor:
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

def retrieve_specific_memory_priors(query_feats: torch.Tensor, query_memory: MemoryBank, target_memory: MemoryBank, k=5):
    """
    对于一个 batch 的 query_feats，从 query_memory 中找 top-k 标签，
    然后从 target_memory 中取出对应特征并取平均。
    
    Args:
        query_feats: [B, C] 查询特征
        query_memory: 查询模态的 memory bank
        target_memory: 用于取平均特征的 memory bank
        k: top-k 数量

    Returns:
        Tensor: [B, C] 每个样本的平均特征
    """
    avg_feats = []
    for i in range(query_feats.size(0)):
        topk_labels, _ = query_memory.get_topk_labels(query_feats[i], k)
        avg_feat = average_memory_features_by_labels(target_memory, topk_labels)
        avg_feats.append(avg_feat)
    return torch.stack(avg_feats, dim=0)  # [B, C]

class REMIND(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num,memory, factory):
        super(REMIND, self).__init__()
        self.feat_dim = int(cfg.MODEL.FEAT_DIM)
        self.visual_encoder = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss = cfg.TEST.MISS
        self.HDM = cfg.MODEL.HDM
        self.ATM = cfg.MODEL.ATM
        self.GLOBAL_LOCAL = cfg.MODEL.GLOBAL_LOCAL
        self.head = cfg.MODEL.HEAD
        self.memory = memory
        self.warmup_epochs = int(cfg.REMIND.MEMORY.WARMUP_EPOCHS)
        self.memory_momentum = float(cfg.REMIND.MEMORY.MOMENTUM)
        self.variance_eps = float(cfg.REMIND.DMC.VARIANCE_EPS)
        self.retrieval_topk = int(cfg.REMIND.RMR.RETRIEVAL_TOPK)
        qk_scale = None if float(cfg.REMIND.DMC.SPECIFIC_PROMPT_QK_SCALE) <= 0 else float(cfg.REMIND.DMC.SPECIFIC_PROMPT_QK_SCALE)
        self.rgb_specific_prompt = TextGuidedCrossAttentionBlock(
            self.feat_dim,
            num_heads=int(cfg.REMIND.DMC.SPECIFIC_PROMPT_HEADS),
            mlp_ratio=float(cfg.REMIND.DMC.SPECIFIC_PROMPT_MLP_RATIO),
            qkv_bias=bool(cfg.REMIND.DMC.SPECIFIC_PROMPT_QKV_BIAS),
            qk_scale=qk_scale,
            drop=float(cfg.REMIND.DMC.SPECIFIC_PROMPT_DROPOUT),
            attn_drop=float(cfg.REMIND.DMC.SPECIFIC_PROMPT_ATTN_DROPOUT),
            drop_path=float(cfg.REMIND.DMC.SPECIFIC_PROMPT_DROP_PATH),
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        )
        self.nir_specific_prompt = copy.deepcopy(self.rgb_specific_prompt)
        self.tir_specific_prompt = copy.deepcopy(self.rgb_specific_prompt)
        self.common_prompt_interaction = copy.deepcopy(self.rgb_specific_prompt)
        self.memory_prior_fusion1 = MemoryPriorFusion(self.feat_dim)
        self.memory_prior_fusion2 = copy.deepcopy(self.memory_prior_fusion1)
        self.common_extractor = ModalityCommonExtractor(
            self.feat_dim,
            num_heads=int(cfg.REMIND.DMC.COMMON_HEADS),
            dropout=float(cfg.REMIND.DMC.COMMON_DROPOUT),
        )
        self.reconstruction_path1 = ExpertBranch(
            self.feat_dim,
            num_experts=int(cfg.REMIND.RMR.NUM_EXPERTS),
            mlp_ratio=float(cfg.REMIND.RMR.EXPERT_MLP_RATIO),
            drop=float(cfg.REMIND.RMR.EXPERT_DROPOUT),
            aug_mode=str(cfg.REMIND.RMR.AUG_MODE),
            aug_ratio=float(cfg.REMIND.RMR.AUG_RATIO),
            aug_noise_std=float(cfg.REMIND.RMR.AUG_NOISE_STD),
        )
        self.reconstruction_path2 = copy.deepcopy(self.reconstruction_path1)

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

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)



    def reconstruct_rgb(self, features):
        f1 = retrieve_specific_memory_priors(features[0][1], self.memory[3], self.memory[0], k=self.retrieval_topk)
        f2 = retrieve_specific_memory_priors(features[0][2], self.memory[3], self.memory[0], k=self.retrieval_topk)
        candidate1 = self.memory_prior_fusion1(f1, features[0][1])
        candidate2 = self.memory_prior_fusion2(f2, features[0][2])
        return self._decode_reconstruction(candidate1, candidate2)

    def reconstruct_nir(self, features):
        f1 = retrieve_specific_memory_priors(features[0][0], self.memory[4], self.memory[1], k=self.retrieval_topk)
        f2 = retrieve_specific_memory_priors(features[0][2], self.memory[4], self.memory[1], k=self.retrieval_topk)
        candidate1 = self.memory_prior_fusion1(f1, features[0][0])
        candidate2 = self.memory_prior_fusion2(f2, features[0][2])
        return self._decode_reconstruction(candidate1, candidate2)

    def reconstruct_tir(self, features):
        f1 = retrieve_specific_memory_priors(features[0][0], self.memory[5], self.memory[2], k=self.retrieval_topk)
        f2 = retrieve_specific_memory_priors(features[0][1], self.memory[5], self.memory[2], k=self.retrieval_topk)
        candidate1 = self.memory_prior_fusion1(f1, features[0][0])
        candidate2 = self.memory_prior_fusion2(f2, features[0][1])
        return self._decode_reconstruction(candidate1, candidate2)

    def _decode_reconstruction(self, candidate1, candidate2):
        use_augmentation = self.training
        feat1 = self.reconstruction_path1(candidate1, use_augmentation)
        feat2 = self.reconstruction_path2(candidate2, use_augmentation)

        if self.training:
            score1 = self.classifier1(self.bottleneck1(feat1))
            score2 = self.classifier2(self.bottleneck2(feat2))
            return [score1, score2], [feat1, feat2]
        return [feat1, feat2]

    def variance_weighted_fusion(self, R, N, T):
        """Variance-aware fusion used by L_ms."""
        var_r = torch.var(R, dim=1)
        var_n = torch.var(N, dim=1)
        var_t = torch.var(T, dim=1)
            
        total_contrib = var_r + var_n + var_t + self.variance_eps  # [batch_size]
        R_weight = var_r / total_contrib
        N_weight = var_n / total_contrib
        T_weight = var_t / total_contrib
        Tw = 3-R_weight-N_weight-T_weight
        rw = (1-R_weight)/Tw
        nw = (1-N_weight)/Tw
        tw = (1-T_weight)/Tw
        fused_freq = (R * rw.unsqueeze(1) +
            N * nw.unsqueeze(1) +
            T * tw.unsqueeze(1))
        return fused_freq, [rw, nw, tw]




    def reconstruct_all_missing(self, featR, featN, featT):

        resR = self.reconstruct_rgb(featR)
        resN = self.reconstruct_nir(featN)
        resT = self.reconstruct_tir(featT)
        
        return resR, resN, resT


    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None, epoch=None):
        RGB = x['RGB']
        NI = x['NI']
        TI = x['TI']
        RGB_cash, RGB_global, textR = self.visual_encoder(RGB, cam_label=cam_label, view_label=view_label)
        NI_cash, NI_global, textN = self.visual_encoder(NI, cam_label=cam_label, view_label=view_label)
        TI_cash, TI_global, textT = self.visual_encoder(TI, cam_label=cam_label, view_label=view_label)
        
        share, attn = self.common_extractor(RGB_global, NI_global, TI_global, self.miss)
        shareR = share[0] + self.common_prompt_interaction(textR, share[0])
        shareN = share[1] + self.common_prompt_interaction(textN, share[1])
        shareT = share[2] + self.common_prompt_interaction(textT, share[2])

        spr = RGB_global + self.rgb_specific_prompt(textR,RGB_global)
        spn = NI_global + self.nir_specific_prompt(textN,NI_global)
        spt = TI_global + self.tir_specific_prompt(textT,TI_global)
        sh = [shareR,shareN,shareT]
        
        sp = [spr,spn,spt]
        if self.training:
                self.memory[0].update(spr, label, momentum=self.memory_momentum)
                self.memory[1].update(spn, label, momentum=self.memory_momentum)
                self.memory[2].update(spt, label, momentum=self.memory_momentum)
                self.memory[3].update(shareR, label, momentum=self.memory_momentum)
                self.memory[4].update(shareN, label, momentum=self.memory_momentum)
                self.memory[5].update(shareT, label, momentum=self.memory_momentum)
                if epoch is not None and epoch > self.warmup_epochs:
                    # memoryr, memoryn, memoryt = self.reconstruct_all_missing([feat_n[:, 0],feat_t[:, 0]], [feat_r[:, 0],feat_t[:, 0]], [feat_r[:, 0],feat_n[:, 0]])
                    memoryr, memoryn, memoryt = self.reconstruct_all_missing([sh,attn], [sh,attn], [sh,attn])

        else:
                
            memoryr, memoryn, memoryt = self.reconstruct_all_missing([sh,attn], [sh,attn], [sh,attn])

        ori1 = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
        ori_global1 = self.bottleneck(ori1)
        ori_score1 = self.classifier(ori_global1)
        ori2 = torch.cat(sp, dim=-1)
        ori_global2 = self.bottlenecks(ori2)
        ori_score2 = self.classifiers(ori_global2)
        if self.training:
            fused_features,weight = self.variance_weighted_fusion(sp[0],sp[1], sp[2])
            if epoch is not None and epoch > self.warmup_epochs:
                return ori_score1, ori1,ori_score2, ori2, fused_features,weight,memoryr, memoryn, memoryt,sp,sh
            else:
                return ori_score1, ori1,ori_score2, ori2, fused_features,weight,sp,sh

        else:
            if self.miss == 'r':
                return torch.cat([(memoryr[0]+memoryr[1])/2, spn, spt],dim=1)
            elif self.miss == 'n':
                return torch.cat([spr, (memoryn[0]+memoryn[1])/2, spt],dim=1)
            elif self.miss == 't':
                return torch.cat([spr, spn, (memoryt[0]+memoryt[1])/2],dim=1)
            elif self.miss == 'rn':
                return torch.cat([memoryr[1], memoryn[1], spt],dim=1)
            elif self.miss == 'rt':
                return torch.cat([memoryr[0], spn, memoryt[1]],dim=1)
            elif self.miss == 'nt':
                return torch.cat([spr, memoryn[0], memoryt[0]],dim=1)
            else:
                return ori2
            
            



__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0, memory=None):
    model = REMIND(num_class, cfg, camera_num, view_num, memory, __factory_T_type)
    print('===========Building REMIND===========')
    return model
