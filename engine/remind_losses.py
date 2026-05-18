import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mmmloss import multiModalMarginLossNew


_mse_loss = nn.MSELoss()
_common_alignment_cache = {}


def _get_loss_cfg(cfg):
    return cfg.REMIND.LOSS


def _common_alignment_loss(margin, dist_type='cos'):
    key = (float(margin), dist_type)
    if key not in _common_alignment_cache:
        _common_alignment_cache[key] = multiModalMarginLossNew(margin=margin, dist_type=dist_type)
    return _common_alignment_cache[key]


def modality_specific_orthogonal_loss(specific_features, fused_feature, modality_weights):
    """
    L_ms in the paper: variance-weighted orthogonality between modality-specific
    features and the fused representation.
    """
    fused_feature = F.normalize(fused_feature, p=2, dim=1)
    loss = fused_feature.new_tensor(0.0)

    for feat, weight in zip(specific_features, modality_weights):
        feat = F.normalize(feat, p=2, dim=1)
        sample_loss = (feat * fused_feature).sum(dim=1).pow(2)
        loss = loss + (weight * sample_loss).sum()

    return loss


def modality_common_alignment_loss(common_features, target, margin):
    """
    L_mc in the paper: cosine-based alignment plus MSE feature-scale constraint.
    """
    common_margin = _common_alignment_loss(margin=margin, dist_type='cos')(
        common_features[0], common_features[1], common_features[2], target
    )
    common_mse = (
        _mse_loss(common_features[1], common_features[0])
        + _mse_loss(common_features[0], common_features[2])
        + _mse_loss(common_features[1], common_features[2])
    )
    return common_margin, common_mse


def reconstruction_loss(reconstruction_outputs, specific_features, target):
    """
    L_Re plus auxiliary ID supervision for the two reconstructed candidates of
    each missing modality.
    """
    recon_rgb, recon_nir, recon_tir = reconstruction_outputs
    sp_rgb, sp_nir, sp_tir = specific_features

    feature_loss = (
        _mse_loss(recon_rgb[1][0], sp_rgb) + _mse_loss(recon_rgb[1][1], sp_rgb)
        + _mse_loss(recon_nir[1][0], sp_nir) + _mse_loss(recon_nir[1][1], sp_nir)
        + _mse_loss(recon_tir[1][0], sp_tir) + _mse_loss(recon_tir[1][1], sp_tir)
    )

    id_loss = (
        F.cross_entropy(recon_rgb[0][0], target) + F.cross_entropy(recon_rgb[0][1], target)
        + F.cross_entropy(recon_nir[0][0], target) + F.cross_entropy(recon_nir[0][1], target)
        + F.cross_entropy(recon_tir[0][0], target) + F.cross_entropy(recon_tir[0][1], target)
    ) / 2.0

    return feature_loss, id_loss


def compute_remind_loss(model_outputs, loss_fn, target, target_cam, epoch, cfg):
    """
    The epoch-dependent training schedule is kept here so that processor.py only
    controls the training loop. All tunable coefficients are read from yml via cfg.
    """
    loss_cfg = _get_loss_cfg(cfg)
    has_reconstruction = len(model_outputs) == 11
    if has_reconstruction:
        (
            ori_score1,
            ori_feat1,
            ori_score2,
            ori_feat2,
            fused_feature,
            modality_weights,
            recon_rgb,
            recon_nir,
            recon_tir,
            specific_features,
            common_features,
        ) = model_outputs
    else:
        (
            ori_score1,
            ori_feat1,
            ori_score2,
            ori_feat2,
            fused_feature,
            modality_weights,
            specific_features,
            common_features,
        ) = model_outputs
        recon_rgb = recon_nir = recon_tir = None

    global_loss = (
        loss_fn(ori_score1, ori_feat1, target, target_cam)
        + loss_fn(ori_score2, ori_feat2, target, target_cam)
    )
    specific_loss = modality_specific_orthogonal_loss(
        specific_features, fused_feature, modality_weights
    )
    common_margin, common_mse = modality_common_alignment_loss(
        common_features, target, margin=float(loss_cfg.COMMON_MARGIN)
    )

    zero = global_loss.new_tensor(0.0)
    recon_feature_loss = zero
    recon_id_loss = zero
    if has_reconstruction:
        recon_feature_loss, recon_id_loss = reconstruction_loss(
            (recon_rgb, recon_nir, recon_tir), specific_features, target
        )

    total_loss = _quiet_loss_schedule(
        epoch=epoch,
        loss_cfg=loss_cfg,
        global_loss=global_loss,
        specific_loss=specific_loss,
        common_margin=common_margin,
        common_mse=common_mse,
        recon_feature_loss=recon_feature_loss,
        recon_id_loss=recon_id_loss,
    )

    return total_loss, ori_score1


def _quiet_loss_schedule(
    epoch,
    loss_cfg,
    global_loss,
    specific_loss,
    common_margin,
    common_mse,
    recon_feature_loss,
    recon_id_loss,
):
    weighted_global = float(loss_cfg.GLOBAL_WEIGHT) * global_loss
    weighted_specific = float(loss_cfg.SPECIFIC_WEIGHT) * specific_loss
    weighted_common = (
        float(loss_cfg.COMMON_MARGIN_WEIGHT) * common_margin
        + float(loss_cfg.COMMON_MSE_WEIGHT) * common_mse
    )
    weighted_reconstruction = (
        float(loss_cfg.RECON_FEATURE_WEIGHT) * recon_feature_loss
        + float(loss_cfg.RECON_ID_WEIGHT) * recon_id_loss
    )

    if epoch is None or epoch <= 1:
        return weighted_global + weighted_specific + weighted_common
    if epoch <= int(loss_cfg.SPECIFIC_LOSS_END_EPOCH):
        return weighted_global + weighted_specific + weighted_common + weighted_reconstruction
    return weighted_global + weighted_common + weighted_reconstruction
