import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
from layers.mmmloss import multiModalMarginLossNew
import torch.nn.functional as F
import torch.distributed as dist

mmmloss = multiModalMarginLossNew(margin=3, dist_type='cos')


def Cyberspace(rgb, nir, tir, fused_features, weight, target):
    loss = 0
    for i in range(len(target)):
        loss1 = torch.mean(torch.matmul(rgb[i].T, fused_features[i])) ** 2
        loss2 = torch.mean(torch.matmul(nir[i].T, fused_features[i])) ** 2
        loss3 = torch.mean(torch.matmul(tir[i].T, fused_features[i])) ** 2
        loss += weight[0][i] * loss1 + weight[1][i] * loss2 + weight[2][i] * loss3
    return loss


def unwrap_model(model):
    """Return the real model when using DataParallel / DistributedDataParallel."""
    return model.module if hasattr(model, 'module') else model


def get_model_state_dict(model):
    """Save the state_dict without the 'module.' prefix."""
    return unwrap_model(model).state_dict()


def is_person_setting(cfg):
    """MODEL.DIRECT=0: vehicle, MODEL.DIRECT=1: persontrian."""
    try:
        return int(cfg.MODEL.DIRECT) == 1
    except Exception:
        return cfg.MODEL.DIRECT == 1


def build_evaluator(cfg, num_query):
    if cfg.DATASETS.NAMES == "MSVR310":
        return R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    return R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)


def save_memory_banks(save_path,
                      rgb_memory,
                      nir_memory,
                      tir_memory,
                      sr_memory,
                      sn_memory,
                      st_memory):
    torch.save({
        'rgb': rgb_memory,
        'nir': nir_memory,
        'tir': tir_memory,
        'sr': sr_memory,
        'sn': sn_memory,
        'st': st_memory,
    }, save_path)



def _compute_vehicle_loss(cfg,model,
                          img,
                          target,
                          target_cam,
                          target_view,
                          epoch,
                          loss_fn):
    if epoch > 1:
        mode1, mode2, mode3, fused_features, weight, memoryr, memoryn, memoryt, sp, sh, ori = model(
            img, label=target, cam_label=target_cam, view_label=target_view, epoch=epoch
        )
        lossr = nn.MSELoss()(memoryr[1][0], sp[0]) + nn.MSELoss()(memoryr[1][1], sp[0])
        lossn = nn.MSELoss()(memoryn[1][0], sp[1]) + nn.MSELoss()(memoryn[1][1], sp[1])
        losst = nn.MSELoss()(memoryt[1][0], sp[2]) + nn.MSELoss()(memoryt[1][1], sp[2])

        idr = (F.cross_entropy(memoryr[0][0], target) + F.cross_entropy(memoryr[0][1], target)) / 2
        idn = (F.cross_entropy(memoryn[0][0], target) + F.cross_entropy(memoryn[0][1], target)) / 2
        idt = (F.cross_entropy(memoryt[0][0], target) + F.cross_entropy(memoryt[0][1], target)) / 2
        lossre = lossr + lossn + losst
        lossid = idr + idn + idt
    else:
        mode1, mode2, mode3, fused_features, weight, sp, sh = model(
            img, label=target, cam_label=target_cam, view_label=target_view, epoch=epoch
        )
        lossre = 0
        lossid = 0

    loss1 = loss_fn(mode1[0], mode1[1], target, target_cam)
    loss2 = loss_fn(mode2[0], mode2[1], target, target_cam)
    loss3 = loss_fn(mode3[0], mode3[1], target, target_cam)

    featr = F.normalize(sp[0], p=2, dim=1)
    featn = F.normalize(sp[1], p=2, dim=1)
    featt = F.normalize(sp[2], p=2, dim=1)
    feat_fused = F.normalize(fused_features, p=2, dim=1)
    lossori = cfg.MODEL.SP_WEIGHT * Cyberspace(featr, featn, featt, feat_fused, weight, target)
    loss4 = mmmloss(sh[0], sh[1], sh[2], target)
    loss5 = nn.MSELoss()(sh[1], sh[0]) + nn.MSELoss()(sh[0], sh[2]) + nn.MSELoss()(sh[1], sh[2])

    if epoch is not None and epoch > 1:
        loss = loss1 + loss2 + loss3 + lossori + loss4 + lossre + lossid + loss5
    else:
        loss = loss1 + loss2 + loss3 + lossori + loss4 + loss5

    if isinstance(mode1[0][0], list):
        acc = (mode1[0][0].max(1)[1] == target).float().mean()
        acc1 = (mode2[0][0].max(1)[1] == target).float().mean()
        acc2 = (mode3[0][0].max(1)[1] == target).float().mean()
    else:
        acc = (mode1[0][0].max(1)[1] == target).float().mean()
        acc1 = (mode2[0][0].max(1)[1] == target).float().mean()
        acc2 = (mode3[0][0].max(1)[1] == target).float().mean()

    return loss, acc, acc1, acc2


def _compute_person_loss(cfg,model,
                         img,
                         target,
                         target_cam,
                         target_view,
                         epoch,
                         loss_fn):
    if epoch > 1:
        ori_score1, ori1, ori_score, ori, fused_features, weight, memoryr, memoryn, memoryt, sp, sh, oris = model(
            img, label=target, cam_label=target_cam, view_label=target_view, epoch=epoch
        )
        lossr = nn.MSELoss()(memoryr[1][0], sp[0]) + nn.MSELoss()(memoryr[1][1], sp[0])
        lossn = nn.MSELoss()(memoryn[1][0], sp[1]) + nn.MSELoss()(memoryn[1][1], sp[1])
        losst = nn.MSELoss()(memoryt[1][0], sp[2]) + nn.MSELoss()(memoryt[1][1], sp[2])

        idr = (F.cross_entropy(memoryr[0][0], target) + F.cross_entropy(memoryr[0][1], target)) / 2
        idn = (F.cross_entropy(memoryn[0][0], target) + F.cross_entropy(memoryn[0][1], target)) / 2
        idt = (F.cross_entropy(memoryt[0][0], target) + F.cross_entropy(memoryt[0][1], target)) / 2

        lossre = lossr + lossn + losst
        lossid = 0.1*(idr + idn + idt)
    else:
        ori_score1, ori1, ori_score, ori, fused_features, weight, sp, sh, oris = model(
            img, label=target, cam_label=target_cam, view_label=target_view, epoch=epoch
        )
        lossre = 0
        lossid = 0

    loss1 = loss_fn(ori_score1, ori1, target, target_cam)
    lossa = loss_fn(ori_score[0], ori[0], target, target_cam)
    lossb = loss_fn(ori_score[1], ori[1], target, target_cam)
    lossc = loss_fn(ori_score[2], ori[2], target, target_cam)
    loss2 = lossa + lossb + lossc

    featr = F.normalize(sp[0], p=2, dim=1)
    featn = F.normalize(sp[1], p=2, dim=1)
    featt = F.normalize(sp[2], p=2, dim=1)
    feat_fused = F.normalize(fused_features, p=2, dim=1)

    loss3 = cfg.MODEL.SP_WEIGHT * Cyberspace(featr, featn, featt, feat_fused, weight, target)
    loss4 = mmmloss(sh[0], sh[1], sh[2], target)
    loss5 = nn.MSELoss()(sh[1], sh[0]) + nn.MSELoss()(sh[0], sh[2]) + nn.MSELoss()(sh[1], sh[2])

    if epoch is not None and epoch > 1:
        loss = loss1 + loss4 + lossre + lossid + loss5 + loss3 + loss2 + lossc
    else:
        loss = loss1 + loss4 + loss5 + loss3 + loss2

    if isinstance(ori_score1, list):
        acc = (ori_score1[0][0].max(1)[1] == target).float().mean()
    else:
        acc = (ori_score1.max(1)[1] == target).float().mean()

    return loss, acc, None, None


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             rgb_memory,
             nir_memory,
             tir_memory,
             sr_memory,
             sn_memory,
             st_memory):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    person_mode = is_person_setting(cfg)

    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("REMIND.train")
    logger.info('start training')
    logger.info("MODEL.DIRECT = {} ({})".format(cfg.MODEL.DIRECT, "persontrian" if person_mode else "vehicle"))

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
            )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter1 = AverageMeter()
    acc_meter2 = AverageMeter()
    scaler = amp.GradScaler()

    # Only complete-modality evaluation is used for checkpoint selection.
    best_index = {'mAP': 0, 'Rank-1': 0, 'Rank-5': 0, 'Rank-10': 0, 'epoch': 0}

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        acc_meter1.reset()
        acc_meter2.reset()

        scheduler.step(epoch)
        model.train()

        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = {
                'RGB': img['RGB'].to(device),
                'NI': img['NI'].to(device),
                'TI': img['TI'].to(device)
            }
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                if person_mode:
                    loss, acc, acc1, acc2 = _compute_person_loss(
                        cfg,model, img, target, target_cam, target_view, epoch, loss_fn
                    )
                else:
                    loss, acc, acc1, acc2 = _compute_vehicle_loss(
                        cfg,model, img, target, target_cam, target_view, epoch, loss_fn
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)
            if acc1 is not None:
                acc_meter1.update(acc1, 1)
            if acc2 is not None:
                acc_meter2.update(acc2, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if person_mode:
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                            epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]
                        )
                    )
                else:
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f},Acc1: {:.3f},Acc2: {:.3f}, Base Lr: {:.2e}".format(
                            epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, acc_meter.avg, acc_meter1.avg, acc_meter2.avg,
                            scheduler._get_lr(epoch)[0]
                        )
                    )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(
                        get_model_state_dict(model),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                    )
            else:
                torch.save(
                    get_model_state_dict(model),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                )

        if epoch % eval_period == 0:
            is_main_process = True
            if cfg.MODEL.DIST_TRAIN:
                is_main_process = (dist.get_rank() == 0)

            if is_main_process:
                logger.info("=" * 60)
                logger.info("Complete-modality evaluation - Epoch: {}".format(epoch))
                logger.info("=" * 60)

                evaluator = build_evaluator(cfg, num_query)
                mAP, cmc = training_neat_eval(
                    cfg, model, val_loader, device, evaluator, epoch, logger,
                    return_pattern=3
                )

                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    best_index['epoch'] = epoch

                    torch.save(
                        get_model_state_dict(model),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth')
                    )
                    save_memory_banks(
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'memory.pth'),
                        rgb_memory, nir_memory, tir_memory, sr_memory, sn_memory, st_memory
                    )

                    logger.info(
                        "Saved complete-modality best checkpoint at epoch {}: "
                        "mAP {:.1%}, Rank-1 {:.1%}, Rank-5 {:.1%}, Rank-10 {:.1%}".format(
                            epoch,
                            best_index['mAP'],
                            best_index['Rank-1'],
                            best_index['Rank-5'],
                            best_index['Rank-10']
                        )
                    )

                logger.info("~" * 50)
                logger.info("Best Epoch: {}".format(best_index['epoch']))
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                logger.info("~" * 50)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 rgb_memory,
                 nir_memory,
                 tir_memory,
                 sr_memory,
                 sn_memory,
                 st_memory,
                 return_pattern=1):
    device = "cuda"
    logger = logging.getLogger("REMIND.test")
    logger.info("Enter inferencing")

    evaluator = build_evaluator(cfg, num_query)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    logger.info("Testing missing-modality setting from cfg.TEST.MISS: {}".format(cfg.TEST.MISS))
    logger.info("~" * 50)
    if return_pattern == 1:
        logger.info("Current is the ori feature testing!")
    elif return_pattern == 2:
        logger.info("Current is the moe feature testing!")
    else:
        logger.info("Current is the [moe,ori] feature testing!")
    logger.info("~" * 50)

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            print(imgpath)
            img = {
                'RGB': img['RGB'].to(device),
                'NI': img['NI'].to(device),
                'TI': img['TI'].to(device)
            }
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(
                img, cam_label=camids, view_label=target_view,
                return_pattern=return_pattern, img_path=imgpath
            )

            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def training_neat_eval(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator,
                       epoch,
                       logger,
                       return_pattern=1):
    evaluator.reset()
    model.eval()

    # Training-time evaluation is always complete-modality evaluation,
    # so the saved best.pth / memory.pth correspond to the original full setting.
    real_model = unwrap_model(model)
    old_miss = getattr(real_model, 'miss', '')
    if hasattr(real_model, 'miss'):
        real_model.miss = ''

    logger.info("~" * 50)
    if return_pattern == 1:
        logger.info("Current is the ori feature testing!")
    elif return_pattern == 2:
        logger.info("Current is the moe feature testing!")
    else:
        logger.info("Current is the [moe,ori] feature testing!")
    logger.info("~" * 50)

    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = {
                'RGB': img['RGB'].to(device),
                'NI': img['NI'].to(device),
                'TI': img['TI'].to(device)
            }
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)

            feat = model(
                img, cam_label=camids, view_label=target_view,
                return_pattern=return_pattern
            )

            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, scenceids, _))
            else:
                evaluator.update((feat, vid, camid, _))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("~" * 50)
    if hasattr(real_model, 'miss'):
        real_model.miss = old_miss
    torch.cuda.empty_cache()
    return mAP, cmc
