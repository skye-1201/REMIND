from utils.logger import setup_logger
from data import make_dataloader
from modeling import make_model
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from layers.make_loss import make_loss
from engine.processor import do_train
import random
import os
import torch
import numpy as np
from modeling.MemoryBank.MemoryBank import MemoryBank
import argparse
import shutil
from config import cfg


def set_seed(seed=1111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeMo Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--fea_cft", default=0, help="Feature choose to be tested", type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.FEAT = args.fea_cft
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f in cfg.SAVE_LIST:
        file_save_path = os.path.join(output_dir, f.split('/')[-1])
        if os.path.exists(file_save_path):
            os.chmod(file_save_path, 0o700)
        shutil.copyfile(f, file_save_path)
        os.chmod(file_save_path, 0o400)  # read only

    logger = setup_logger("DeMo", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    print("data is ready")

    rgb_memory = MemoryBank(feature_dim=768, num_classes=num_classes, device='cuda')
    nir_memory = MemoryBank(feature_dim=768, num_classes=num_classes, device='cuda')
    tir_memory = MemoryBank(feature_dim=768, num_classes=num_classes, device='cuda')
    sr_memory = MemoryBank(feature_dim=768, num_classes=num_classes, device='cuda')
    sn_memory = MemoryBank(feature_dim=768, num_classes=num_classes, device='cuda')
    st_memory = MemoryBank(feature_dim=768, num_classes=num_classes, device='cuda')

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        memory=[rgb_memory, nir_memory, tir_memory, sr_memory, sn_memory, st_memory]
    )

    if hasattr(model, 'flops'):
        logger.info(str(model))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of parameters:{n_parameters / 1e6}")
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    else:
        print("model has no flops")

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query,
        args.local_rank,
        rgb_memory,
        nir_memory,
        tir_memory,
        sr_memory,
        sn_memory,
        st_memory
    )
