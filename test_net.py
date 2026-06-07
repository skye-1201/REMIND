import os
import argparse
import torch

from config import cfg
from data import make_dataloader
from modeling import make_model
from engine.processor import do_inference
from utils.logger import setup_logger


def load_memory_banks(memory_path):
    memory_data = torch.load(memory_path, map_location='cuda')
    return [
        memory_data['rgb'],
        memory_data['nir'],
        memory_data['tir'],
        memory_data['sr'],
        memory_data['sn'],
        memory_data['st'],
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeMo Testing")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument(
        "--return_pattern",
        default=3,
        type=int,
        help="feature pattern used in inference; default is 3 for [moe,ori]"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("DeMo", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    ckpt_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth')
    memory_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'memory.pth')

    logger.info("=" * 60)
    logger.info("Testing with complete-modality best checkpoint")
    logger.info("Forward missing flag is read from cfg.TEST.MISS / model.self.miss: {}".format(cfg.TEST.MISS))
    logger.info("Checkpoint path: {}".format(ckpt_path))
    logger.info("Memory path: {}".format(memory_path))
    logger.info("=" * 60)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(ckpt_path))
    if not os.path.exists(memory_path):
        raise FileNotFoundError("Memory bank not found: {}".format(memory_path))

    rgb_memory, nir_memory, tir_memory, sr_memory, sn_memory, st_memory = load_memory_banks(memory_path)

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        memory=[rgb_memory, nir_memory, tir_memory, sr_memory, sn_memory, st_memory]
    )
    model.load_param(ckpt_path)
    model.eval()

    do_inference(
        cfg,
        model,
        val_loader,
        num_query,
        rgb_memory,
        nir_memory,
        tir_memory,
        sr_memory,
        sn_memory,
        st_memory,
        return_pattern=args.return_pattern
    )
