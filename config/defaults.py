from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'REMIND'
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH_T = './pretrained/vit_base_patch16_224.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 0.25
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
_C.MODEL.PROMPT = False # From MambaPro
_C.MODEL.ADAPTER = False # From MambaPro
_C.MODEL.FROZEN = False # whether to freeze the backbone
_C.MODEL.HDM = False # legacy switch, unused by REMIND
_C.MODEL.ATM = False # legacy switch, unused by REMIND
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with the contact feanotture
_C.MODEL.DIRECT = 1

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'ViT-B-16'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.FEAT_DIM = 768
_C.MODEL.GLOBAL_LOCAL = False # legacy switch, unused by REMIND
_C.MODEL.HEAD = 12 # Number of heads in the ATMoE

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = True
_C.MODEL.SIE_VIEW = False  # We do not use this parameter



# -----------------------------------------------------------------------------
# REMIND-specific hyperparameters
# -----------------------------------------------------------------------------
_C.REMIND = CN()

_C.REMIND.MEMORY = CN()
# The first epoch only warms up dual memories. Reconstruction starts afterwards.
_C.REMIND.MEMORY.WARMUP_EPOCHS = 1
_C.REMIND.MEMORY.MOMENTUM = 0.4

_C.REMIND.DMC = CN()
_C.REMIND.DMC.SPECIFIC_PROMPT_HEADS = 16
_C.REMIND.DMC.SPECIFIC_PROMPT_MLP_RATIO = 4.0
_C.REMIND.DMC.SPECIFIC_PROMPT_QKV_BIAS = False
# <=0 means using the default attention scale.
_C.REMIND.DMC.SPECIFIC_PROMPT_QK_SCALE = 0.0
_C.REMIND.DMC.SPECIFIC_PROMPT_DROPOUT = 0.0
_C.REMIND.DMC.SPECIFIC_PROMPT_ATTN_DROPOUT = 0.0
_C.REMIND.DMC.SPECIFIC_PROMPT_DROP_PATH = 0.1
_C.REMIND.DMC.COMMON_HEADS = 4
_C.REMIND.DMC.COMMON_DROPOUT = 0.1
_C.REMIND.DMC.VARIANCE_EPS = 1e-9

_C.REMIND.RMR = CN()
_C.REMIND.RMR.RETRIEVAL_TOPK = 5
_C.REMIND.RMR.NUM_EXPERTS = 5
_C.REMIND.RMR.EXPERT_MLP_RATIO = 4.0
_C.REMIND.RMR.EXPERT_DROPOUT = 0.0
_C.REMIND.RMR.AUG_MODE = 'noise'
_C.REMIND.RMR.AUG_RATIO = 0.1
_C.REMIND.RMR.AUG_NOISE_STD = 0.02

_C.REMIND.LOSS = CN()
_C.REMIND.LOSS.GLOBAL_WEIGHT = 1.0
_C.REMIND.LOSS.SPECIFIC_WEIGHT = 0.1
_C.REMIND.LOSS.COMMON_MARGIN = 3.0
_C.REMIND.LOSS.COMMON_MARGIN_WEIGHT = 1.0
_C.REMIND.LOSS.COMMON_MSE_WEIGHT = 1.0
_C.REMIND.LOSS.RECON_FEATURE_WEIGHT = 1.0
_C.REMIND.LOSS.RECON_ID_WEIGHT = 1.0
# Keep the epoch-dependent loss schedule away from processor.py, but make it tunable here.
_C.REMIND.LOSS.SPECIFIC_LOSS_END_EPOCH = 3

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = 'RGBNT201'
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = './data'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 14  # This may be affected by the order of data reading
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16  # You can adjust it to 8 to save memory while the batch_size need to be 64 to ensure the number of ID

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 0.00035
_C.SOLVER.CLIP_BACKBONE_LR = 0.000005
_C.SOLVER.IMAGENET_BACKBONE_LR_FACTOR = 0.8
_C.SOLVER.AMP_ENABLED = True
# Factor of learning bias
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 2
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
_C.SOLVER.SEED = 1111
_C.MODEL.NO_MARGIN = True
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 10
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 1
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 128  # You can adjust it to 64

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 256
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'before'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Pattern of test augmentation
_C.TEST.MISS = ''
_C.TEST.RATIO = 0.0
_C.TEST.MAX_RANK = 50
_C.TEST.WEIGHT_PATH = './outputs/REMINDbest.pth'
_C.TEST.MEMORY_PATH = './outputs/REMINDmemory.pth'
_C.TEST.MEMORY_N = 40
_C.TEST.MEMORY_SUBSET = 'random'
_C.TEST.MEMORY_SEED = 1234
_C.TEST.LATENCY_WARMUP_CALLS = 10
_C.TEST.LATENCY_MAX_RECORD_CALLS = 0
# ----------------------------------------------------------a------------------ #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./outputs/REMIND"
_C.SAVE_LIST = ['modeling/make_model.py', 'engine/processor.py', 'engine/remind_losses.py']
