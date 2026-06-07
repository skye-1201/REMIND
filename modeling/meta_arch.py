import torch
import torch.nn as nn
from .clip import clip
from timm.models.layers import trunc_normal_
from modeling.make_model_clipreid import load_clip_to_cpu
from modeling.clip.LoRA import mark_only_lora_as_trainable as lora_train


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # prompts: [B, 77, C]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)   # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)   # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # EOT token position
        x = x[torch.arange(x.shape[0], device=x.device),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """
    Class-agnostic learnable prompt:
    [SOS] + learnable context + [person/vehicle . EOS PAD ...]
    This keeps the same external interface as your current fixed-prompt code.
    """
    def __init__(self, dataset_name, dtype, token_embedding, n_ctx=4):
        super().__init__()

        if dataset_name == 'RGBNT201':
            class_name = "person"
        else:
            class_name = "vehicle"

        # initialize learnable context from a natural prompt prefix
        # current fixed prompt is "A photo of a person/vehicle."
        # here we let "a photo of a" become learnable context
        ctx_init = "a photo of a"
        ctx_dim = token_embedding.weight.shape[1]

        ctx_init = ctx_init.replace("_", " ")
        init_words = ctx_init.split(" ")
        assert len(init_words) == n_ctx, f"Expected n_ctx={n_ctx}, but got {len(init_words)} words."

        prompt = ctx_init + " " + class_name + "."
        tokenized_prompts = clip.tokenize(prompt)

        device = token_embedding.weight.device
        tokenized_prompts = tokenized_prompts.to(device)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        # learnable context initialized from "a photo of a"
        ctx_vectors = embedding[0, 1:1 + n_ctx, :].clone()   # skip SOS
        self.ctx = nn.Parameter(ctx_vectors)                 # [n_ctx, dim]

        # fixed parts
        self.register_buffer("token_prefix", embedding[:, :1, :])         # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :]) # class + "." + EOS + PAD
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        self.n_ctx = n_ctx

    def forward(self):
        ctx = self.ctx.unsqueeze(0)  # [1, n_ctx, dim]
        prompts = torch.cat(
            [
                self.token_prefix,   # [1, 1, dim]
                ctx,                 # [1, n_ctx, dim]
                self.token_suffix    # [1, *, dim]
            ],
            dim=1
        )
        return prompts


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory, feat_dim):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.in_planes = feat_dim
        self.cv_embed_sign = cfg.MODEL.SIE_CAMERA
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE
        self.direct = cfg.MODEL.DIRECT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            self.camera_num = camera_num
        else:
            self.camera_num = 0

        # No view
        self.view_num = 0

        if cfg.MODEL.TRANSFORMER_TYPE == 'vit_base_patch16_224':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
                img_size=cfg.INPUT.SIZE_TRAIN,
                sie_xishu=cfg.MODEL.SIE_COE,
                num_classes=num_classes,
                camera=self.camera_num,
                view=self.view_num,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate=cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE
            )
            self.clip = 0
            self.base.load_param(model_path)
            print('Loading pretrained model from ImageNet')
            if cfg.MODEL.FROZEN:
                lora_train(self.base)

        elif cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16':
            self.clip = 1
            self.sie_xishu = cfg.MODEL.SIE_COE
            clip_model = load_clip_to_cpu(
                cfg,
                self.model_name,
                cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0],
                cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1],
                cfg.MODEL.STRIDE_SIZE
            )
            print('Loading pretrained model from CLIP')
            clip_model.to("cuda")

            self.base = clip_model.visual
            self.device = next(clip_model.parameters()).device

            # ===== learnable prompt =====
            self.prompt_learner = PromptLearner(
                dataset_name=cfg.DATASETS.NAMES,
                dtype=clip_model.dtype,
                token_embedding=clip_model.token_embedding,
                n_ctx=4
            )
            self.text_encoder = TextEncoder(clip_model)

            # freeze text encoder; only learn prompt tokens
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            if cfg.MODEL.FROZEN:
                lora_train(self.base)

            if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_CAMERA:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(view_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(view_num))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

    def forward(self, x, label=None, cam_label=None, view_label=None, modality=None):
        if self.clip == 0:
            x = self.base(x, cam_label=cam_label, view_label=view_label)
            global_feat = x[:, 0]
            x = x[:, 1:]
            return x, global_feat, None

        else:
            if self.cv_embed_sign:
                cv_embed = self.sie_xishu * self.cv_embed[cam_label]
            else:
                cv_embed = None

            x = self.base(x, cv_embed, modality)

            # learnable prompt -> text feature
            prompts = self.prompt_learner()  # [1, 77, C]
            prompts = prompts.to(self.device)
            tokenized_prompts = self.prompt_learner.tokenized_prompts.to(self.device)
            text = self.text_encoder(prompts, tokenized_prompts)  # [1, C]

            global_feat = x[:, 0]
            x = x[:, 1:]

            return x, global_feat, text

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))