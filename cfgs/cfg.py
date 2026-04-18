from easydict import EasyDict as edict

cfg = edict()
cfg.seed = 12345
cfg.data_root = None
cfg.save_poses = False
cfg.output_dir = None
cfg.dataloader_num_workers = 0
cfg.test_batch_size = 1

# Dataset
cfg.data = edict()
cfg.data.benchmark_name = None

# Model
cfg.model = edict()
cfg.model.model_name = None
# ---- checkpoint ----
cfg.model.fuser_checkpoint = None
cfg.model.prior_checkpoint = None
cfg.model.surrogate_checkpoint = None
# ---- absolute geometric encoder ----
cfg.model.mink_in_feats_dim = 1
cfg.model.mink_conv1_kernel_size = 5
cfg.model.mink_normalize_feature = True
cfg.model.mink_bn_momentum = 0.05
# ---- geometric alternating attention ----
cfg.model.dec_embed_dim = 1024
cfg.model.dec_num_heads = 16
cfg.model.mlp_ratio = 4
cfg.model.dec_depth = 36
# ---- camera head ----
cfg.model.cam_dec_num_heads = 16
cfg.model.cam_out_dim = 512
# ---- pos embedding ----
cfg.model.pos_emb_scaling = 1.0
# ---- register token ----
cfg.model.num_register_tokens = 5

# SE(3)^N Diffusion Refinement
cfg.diffusion = edict()
cfg.diffusion.num_steps = 10
cfg.diffusion.add_noise = True
cfg.diffusion.sigma_r = 0.1
cfg.diffusion.sigma_t = 0.01