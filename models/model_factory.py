from models.FUSER import FUSER
from models.FUSER_DF import FUSER_DF
from utils.checkpoints import load_fuser_df_checkpoints, load_fuser_checkpoint

SUPPORTED_MODEL_NAMES = ("FUSER", "FUSER_DF")


def build_model(cfg):
    if cfg.model.model_name == "FUSER":
        return FUSER(cfg)
    if cfg.model.model_name == "FUSER_DF":
        return FUSER_DF(cfg)
    raise ValueError(f"Unsupported model_name: {cfg.model.model_name}")

def load_model_weights(model, cfg):
    if cfg.model.model_name == "FUSER":
        checkpoint_path, missing_keys, unexpected_keys = load_fuser_checkpoint(
            model,
            cfg.model.fuser_checkpoint,
            strict=False,
        )
        return {
            "checkpoint_path": checkpoint_path,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }
    elif cfg.model.model_name == "FUSER_DF":
        return load_fuser_df_checkpoints(
            model,
            cfg.model.prior_checkpoint,
            cfg.model.surrogate_checkpoint,
            strict=False,
        )
    else:
        raise NotImplementedError
