from pathlib import Path
from safetensors.torch import load_file

def resolve_safetensors_path(path_like, default_name="model.safetensors"):
    path = Path(path_like).expanduser().resolve()
    if path.is_dir():
        path = path / default_name
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_fuser_checkpoint(model, checkpoint_path, default_name="model.safetensors", strict=False):
    resolved_path = resolve_safetensors_path(checkpoint_path, default_name=default_name)
    state_dict = load_file(str(resolved_path))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return resolved_path, list(incompatible.missing_keys), list(incompatible.unexpected_keys)

def load_fuser_df_checkpoints(model, prior_checkpoint, surrogate_checkpoint, strict=False):
    prior_path = resolve_safetensors_path(prior_checkpoint)
    surrogate_path = resolve_safetensors_path(surrogate_checkpoint)

    prior_state_dict = load_file(str(prior_path))
    surrogate_state_dict = load_file(str(surrogate_path))

    prior_incompatible = model.load_state_dict(prior_state_dict, strict=strict)
    surrogate_incompatible = model.surrogate.load_state_dict(surrogate_state_dict, strict=strict)

    return {
        "prior_path": prior_path,
        "prior_missing_keys": list(prior_incompatible.missing_keys),
        "prior_unexpected_keys": list(prior_incompatible.unexpected_keys),
        "surrogate_path": surrogate_path,
        "surrogate_missing_keys": list(surrogate_incompatible.missing_keys),
        "surrogate_unexpected_keys": list(surrogate_incompatible.unexpected_keys),
    }

def log_checkpoint_info(LOGGER, cfg, checkpoint_info):
    if cfg.model.model_name == "FUSER":
        if checkpoint_info["missing_keys"]:
            LOGGER.warning("Missing keys when loading checkpoint: %s", checkpoint_info["missing_keys"])
        if checkpoint_info["unexpected_keys"]:
            LOGGER.warning("Unexpected keys when loading checkpoint: %s", checkpoint_info["unexpected_keys"])
    elif cfg.model.model_name == "FUSER_DF":
        filtered_prior_missing = [
            key for key in checkpoint_info["prior_missing_keys"]
            if not key.startswith("surrogate.") and not key.startswith("diff_scheduler.")
        ]
        if filtered_prior_missing:
            LOGGER.warning("Missing prior-model keys: %s", filtered_prior_missing)
        if checkpoint_info["prior_unexpected_keys"]:
            LOGGER.warning("Unexpected prior-model keys: %s", checkpoint_info["prior_unexpected_keys"])
        if checkpoint_info["surrogate_missing_keys"]:
            LOGGER.warning("Missing surrogate-model keys: %s", checkpoint_info["surrogate_missing_keys"])
        if checkpoint_info["surrogate_unexpected_keys"]:
            LOGGER.warning("Unexpected surrogate-model keys: %s", checkpoint_info["surrogate_unexpected_keys"])
    else:
        raise NotImplementedError