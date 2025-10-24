import yaml

_GLOBAL_CONFIG = None
_GLOBAL_TOKENIZER = None


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def set_config(config):
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config


def get_config():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_CONFIG, "config")
    return _GLOBAL_CONFIG


def _build_tokenizer(config):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, "tokenizer")
    from data.multimodal_tokenizer import MultimodalTokenizer

    tokenizer_kwargs = {
        "use_text_instruction": config["use_text_instruction"],
        "add_round_idx": config["add_round_idx"],
        "audio_head_num": config["audio_head_num"],
    }

    # if config['conversation_bos_token'] is not None:
    #    tokenizer_kwargs["conversation_bos_token"] = config['conversation_bos_token']
    # if config['conversation_eos_token'] is not None:
    #    tokenizer_kwargs["conversation_eos_token"] = config['conversation_eos_token']
    _GLOBAL_TOKENIZER = MultimodalTokenizer(
        config["tokenizer_name_or_path"], **tokenizer_kwargs
    )
    return _GLOBAL_TOKENIZER


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, "tokenizer")
    return _GLOBAL_TOKENIZER


def set_global_variables(configs, build_tokenizer=True):
    """Set args, tokenizer."""
    _ensure_var_is_not_initialized(_GLOBAL_CONFIG, "config")
    set_config(configs)
    if build_tokenizer:
        _ = _build_tokenizer(configs)
