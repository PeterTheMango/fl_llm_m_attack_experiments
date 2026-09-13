"""Explicit FP32 model loading and NumPy-safe Flower transport."""


def load_causal_model(identifier, revision=None):
    import torch
    from transformers import AutoModelForCausalLM
    # The older spelling also works with the installed Transformers 4.x series.
    # Never inherit a checkpoint's BF16 dtype for full-model Adam/NumPy transport.
    return AutoModelForCausalLM.from_pretrained(identifier, revision=revision, torch_dtype=torch.float32)


def tensor_array(value):
    import torch
    value = value.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy().copy()


def model_parameters(model):
    return [tensor_array(value) for value in model.state_dict().values()]
