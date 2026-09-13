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


def set_serialized_parameters(model, parameters):
    """Copy one Flower tensor at a time; avoid a second full state dict."""
    import torch
    from flwr.common import bytes_to_ndarray
    state = model.state_dict()
    if len(state) != len(parameters.tensors):
        raise ValueError("Model parameter count mismatch")
    with torch.no_grad():
        for destination, raw in zip(state.values(), parameters.tensors):
            source = torch.from_numpy(bytes_to_ndarray(raw))
            if source.shape != destination.shape:
                raise ValueError("Model parameter shape mismatch")
            destination.copy_(source)
