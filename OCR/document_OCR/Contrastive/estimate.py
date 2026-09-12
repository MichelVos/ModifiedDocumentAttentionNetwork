import torch
import torch.nn as nn
from collections import OrderedDict

def _num_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def _to_mb(nbytes: int) -> float:
    return nbytes / (1024**2)

def profile_activation_memory(
    model: nn.Module,
    input_shape=(1, 1, 32, 1300),   # (N, C, H, W) for images
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float32,
    train_mode=True,                # True → simulates training (activations kept for backward)
    verbose=True
):
    """
    Returns:
        report: list of dict rows (layer_name, out_shape, out_MB)
        totals: dict with totals for activations / params / grads / optimizer est.
    """
    model = model.to(device=device, dtype=dtype)
    model.train(train_mode)

    # Make a dummy input that requires grad if we're simulating training
    x = torch.randn(*input_shape, device=device, dtype=dtype, requires_grad=train_mode)

    # Collect parameter sizes
    param_bytes = 0
    for p in model.parameters():
        param_bytes += _num_bytes(p)

    # Register hooks to capture outputs
    handles = []
    report_rows = []
    layer_id = 0

    def hook_fn(name):
        def _hook(module, inp, out):
            nonlocal layer_id
            layer_id += 1

            # Normalize output(s) to a list of tensors
            tensors = []
            if torch.is_tensor(out):
                tensors = [out]
            elif isinstance(out, (list, tuple)):
                tensors = [t for t in out if torch.is_tensor(t)]
            else:
                tensors = []

            out_bytes = sum(_num_bytes(t) for t in tensors)
            out_shapes = [tuple(t.shape) for t in tensors]

            report_rows.append(OrderedDict(
                layer_id=layer_id,
                name=name,
                out_shapes=str(out_shapes) if out_shapes else "[]",
                out_MB=round(_to_mb(out_bytes), 3),
            ))
        return _hook

    # Name modules for a nice report (skip the root)
    named_modules = [(n, m) for n, m in model.named_modules() if n]

    for name, module in named_modules:
        try:
            h = module.register_forward_hook(hook_fn(name))
            handles.append(h)
        except Exception:
            pass  # some modules may not support hooks cleanly

    # Do a forward pass
    with torch.set_grad_enabled(train_mode):
        _ = model(x)

    # Cleanup hooks
    for h in handles:
        h.remove()

    # Summaries
    total_activation_mb = round(sum(r["out_MB"] for r in report_rows), 3)

    # Training-time memory rule of thumb:
    # - Forward activations must be kept for backward → ~1× activation bytes
    # - Gradients for activations are transient but add overhead during backward
    # - Param gradients ~= params in size
    # - Optimizer states: Adam/AdamW ~ 2× params
    #
    # We’ll report:
    #   activations_fwd_mb ~ sum of recorded outputs
    #   params_mb
    #   param_grads_mb ~ params_mb (if training)
    #   optim_states_mb ~ 2× params_mb (if Adam/AdamW)
    params_mb = round(_to_mb(param_bytes), 3)
    param_grads_mb = params_mb if train_mode else 0.0
    optim_states_mb = round(2.0 * params_mb, 3) if train_mode else 0.0

    totals = {
        "activations_forward_MB": total_activation_mb,
        "params_MB": params_mb,
        "param_grads_MB": param_grads_mb,
        "optimizer_states_MB_est": optim_states_mb,
        "rough_peak_training_MB": round(
            total_activation_mb + params_mb + param_grads_mb + optim_states_mb, 3
        ),
        "dtype": str(dtype),
        "device": device,
        "train_mode": train_mode,
        "input_shape": input_shape,
    }

    if verbose:
        # Pretty print
        print(f"\n=== Activation profile (dtype={dtype}, device={device}, train_mode={train_mode}) ===")
        print(f"Input shape: {input_shape}")
        print(f"{'ID':>3}  {'Layer':<55} {'Out shapes':<40} {'Out MB':>8}")
        for r in report_rows:
            print(f"{r['layer_id']:>3}  {r['name']:<55} {r['out_shapes']:<40} {r['out_MB']:>8.3f}")
        print("\n--- Totals (approx) ---")
        for k, v in totals.items():
            if k.endswith("_MB") or k.endswith("_MB_est") or k.startswith("rough_peak"):
                print(f"{k:>28}: {v:>8.3f} MB")
        print()

    return report_rows, totals

# ---------- Example usage ----------
# Suppose `encoder` is your DAN encoder module
# from your_model import encoder

# 1) Line strip
# _, totals_lines = profile_activation_memory(encoder, input_shape=(1, 1, 32, 1300))

# 2) Full page (example 1x1x1000x1000)
# _, totals_pages = profile_activation_memory(encoder, input_shape=(1, 1, 1000, 1000))

# Print a quick comparison (uncomment after running the two lines above)
# print("Forward activations (MB) — lines vs pages:",
#       totals_lines["activations_forward_MB"], "→", totals_pages["activations_forward_MB"])
