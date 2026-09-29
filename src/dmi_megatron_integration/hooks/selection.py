"""Hook-selection parsing for the Megatron integration."""

from __future__ import annotations


def parse_hook_selection(
    selection: str | None,
    *,
    default: str = "router-summary",
) -> set[str]:
    selected = {
        part.strip()
        for part in str(selection if selection is not None else default).split(",")
    }
    if "" in selected:
        raise ValueError(f"Invalid empty DMI hook selection entry: {selection!r}")
    if "none" in selected:
        if selected != {"none"}:
            raise ValueError("DMI hook selection 'none' cannot be combined with hook names")
        return set()
    return selected


# Names installed by startup; router-topk is the existing two-output alias.
HOOK_SELECTION_NAMES = frozenset({
    "router-summary", "router-logits", "router-entropy", "expert-counts",
    "router-topk", "router-topk-expert-ids", "router-topk-weights",
    "hidden-states", "resid_final", "moe-input", "moe-inverse-map",
    "moe-packed-weighted-output", "vocab-logits", "vocab-logits-topk",
    "loss-summary", "token-loss", "grad-norm", "q-weights", "k-weights",
    "router-weights",
})


def resolve_phase_hook_selections(
    common: str, *, train: str | None = None, valid: str | None = None,
    test: str | None = None, additional_names=(),
) -> dict[str, frozenset[str]]:
    """Resolve replacement overrides, validating names and expanding aliases."""
    selections = {}
    for phase, override in (("train", train), ("valid", valid), ("test", test)):
        names = parse_hook_selection(common if override is None else override)
        unknown = names - HOOK_SELECTION_NAMES - set(additional_names)
        if unknown:
            raise ValueError(f"Unknown DMI {phase} hook selections: {sorted(unknown)}")
        if "router-topk" in names:
            names = (names - {"router-topk"}) | {"router-topk-expert-ids", "router-topk-weights"}
        selections[phase] = frozenset(names)
    return selections


def hook_enabled_in_phase(hook, phase: str) -> bool:
    """Unrestricted by default for existing/custom direct hook bindings."""
    return phase in getattr(hook, "megatron_enabled_phases", ("train", "valid", "test"))


__all__ = ["parse_hook_selection", "resolve_phase_hook_selections", "hook_enabled_in_phase"]
