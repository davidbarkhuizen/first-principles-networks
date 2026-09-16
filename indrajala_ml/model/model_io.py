import json


def save_json(path: str, state: dict) -> None:
    """
    Bare open/json.dump wrapping, shared by every save() in this codebase whose envelope shape
    doesn't fit save_model_json's fixed layer_sizes/class_count fields (e.g.
    ConvMultiClassBackpropClassifierNetwork.save, which has conv hyperparameters instead).
    """
    with open(path, "w") as f:
        json.dump(state, f)


def load_json(path: str) -> dict:
    """The load-side counterpart to save_json - bare open/json.load, no envelope assumptions."""
    with open(path) as f:
        return json.load(f)


def save_model_json(
    path: str,
    *,
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    class_count: int,
    snapshot: object,
) -> None:
    """
    The shared JSON envelope behind MultiClassBackpropClassifierNetwork.save and
    EnsembleBackpropClassifierNetwork.save - both need exactly enough to reconstruct a
    network's shape (layer_sizes, dimension, input_bounds, class_count) plus its trained
    weights (snapshot); only what goes into snapshot and how many networks get built from this
    envelope differs between the two.
    """

    save_json(
        path,
        {
            "layer_sizes": layer_sizes,
            "dimension": dimension,
            "input_bounds": input_bounds,
            "class_count": class_count,
            "snapshot": snapshot,
        },
    )


def load_model_json(path: str) -> dict:
    """
    The load-side counterpart to save_model_json: reads the envelope back, with input_bounds
    already restored to tuples (JSON only has arrays, so a saved (lo, hi) tuple round-trips as a
    2-element list otherwise) - everything else in the envelope is returned as-is for the caller
    to reconstruct its own network shape(s) from.
    """

    state = load_json(path)
    state["input_bounds"] = [tuple(bound) for bound in state["input_bounds"]]
    return state
