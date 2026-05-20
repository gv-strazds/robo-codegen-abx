"""Mock implementation of isaacsim.core.utils.semantics with in-memory label store."""

# Module-level label store: {prim_key: {instance_name: [labels]}}
_label_store = {}


def _get_prim_key(prim):
    """Extract a stable string key from a prim-like object.

    Tries (in order):
    1. prim.name (works for MockPickObj, Isaac Sim wrapped prims)
    2. prim.GetPrimPath() (works for raw USD prims)
    3. id(prim) as fallback

    Filters out auto-generated MagicMock names (contain 'mock.' or 'MagicMock').
    """
    name = getattr(prim, 'name', None)
    if name is not None and isinstance(name, str) and 'mock.' not in name.lower() and 'magicmock' not in name.lower():
        return name
    get_path = getattr(prim, 'GetPrimPath', None)
    if get_path is not None:
        try:
            path = str(get_path())
            if path and 'mock.' not in path.lower():
                return path
        except Exception:
            pass
    return f"__id_{id(prim)}"


def add_labels(prim, labels=None, instance_name="class", overwrite=True):
    """Store semantic labels for a prim in the in-memory label store."""
    key = _get_prim_key(prim)
    if key not in _label_store:
        _label_store[key] = {}
    if overwrite or instance_name not in _label_store[key]:
        _label_store[key][instance_name] = list(labels) if labels else []
    else:
        existing = _label_store[key].get(instance_name, [])
        existing.extend(labels or [])
        _label_store[key][instance_name] = existing


def get_labels(prim):
    """Retrieve semantic labels for a prim from the in-memory label store."""
    key = _get_prim_key(prim)
    stored = _label_store.get(key, {})
    return {k: list(v) for k, v in stored.items()}


def set_labels_by_name(prim_name, labels_dict):
    """Convenience: directly set labels for a prim by name string.

    Args:
        prim_name: String key (prim name).
        labels_dict: Dict of {instance_name: [labels]}.
    """
    _label_store[prim_name] = {k: list(v) for k, v in labels_dict.items()}


def clear_all_labels():
    """Clear the entire label store (useful between test runs)."""
    _label_store.clear()


def add_update_semantics(prim, semantic_label, type_label="class", suffix=""):
    """Mock for deprecated SemanticsAPI. Routes to add_labels."""
    add_labels(prim, labels=[semantic_label], instance_name=type_label)


def remove_all_semantics(prim, recursive=False):
    """Mock for deprecated remove_all_semantics."""
    key = _get_prim_key(prim)
    _label_store.pop(key, None)


def remove_labels(prim, instance_name=None, include_descendants=False):
    """Mock for remove_labels."""
    key = _get_prim_key(prim)
    if instance_name is None:
        _label_store.pop(key, None)
    elif key in _label_store:
        _label_store[key].pop(instance_name, None)
