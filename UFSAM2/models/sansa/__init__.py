"""SANSA package with a lazy public builder import."""

__all__ = ["sansa", "adapter", "model_utils", "build_sansa"]


def __getattr__(name):
    if name == "build_sansa":
        from .sansa import build_sansa

        return build_sansa
    raise AttributeError(name)
