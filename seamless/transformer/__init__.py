try:
    from seamless_transformer import *

    __all__ = [
        "direct",
        "delayed",
        "Transformer",
        "Transformation",
        "parallel",
        "parallel_async",
        "TransformationList",
        "spawn",
        "has_spawned",
    ]

except ImportError:
    __all__ = []


def __getattr__(name):
    import seamless_transformer

    return getattr(seamless_transformer, name)
