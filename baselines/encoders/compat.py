"""Compatibility with recent setuptools for the original code of the baselines."""
import sys


def pkg_resources_packaging():
    """PAC-S and SPECS do `from pkg_resources import packaging` (as the original CLIP code did), which
    recent setuptools no longer provides; expose the packaging module there so their code runs unchanged."""
    try:
        import pkg_resources
    except ImportError:
        import types
        pkg_resources = sys.modules["pkg_resources"] = types.ModuleType("pkg_resources")
    if not hasattr(pkg_resources, "packaging"):
        import packaging
        import packaging.version
        pkg_resources.packaging = packaging
