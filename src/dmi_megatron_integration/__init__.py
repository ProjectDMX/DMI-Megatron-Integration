"""DMI integration for ProjectDMX Megatron-LM-DMI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("DMI-Megatron-Integration")
except PackageNotFoundError:  # Source tree used without installation.
    __version__ = "0.17.1"

__all__ = ["__version__"]
