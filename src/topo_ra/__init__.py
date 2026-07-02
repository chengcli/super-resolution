"""TopoRA terrain-aware wind downscaler.

Vendored subset of the private TopoRA research code (v0.1.0): model, online
training, and data utilities needed to run the snapy online-training sidecar.
Offline distillation and the FuXi-CFD teacher stack are intentionally not
included.
"""

from topo_ra.models.topo_ra import TopoRA

__all__ = ["TopoRA"]
__version__ = "0.1.0"
