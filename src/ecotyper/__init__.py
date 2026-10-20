"""EcoTyper in Python, with numerical parity to the R implementation.

scverse-style API::

    import ecotyper as et

    et.recovery(adata, model="Carcinoma")         # bulk, scRNA-seq or Visium (inferred)
    model = et.discovery(adata_sc)                 # scRNA-seq discovery -> EcoTyperModel
    model.save("my_model"); et.EcoTyperModel.load("my_model")

File-based workflows mirroring the R scripts are in :mod:`ecotyper.workflows`.
"""
from .api import discovery, infer_data_type, recovery
from .model import EcoTyperModel

__version__ = "0.1.0"
__all__ = ["EcoTyperModel", "discovery", "infer_data_type", "recovery"]
