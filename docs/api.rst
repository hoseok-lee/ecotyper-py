API reference
=============

Most users only need the scverse API (:func:`ecotyper.recovery`,
:func:`ecotyper.discovery`, :class:`ecotyper.EcoTyperModel`). :mod:`ecotyper.workflows`
mirrors the R wrapper scripts on pandas objects and files, and :mod:`ecotyper.io`
reads the R pipeline's input files. The other modules expose each
step of the R pipeline as a function, for custom pipelines and for checking
intermediate results against R.

scverse API
-----------

.. autosummary::
   :nosignatures:

   ~ecotyper.recovery
   ~ecotyper.discovery
   ~ecotyper.infer_data_type
   ~ecotyper.EcoTyperModel

File-based and pandas workflows
-------------------------------

Recovery
~~~~~~~~

.. autosummary::
   :nosignatures:

   ~ecotyper.workflows.recover_bulk
   ~ecotyper.workflows.recover_bulk_frame
   ~ecotyper.workflows.recover_bulk_files
   ~ecotyper.workflows.recover_scrna
   ~ecotyper.workflows.recover_scrna_frame
   ~ecotyper.workflows.recover_scrna_files
   ~ecotyper.workflows.read_visium
   ~ecotyper.workflows.recover_visium_frame
   ~ecotyper.workflows.recover_visium_files
   ~ecotyper.workflows.ecotype_abundances_long

Discovery
~~~~~~~~~

.. autosummary::
   :nosignatures:

   ~ecotyper.workflows.discover_from_profiles
   ~ecotyper.workflows.discover_files
   ~ecotyper.workflows.read_profiles
   ~ecotyper.workflows.write_discovery
   ~ecotyper.workflows.discover_scrna
   ~ecotyper.workflows.discover_scrna_frame
   ~ecotyper.workflows.write_discovery_scrna
   ~ecotyper.workflows.DiscoveryResult

Modules
-------

.. autosummary::
   :toctree: generated
   :recursive:

   ecotyper.api
   ecotyper.workflows
   ecotyper.io
   ecotyper.builtin
   ecotyper.model
   ecotyper.recovery_steps
   ecotyper.discovery_steps
   ecotyper.discovery_scrna
   ecotyper.nmf
   ecotyper.rlayout
   ecotyper.validate
   ecotyper.cli
   ecotyper.rcompat
