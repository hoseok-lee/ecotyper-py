API reference
=============

Most users only need :mod:`ecotyper.workflows` (the high-level workflows) and
:mod:`ecotyper.io` (readers and AnnData conversion). The other modules expose each
step of the R pipeline as a function, for custom pipelines and for checking
intermediate results against R.

High-level workflows
--------------------

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

   ecotyper.workflows
   ecotyper.io
   ecotyper.model
   ecotyper.recovery
   ecotyper.discovery
   ecotyper.discovery_scrna
   ecotyper.nmf
   ecotyper.rlayout
   ecotyper.validate
   ecotyper.cli
   ecotyper.rcompat
