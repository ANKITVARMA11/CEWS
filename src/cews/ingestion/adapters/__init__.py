"""Source adapters. Each module defines one adapter class decorated with ``@register_adapter``.

Modules here are imported automatically by :func:`cews.ingestion.base.adapter_classes`, so a
new source only needs a new module and a registry entry in ``config/source_registry.yaml``.
Real adapters are added in Phase 4; until then every source is reported as skipped.
"""
