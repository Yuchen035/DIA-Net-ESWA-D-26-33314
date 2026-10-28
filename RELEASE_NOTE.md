# Release note and scope

This edition restores reviewer-facing documentation, license texts and dependency inventories to the supplied source collection. Historical provenance and verification records refer to the earlier 12 October 2026 package, not to this package as a whole. The manuscript-to-source map was prepared against the original submitted manuscript; revised-manuscript page numbers may differ.

One class docstring in the retained Diffusers snapshot was removed; no executable statement in that file was changed. The dependency files are inventories, not a verified environment lock. Model weights, data and complete experiment checkpoints are not included.

The supplied source does not provide a complete, integrated implementation of the manuscript's four-category semantic-conditioned dual-routing workflow or its full four-term training objective. Static package checks do not demonstrate experimental reproduction. No training, model inference, GPU use, model or dataset download was performed when preparing this edition.

The SHA-256 manifest is regenerated for this edition. Run `python tools/verify_package.py` for archive integrity and `python tools/review.py check` for a model-free static inspection.
