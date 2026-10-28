# Distribution and source notices

This is the v0.2.0 source submission candidate for manuscript ESWA-D-26-33314.
It contains available source and documented engineering corrections, not a
complete or experimentally validated reproduction of the manuscript method.
The method coverage and remaining gaps are described in `docs/PAPER_CODE_MAP.md`.

Third-party source retains its original copyright notices and applicable
licenses. License texts are included with the vendored PEFT and Diffusers
source and under `licenses/`. Notices for ICEdit and PhotoDoodle identify
related source and external-weight provenance; their complete repositories
are not bundled as runtime dependencies. The concrete correspondence between
ICEdit's token MoE and the retained PEFT implementation is disclosed in
`docs/THIRD_PARTY_NOTICES.md`; its implementation lineage requires author
confirmation.

The supplied project did not establish a single license covering all
project-specific files. This packaging operation does not assign third-party
copyright to the manuscript authors or silently place all code under a new
MIT/Apache license. Any general public license for project-specific additions
must be determined by their rights holders. This document records the supplied
notices; it does not certify ownership or redistribution permission.

Model weights and dataset images are not included. The two partial LoRA exports
found in the original archive are associated with debugging paths and are not
authenticated as manuscript checkpoints. Their omission is recorded in
`provenance/EXCLUSIONS.json`; they are not described as Mock-generated weights.
External models and datasets retain their own access and redistribution terms.
A resource URL is not a redistribution grant.

Historical debugging logs and planning reports are author-facing evidence and
need not all appear in the reviewer package. Their omission does not remove
the obligation to disclose known method or engineering limitations. No missing
scientific algorithm, paper metric or experiment was supplied by this revision.

No public hosting, repository publication or submission to the journal was
performed in preparing this package.
