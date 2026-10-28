> Historical v0.2 record: statements and checks below refer to the 12 October 2026 package, not the current package. See RELEASE_NOTE.md for current scope.

# Checks performed for version 0.2.0

These checks validate source syntax, selected engineering contracts and package
integrity. They do not validate the model, the complete manuscript method, or
any reported numerical result. Preparation date: 12 October 2026.

## Performed

- Parsed all 1,062 packaged Python files with the host Python 3.14 AST parser.
  This does not prove execution compatibility with a Python 3.10 environment.
- Ran 16 isolated checkpoint-contract checks with fake model/tensor/serializer
  objects. These cover raw standard/expert/router keys and full configuration,
  plus rejection of missing metadata, changed hashes, wrong architecture,
  missing fields, empty state, shape/key mismatches, unsupported legacy files
  and multiple adapters. CLI help and legacy refusal precede model imports.
  The fake serializer does not validate real Safetensors I/O or PyTorch APIs.
- Ran 11 release-boundary checks, including the AST check above. Import/process
  guards confirmed that inspection and dry-run plans do not import model
  libraries or launch a child workload. Unresolved execution plans are refused;
  the explicit base-only option is forwarded. An isolated optimizer factory
  test confirms that missing AdamW8bit does not invoke ordinary AdamW.
- Compared the main Trainer methods against v0.1: only `_run_validation`
  changed. The training objective and main loop methods are unchanged.
- Compared bundled vendor trees against v0.1: all 193 PEFT-distribution files
  and 780 Diffusers-distribution files are byte-identical. The original local
  PEFT changes and supplied license texts therefore remain intact.
- Regenerated the source manifest, engineering patch and SHA-256 file list.
  Source-to-original mappings and candidate-to-parent changes are recorded in
  `provenance/`. Final ZIP/re-extraction results and the ZIP hash are recorded
  in the separate delivery-verification file, avoiding a self-referential hash.

Detailed results are in `adapter_contract_results.json` and
`release_boundary_results.json`. The check scripts are retained in the separate author check-record archive.
`tools/review.py check` intentionally has the narrower entry-point scope of
19 parsed files; it should not be confused with the all-source AST count.

## Not performed

No model-library import, dependency installation, model loading, real tensor
save/load round trip, training, model inference, model/data download, GPU use,
benchmark evaluation, weight conversion or metric recomputation was performed.
No archived model/GPU tests were run. The selected runtime environment and
external model/data assets remain unverified. No new scientific algorithm was
added to fill the manuscript coverage gaps.

The new checkpoint format is an engineering correction for future checkpoint
pairs produced by this version. It is not a recovered manuscript checkpoint or
evidence that historical weights implement the reported model. The model-free
fake-object checks and their passing results must not be reported as experimental
reproduction or a successful real-model save/load test.
