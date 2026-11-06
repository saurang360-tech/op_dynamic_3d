# Implementation Plan: Six Open Questions

**Status:** saved before source edits; implementation was then completed in the planned order. Existing sweep HDF5 files remain untouched.

## Chosen order and rationale

1. **Q6 — Fail loudly for unsupported RDN methods.** Prevents future runs from silently generating apparently valid data for requested methods 2/3. It affects only those currently unsupported selections.
2. **Q1 — Preserve and explicitly document RK4 stage-wise resampling.** Behavior-preserving comments/doc updates first; avoid changing RNG draw count/order or the stochastic integration scheme.
3. **Q2 — Preserve cadence, rename the counter to call-based terminology.** Behavior-preserving internal rename/docs. Keep the exact counter lifetime and refresh calls, including persistence across runs.
4. **Q3 — Normalize good probability rows even when other rows use fallback.** A numerical correction, isolated and tested with deterministic synthetic rows before proceeding.
5. **Q4 — Warn on ill-conditioned metrics and clamp very small distances.** A numerical edge-case change after Q3 so fallback normalization is already correct. Fixed-seed ordinary-parameter comparisons will establish whether ordinary trajectories are untouched.
6. **Q5 — Save the true integrated endpoint in `final_opinions`, preserving the prior snapshot under `last_history_step`.** Do this last because it changes HDF5 values/schema and the pre-to-post starting state; update writers/readers/visualization together without rewriting old output files.

This order puts the loud failure ahead of changes that might otherwise produce silent bad output, preserves existing stochastic semantics before numerical corrections, and leaves the consumer-facing output change until all simulation behavior changes are settled.

## Per-question decisions and validation

### Q6 — Unsupported RDN methods

- **Intended change:** Keep methods 0 and 1 exactly as implemented; `calculate()` raises `NotImplementedError` for method values 2, 3, and other unsupported values. Document the implemented methods.
- **Numerical effect:** None for methods 0/1. Unsupported selections change from a silent zero nudge to a visible exception by design.
- **Validation:** Add tests for method 0, fixed-seed method 1, and exceptions for 2, 3, and an unknown value. Run the existing seeded baseline with method 1 and require exact equality.
- **Rollback:** Revert only the `calculate()` unsupported case and its test/doc change; leave supported method behavior untouched.

### Q1 — RK4 network resampling

- **Decision:** Preserve the current behavior: each of the four RK4 RHS calls draws a new adjacency matrix (and a new RDN sample when enabled). Document this as the model's current stage-wise stochastic semantics in an inline source comment and the audit document. Do not claim it is classical deterministic RK4.
- **Numerical effect:** None; no draw count/order or stage evaluation changes.
- **Validation:** Add a behavioral test asserting four RHS/network constructions per RK4 step and their stage states/order; fixed-seed short-run comparison must match the pre-change baseline exactly.
- **Rollback:** Revert only the explanatory comment/documentation. No model code path is changed.

### Q2 — `p_conn` refresh cadence

- **Decision:** Preserve the existing every-10-dynamics-call refresh and counter persistence across runs. Rename private state to call-based names (for example `_p_conn_dynamics_call_counter` and `_p_conn_refresh_every_dynamics_calls`) and document that Euler refreshes each 10 calls while RK4 refreshes at stages 10, 20, etc. Do not reset the counter in run methods.
- **Numerical effect:** None; refresh points and random draw order stay identical.
- **Validation:** Add Euler/RK4 counter and refresh-count tests, including a second run on the same instance; compare fixed-seed trajectories and `p_conn` exactly against the baseline.
- **Rollback:** Restore the old private attribute names and update only tests/docs; preserve the unchanged counter logic.

### Q3 — Probability fallback normalization

- **Intended change:** Compute bad-row mask from row sums, assign uniform probabilities to bad rows, and normalize all other rows independently in either branch. Preserve diagonal zeros and current fallback definition (NaN or zero sum).
- **Numerical effect:** None when all probability rows are valid (expected normal case). Rows that previously remained unnormalized because another row triggered fallback will change intentionally.
- **Validation:** Add synthetic mixed good/bad row tests that check every row sum, zero diagonal, and uniform fallback. Compare ordinary seeded `p_conn` and simulation arrays exactly. Inspect existing HDF5 keys: the current schema stores opinions/time/final opinions, not `p_conn`, so probability row sums cannot be checked directly in historical files. Do not infer otherwise. As an additional check, run a fixed-seed representative simulation with fallback-row instrumentation and report whether it occurs.
- **Rollback:** Revert only the normalization branch if ordinary or synthetic tests fail; do not change distance, RNG, or topology behavior.

### Q4 — Degenerate metric and zero distances

- **Intended change:** Warn when `Phi` is singular or severely ill-conditioned (based on its condition number), and clamp distances to a documented positive float32 epsilon before raising them to `-beta`. Use an epsilon on the order of float32 machine precision times the project's initial opinion scale (initial opinions have standard deviation 3), rather than a large scientific cutoff. Delta 90 degrees produces approximately the identity metric and is not singular; degeneracy occurs near multiples of 180 degrees for this `Phi` definition.
- **Numerical effect:** Ordinary finite distances above epsilon are unchanged. Exact/near-zero distances and singular-metric runs change from possible infinities/NaNs to finite weights. The clamp can affect extremely close, nonzero pairs; this is an explicit, localized model-numerics change.
- **Validation:** Add tests for warning at delta 0 (and another degenerate angle), no warning at delta 90, finite normalized `p_conn` for coincident opinions, and ordinary seeded delta=50 equality against the pre-change baseline. Also check NaN/Inf handling.
- **Rollback:** Revert the clamp/warning only if ordinary trajectories change or degenerate tests fail; keep Q3 fallback normalization independently.

### Q5 — True final opinions and output schema

- **Decision:** Choose option (a). Save `sn.get_opinions()` as `final_opinions` and save the existing `result.y[:, :, -1]` under `last_history_step`. Keep history length, values, and the existing `final_opinions` key. Update all three HDF5 writing paths (master pre/post, standalone pre/post as applicable), return values, and readers/visualization compatibility. Existing HDF5 files remain untouched.
- **Numerical effect:** Changes the `final_opinions` dataset by one completed integration update. The post-RDN phase in the master workflow will consequently start from the true pre-RDN endpoint for newly generated runs, so post trajectories change relative to old outputs. Opinion histories themselves are unchanged by this output correction. `last_history_step` preserves access to the previous saved value.
- **Validation:** Add writer tests for both keys and their values; verify loader and visualizer still select `final_opinions`; fixed-seed test that it equals the final in-memory endpoint while `last_history_step` equals the last history frame. Compare histories exactly with baseline and explicitly quantify the intentional endpoint delta. Do not rewrite the 300 existing sweep files.
- **Rollback:** Restore writers to old final-value behavior and remove the extra key/readers only from newly generated files; leave existing data unchanged.

## Global validation and reproducibility controls

- Keep old and new HDF5 data side by side only in temporary test directories; never write under `sweep_1/` or `sweep_2/`.
- Before numerical changes, capture fixed-seed baselines for an ordinary parameter set (`delta=50`, no degeneracy), Euler, RK4, RDN method 0/1, and HDF5 values/schema.
- Run focused tests and fixed-seed comparisons immediately after each question before editing for the next one.
- For Q3/Q4, explicitly distinguish expected equality in ordinary cases from intentional differences for fallback/degenerate inputs.
- At completion, run the full test suite and report exact/non-exact comparisons, warning behavior, output schema, and any limitations caused by historical files not containing `p_conn`.


## Completion note

All six items were implemented and validated one at a time. The final suite has 23 passing tests. The historical HDF5 schema contains no connection-probability matrix, so historical row sums could not be inspected directly. The fixed-seed ordinary trajectory remained exact through Q1-Q4; Q5 intentionally updates only endpoint semantics and preserves the prior snapshot as `last_history_step`.
