# Source Audit and Optimization Plan

**Scope:** this directory's nine Python modules, `config.json`, sweep log, and generated HDF5 outputs, checked against the accompanying pasted `PROJECT_HANDOFF.md`. No source edits were made before completing this audit and plan.

## Repository inventory and state

- Python modules: `ODE_solver.py`, `RDN.py`, `master.py`, `runner_pre_RDN.py`, `runner_post_RDN.py`, `Bulk_visualiser.py`, `h5_loader.py`, `stats.py`.
- Configuration: `config.json`; existing simulation outputs are in `sweep_1/` and `sweep_2/`; `sweep.log` is present.
- No test files or in-folder documentation were found. The handoff is supplied as an attachment, not present in this directory.
- The containing Git worktree has extensive unrelated staged/untracked content (including an environment and sibling projects); work must be limited to this project folder. No broad Git cleanup/reset is appropriate.
- Existing HDF5 artifacts must be read only during audit; do not rewrite them.
- Existing generated data inventory: 300 HDF5 files across `sweep_1/` and `sweep_2/`; sampled files have keys `opinions`, `time`, `final_opinions`, shapes `(1000,2,400)`, `(400,)`, `(1000,2)`, and 20 attributes. Files were read-only inspected.
inal_opinions, shapes (1000,2,400), (400,), (1000,2), and 20 attributes. Files were read-only inspected.

## Actual implementation reconstructed from source

Notation: `X` is an `(N,2)` float32 opinion matrix, `Phi=[[1,c],[c,1]]` with `c=cos(radians(delta))`, and `A` is an `(N,N)` int8 adjacency matrix.

1. **Metric distance.** `social_network.dist(A1,A2)` computes `d=A1-A2`, then `sqrt((Phi @ d)^T (Phi @ d))`. `connect_prob_matrix` instead forms pairwise row differences `d_ij=X_i-X_j`, computes `d_ij @ Phi`, and takes its Euclidean norm. `Phi` is symmetric, so these represent the same metric for ordinary finite 2-vectors. Batched probability calculation currently allocates `diff (N,N,2)`, loops over source rows in Python, and stores `distances (N,N)`.
2. **Connection probability.** Off-diagonal values are `distance_ij**(-beta)` with diagonal zero, then each row is divided by its row sum. The current fallback branches on whether *any* row sum is NaN/zero: when true, only such rows are assigned uniform values and all other rows are left unnormalized. Infinite distances/weights can also propagate NaNs through division. This is actual behavior and must not be silently changed.
3. **Activity.** One legacy global NumPy uniform draw per agent, `U_i`, then `a_i=[eps**(1-gamma)+(1-eps**(1-gamma))*U_i]**(1/(1-gamma))`; stored float32. `set_activity` runs once before integration in the supplied runners.
4. **Adjacency and orientation.** Each dynamics evaluation draws `N` activation uniforms. For each active agent `j`, it samples `m` distinct targets from row `p_conn[j]`, and sets `A[target,j]=1`. Thus columns encode active sources; `A @ influence` sums incoming source influences by target. A Python loop performs each `np.random.choice`; this order is scientifically sensitive. The zero-probability fallback is triggered when the number of zero entries exceeds `N-m`.
5. **Opinion influence.** For each dynamics state `X`, source code computes `transformed=X @ Phi`; then `g[:,0]=tanh(alpha*transformed[:,0])`, `g[:,1]=tanh(alpha2*transformed[:,1])`. Base derivative is `dX/dt=-X+K*(A @ g)`. The docstring's matrix-order claim is not authoritative; current `Phi` is symmetric.
6. **RDN.** For `method==1`, compute the selected topic's scalar global mean, draw `n` indices without replacement, and return `sqrt(n)*(sample_mean-global_mean)`. `op_dynamics` adds `D*nudge` to that topic for every agent. It is a shared scalar drift. Other method numbers currently return `0.0`; topic `2` is not a supported index in this implementation. RDN draws occur once per `op_dynamics` call when enabled.
7. **Euler.** One `op_dynamics` call and one freshly sampled adjacency per step; then in-place `y += dy*dt`. State arrays/history/time are float32. `result.y[:,:,i]` records the state *before* step `i`; the final post-integration state is `self._opinions` and is not in the history. Runners/master currently serialize the last history frame as `final_opinions`, which is one Euler/RK4 update behind the actual final state.
8. **RK4.** Four `op_dynamics` calls per step at `y`, `y+dt*k1/2`, `y+dt*k2/2`, and `y+dt*k3`. Every call draws a new activation vector and new sampled adjacency, and (if enabled) a new RDN sample. Therefore this is stochastic stage-wise resampling, not classical RK4 over one fixed RHS/network. Keep all four calls and their RNG order unchanged in any safe optimization.
9. **`p_conn` cadence.** Constructor starts the counter at zero and frequency at 10. `set_social_interactions` computes an initial matrix but does not reset/increment the counter. Every `op_dynamics` call increments it and refreshes the matrix exactly when the counter is divisible by 10, from that call's stage state. Euler means every 10 time steps; RK4 means every 10 stage calls (alternating 2 and 3 steps, average 2.5). A run made after a previous run continues the counter. This is implementation behavior and a documented-vs-step cadence discrepancy, not changed here.
10. **Data/analysis.** Writers save `opinions`, `time`, `final_opinions` and attributes using gzip for opinion histories/final arrays (time uncompressed); master and standalone filename patterns differ. `Bulk_visualiser.py` reads `final_opinions`, stacks files, filters outliers and renders per-agent histogram-density coloring. `h5_loader.py` has fixed-path single-file loaders and a folder-based multiple loader. `stats.py` computes basic moments, outlier masks, and quadrant counts. Configuration defaults include N=1000, dt=0.01, duration=3; full production sweeps are much larger than the small baseline below.

## SCIENTIFICALLY SENSITIVE OPERATIONS

- Global NumPy RNG state, call order, and legacy `np.random` API: changing any draw or vectorization order can change all later runs.
- Random active-agent selection, per-agent `choice` and no-replacement sampling: affects network topology and its directed orientation.
- RDN sample count, topic, sample timing, and shared scalar application: changes the intervention distribution.
- RK4 stage-specific graph/RDN resampling: changing to a fixed graph or reusing draws changes the stochastic process.
- `p_conn` refresh counter and its persistence: changes which opinion state governs subsequent edges.
- `Phi` orientation and `alpha`/`alpha2` topic assignment, plus `tanh` placement: changes the coupled dynamics.
- Distance exponent and row normalization including exceptional rows: changes interaction probabilities.
- float32 coercions, operation order, and time-step truncation (`int(t_dur/dt)`): may alter trajectories or number of steps.
- History snapshots exclude the post-final state; downstream HDF5 values rely on this existing convention.
- HDF5 keys, dtypes, compression choices, and filenames are consumer-facing data contracts.

## Baseline and profiling (MEASURED)

Measured in this environment using Python 3.12.10 and NumPy 2.5.1, fixed seed 1729, `N=160`, `m=5`, `dt=0.01`, duration `0.10`, RK4, RDN method 1 (`n=10`):

- Setup: 0.00535 s; integration: 0.09059 s; whole run: 0.09594 s (single short run; not a robust benchmark).
- `tracemalloc` peak during this run: 716,873 bytes. This excludes some native NumPy allocation details and tracing affects timing.
- History `(160,2,10)`, time `(10,)`, final state `(160,2)`; all final opinions finite; checksum of current final state `-105.44085693359375`, history checksum `-1125.013671875`.
- 40 RHS calls for 10 RK4 steps; counter ended at 40.
- Twelve-call cProfile: `op_dynamics` 0.012 s cumulative; `create_adj_matrix` 0.009 s over 12 calls; initial `connect_prob_matrix` 0.002 s once; remaining self time included NumPy random choice and array reductions. This small N profile points to adjacency sampling as a likely cost, but is not sufficient to assert a speedup at production N.
- At N=160, `_adj` occupies 25,600 bytes; `p_conn` 102,400 bytes; opinions 1,280 bytes. Core pairwise difference temporary scales as `8*N*N` bytes for two float32 coordinates; a full extra transformed pair array would add similar storage.
- Candidate distance-loop benchmark at N=1000: row-by-row peak 12,021,524 traced bytes; 64-row chunks 13,281,540 bytes (increase 1,260,016 bytes); full batching 28,001,380 bytes. Chunking avoids the full batch's roughly 16 MB peak increase.
- Baseline trajectory/time/activity/p_conn arrays were captured outside the workspace at `%LOCALAPPDATA%\Temp\codex_opdyn_baseline.npz` for post-change deterministic-seed comparison.


### Post-change measurements

- Full `connect_prob_matrix` refresh at N=1000: prior row-loop median 40.490 ms; 64-row chunk implementation median 34.964 ms (five measured calls each; 13.6% lower). Probability arrays were exactly equal (`max_abs_diff=0`).
- Short fixed-seed N=160 RK4 run comparing both original code paths with both optimized code paths: previous median 32.331 ms; updated median 25.635 ms (five measured calls each; 20.7% lower). Runs are short and noisy, so this is directional, not a production throughput guarantee.
- The original 10-step seeded baseline trajectory remained exactly equal for history, time, in-memory endpoint, activity values, and final connection-probability matrix.
- Adjacency fallback condition microbenchmark at N=1000: 1,000 row checks median 5.106 ms for `sum(row == 0) > N-m` and 0.931 ms for `count_nonzero(row) < m` (seven batches). The boolean decisions agree because `count_nonzero(row) = N - count(row == 0)`, including NaNs (counted nonzero by both formulations).
- All ten behavioral tests pass. `compileall` succeeds. Existing output samples retained the expected schema; 300 generated files were found and left untouched.
## CODE-BASED COMPLEXITY ANALYSIS

- Probability matrix: O(N²) arithmetic and at least O(N²) output/storage, with current `(N,N,2)` differences and Python loop over N source rows. It is recomputed initially and every 10 RHS calls.
- Adjacency generation: O(N²) probability-row inspection in active-agent fallback checks plus up to O(N*m) weighted sampling calls; repeated once per Euler step or four times per RK4 step. Sampling cost is NumPy-dependent and cannot be accurately inferred from asymptotics alone.
- Opinion influence: each RHS has `A @ g`, nominal O(N²) dense multiplication despite sparse-like adjacency; four times per RK4 step. `tanh` and 2D transforms are O(N).
- History allocation: O(N*2*steps) float32, plus full HDF5 write/read and gzip cost.
- Measured candidate microbenchmark (N=1000, three repetitions, median, tracing enabled): per-row distance loop 39.0 ms/12.0 MB peak; 64-row chunks 26.0 ms/13.3 MB; single full batch 26.4 ms/28.0 MB. All candidate distance arrays matched the row-loop result exactly. This is a microbenchmark, not a full-run speedup claim. Impact on whole-run time may be limited because adjacency sampling and dense matrix multiplication repeat more often.

## Optimization plan (written before source edits)

| Optimization | Current Cost | Proposed Change | Expected Benefit | Scientific Risk | Validation Method |
| --- | --- | --- | --- | --- | --- |
| Batch the existing pairwise metric transform in 64-row chunks | O(N²) data plus N Python loop iterations; two-coordinate temporaries | Replace per-row loop with bounded chunks using `diff[lo:hi] @ Phi` and the same square/sum/sqrt axes | Measured 33% faster for distance calculation at N=1000; same formula and float32 operands | Low-to-medium: backend batching may alter last-bit rounding; chunk temporaries increase peak memory modestly; keep draw order untouched | Compare baseline p_conn and full seeded trajectory; strict/near equality and max error; separately time refresh at N=1000 |
| Remove temporary boolean-array reduction from adjacency fallback checks | O(N) scan and temporary boolean vector per active agent | Replace `sum(probs == 0) > N-m` with the equivalent `count_nonzero(probs) < m` | Measured 5.11 ms to 0.93 ms across 1,000 row checks; reduces temporary allocation | Very low: integer identity `N-zero_count < m`; RNG calls and row values unchanged | Verify truth-value equivalence over zero, finite, infinite, and NaN rows; compare seeded adjacency and full seeded trajectory |
| Defer other numerical optimizations | Adjacency sampling and dense matrix products repeat each stage; RNG sampling controls behavior | Preserve these paths until separately measured | Avoid unsupported performance claims and stochastic drift | Low | Profile representative N and measure Euler/RK4 plus peak memory before proposing another change |
| Behavioral tests for numerical contracts | No test suite found | Add deterministic tests for metric, probability normalization/current edge behavior, orientation, activity/RDN RNG, Euler/RK4 call counts, HDF5 schema/load | Protect model and output contracts before future edits | Low; tests codify present behavior including oddities | Run tests against pre-change and post-change code; fixed seeds and explicit expected arrays |
| Audit/report documentation | Handoff is external and has some unchecked statements | Keep this source-grounded audit, sensitivity list, baseline and plan in repository | Maintainable record of choices and uncertainties | None to simulation | Review formulas/code correspondence; ensure measured/hypothesis labels remain explicit |

Initial optimization work was limited to safe vectorization and allocation reductions. The six subsequent open questions are resolved below with explicit decisions.

## Discrepancy report: historical `final_opinions` values

**HISTORICAL IMPLEMENTATION:** integration saves each pre-step state to history; old writers stored `result.y[:,:, -1]` under HDF5 `final_opinions`.

**RESOLUTION:** newly written files now store the post-final-step state (`sn.get_opinions()`) in the stable `final_opinions` key and retain the historical value under `last_history_step`.

**SCIENTIFIC CONSEQUENCE:** final distributions and new post-RDN starting states change by one integration update. The history itself is unchanged.

**MIGRATION:** existing sweep files are not rewritten; they continue to carry their historical value under `final_opinions` and do not have `last_history_step`.


## Additional source-vs-handoff gaps

- `r` is stored and passed through configuration but is not read by `op_dynamics`, `create_adj_matrix`, or the runners' simulation equations; no reciprocity operation is implemented in this source snapshot.
- `run_network_rk4` accepts `N` and `D`, passes them into `op_dynamics`, but `op_dynamics` never uses either argument.
- `set_social_interactions` stores `r` without a conversion and sets `beta`; no network rule uses `r` afterward.

These are descriptive audit findings only. Their intended scientific meaning and any correction require researcher input.
## Remaining scientific uncertainties

- Whether a future study should use one fixed network across RK4 stages (classical fixed-RHS RK4) or preserve this project's current stage-wise stochastic network/RDN resampling. This work chose preservation for reproducibility.
- Whether a future study should refresh `p_conn` on simulation time/steps and reset between runs or preserve the current every-ten-RHS-call cadence with a persistent per-instance counter. This work chose preservation.
- Whether the tiny-distance floor should be recalibrated for future parameter regimes with opinion units or scales substantially different from the current initial standard deviation of 3.0.

## Resolution log: six open questions

### Q6 — Unsupported RDN methods (completed first)

`RDN.calculate()` now raises `NotImplementedError` for every method except the existing methods 0 (no-op) and 1 (sample-mean nudge). This intentionally changes only unsupported selections; methods 0/1 are unchanged. Tests cover methods 0, 1, 2, 3, and an unknown value. The full suite passed (12 tests), and a fixed-seed method-1 run matched the prior history and endpoint exactly.


### Q1 — RK4 network resampling (completed)

The existing stage-wise stochastic behavior is preserved and made explicit in an `op_dynamics()` code comment: each of RK4's four RHS evaluations builds a fresh adjacency matrix (and method 1 continues drawing a fresh RDN sample). This is not classical RK4 with a fixed deterministic RHS; it is the current stochastic model semantics. No numerical result or RNG order changed. A new test confirms four adjacency constructions per RK4 step. The full suite passed (13 tests), and the fixed-seed RK4 baseline history and endpoint matched exactly.


### Q2 — `p_conn` refresh cadence (completed)

The cadence remains every ten `op_dynamics()` calls, and the call counter still persists across runs. The private fields were renamed to `_p_conn_dynamics_call_counter` and `_p_conn_refresh_every_dynamics_calls`, with an inline comment explaining that RK4 stage calls count separately. This is a behavior-preserving terminology clarification; no refresh timing or RNG order changed. A new test verifies Euler's tenth-call refresh, RK4's tenth/twentieth-call refresh, and persistence across consecutive runs. The full suite passed (14 tests), and fixed-seed history, endpoint, and `p_conn` matched exactly.


### Q3 — Probability fallback normalization (completed)

`connect_prob_matrix()` now assigns uniform probabilities to zero/non-finite-sum rows and normalizes every remaining row independently, even when a different row falls back. This changes results only for mixed good/bad rows (or non-finite sums); ordinary fixed-seed `p_conn` and trajectory arrays remained exactly equal to the pre-change baseline. A synthetic overflow case verified a uniform fallback row, unit sums for every row, and a zero diagonal. The full suite passed (15 tests).

All 300 existing HDF5 files were checked for `p_conn`/connection-probability datasets; none stores such a matrix. Therefore historical probability row sums cannot be directly verified from those files. The files were not modified.


### Q4 — Degenerate metric and zero-distance guard (completed)

`initalise_phi()` now rejects non-finite delta and emits a `RuntimeWarning` when the metric condition number exceeds 1e6. `connect_prob_matrix()` clamps distances at `3 * float32 epsilon` (approximately 3.58e-7 opinion units), one float32 precision unit at the initial opinion scale of 3.0. This leaves ordinary distances unchanged but intentionally regularizes exact or extremely close pairs. The source metric is nonsingular at delta=90 degrees (approximately identity); singularity occurs near multiples of 180 degrees, so 90 degrees does not warn. Tests cover these cases, coincident opinions, and non-finite input. At delta=50, fixed-seed history and `p_conn` matched the original baseline exactly; the full suite passed (17 tests).


### Q5 — HDF5 final endpoint semantics (completed last)

All master and standalone pre/post writers now save `sn.get_opinions()` as `final_opinions`, and add `last_history_step` with the prior last history frame. Both pre- and post-run entry points return the true integrated endpoint. `h5_loader` keeps its existing final-opinion API and adds `load_last_history_step()` plus `data_type="last_history_step"` support with a fallback for legacy files. The visualizer retains the stable `final_opinions` key and now describes it as the integrated endpoint for new files. The `opinions` history and its length are unchanged.

This intentionally changes newly written `final_opinions` values by one integration update and makes new master post-RDN runs start from the corrected pre-RDN endpoint. The old value remains available as `last_history_step`; existing sweep files were not rewritten. Fixed-seed history matched exactly; the new endpoint matched the pre-change in-memory endpoint exactly, and `last_history_step` matched the previously serialized value exactly. For the N=160, 10-step comparison, the endpoint correction had maximum absolute magnitude 0.0889897, mean absolute magnitude 0.0211368, and changed 320 of 3200 opinion entries. Writer, legacy-loader, `load_multiple`, and visualizer compatibility tests passed.


## Six-question closeout

- Q1 preserved stage-wise stochastic network/RDN resampling and documented it explicitly.
- Q2 preserved call-based refresh cadence and cross-run counter state under accurate private names.
- Q3 fixed mixed-row fallback normalization; ordinary seeded results stayed exact. Historical probability matrices were not stored, so historic row sums were unobservable.
- Q4 added the metric conditioning warning and tiny-distance clamp. Ordinary delta=50 results stayed exact; degenerate metrics now produce finite normalized probabilities.
- Q5 corrected endpoint values for new files and preserved old snapshots under `last_history_step`; old HDF5 artifacts remain untouched.
- Q6 now rejects unsupported RDN method numbers; supported methods retain their behavior.

Final verification: all 23 tests pass and `compileall` succeeds. No existing HDF5 files were modified. Remaining open scientific choice: whether a researcher eventually wants classical fixed-network RK4 or time-based `p_conn` refresh; both were deliberately preserved here to maintain the existing stochastic model and reproducibility.
