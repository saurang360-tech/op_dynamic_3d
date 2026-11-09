# Editable project diagrams

Open `op_dynamic_3d.drawio` in https://app.diagrams.net using **Device → Open Existing Diagram**, or **File → Open From → Device**. The file contains native editable shapes and connectors, not screenshots.

Pages:
1. **Workflow** — master sweep, pre/post phases, stored data, and analysis.
2. **File map** — entry points, solver, RDN, output files, utilities, and tests.
3. **Simulation step** — integrators and the operations inside each derivative evaluation.

Based on source commit `e08b511`, reviewed 2026-11-09. These diagrams describe the current two-topic implementation despite the repository's 3D name.

## Propose changes

- Duplicate a purple proposal box, write the change and its reason, and connect it to the relevant step with a dashed arrow.
- Keep blue current-behavior boxes intact until an implementation decision is made.
- Use page tabs to move between diagrams. Double-click labels to edit; drag boxes and connector endpoints to restructure.
- Save back to this `.drawio` file. Send the edited file here when ready to discuss or implement changes.

## Source details to review

- `config.json` includes RDN method 2; `RDN.calculate()` supports only methods 0 and 1. Master catches and logs per-repetition failures.
- `connect_prob_matrix()` reads `self._opinions`, not the stage state passed into `op_dynamics()`. Both integrators evolve a local copy and assign `self._opinions` at the end, so refreshes during a run use the stored initial state.
- Master uses RK4 for both phases; the standalone pre runner uses Euler.
- HDF5 `final_opinions` is the integrated endpoint; `last_history_step` is the final pre-step snapshot.

Solid arrows show current flow. Purple dashed arrows/callouts provide annotations or proposed edits, not implemented behavior. No simulation source was changed.
