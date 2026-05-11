# LV Network

This repository contains low-voltage network, heating, EV, and demand-profile simulation work. The active workflow is notebook-driven, with reusable Python modules under `Codes/sourcecode`.

## Start Here

- Project map and operating rules: `markdowns/main.md`
- Model and experiment assumptions: `markdowns/key assumptions.md`
- Cross-chat handoff notes: `markdowns/chat_handoff.md`
- Agent-specific instructions: `AGENTS.md`

## Active Workflow

- `Codes/simulation.ipynb`: run generation and manifest export.
- `Codes/analysis.ipynb`: post-processing, cache-based experiment generation, plots, and summaries.
- `Codes/Diagnose_HHP_Infeasibility.ipynb`: infeasible-run replay and diagnosis.
- `Codes/sourcecode/`: reusable optimization, stochastic simulation, aggregation, network, and plotting modules.

## Data Handling

Large input and output folders are part of the repository. Inspect their structure by filenames and folders first; avoid opening raw data/output files unless a task explicitly requires it.
