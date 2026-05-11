# Agent Bootstrap

Use this file to start any new coding-agent chat for this repository. It is a bootstrap guide, not the full project map. The authoritative protocol is `markdowns/main.md`.

## New Chat Prompt

Paste this into a new chat when switching sessions:

```text
You are working in the LV-network repository. Before doing anything else, read AGENTS.md, then re-open markdowns/main.md and follow its prompt-handling protocol. After that, read markdowns/chat_handoff.md for current workflow state. Start by running a lightweight git status check. Do not inspect raw data/output files unless explicitly needed; map structure via filenames, notebook headings, and source function summaries. If code/notebook structure changes, update markdowns/main.md Section 6 and the affected structure/output sections in the same session. If model assumptions change, update markdowns/key assumptions.md too.
```

## Required Start Sequence For The Agent

1. Read this file.
2. Re-open and read `markdowns/main.md`.
3. Read `markdowns/chat_handoff.md` if present.
4. Run `git status --short`.
5. Sync with current code/notebook structure before proposing or making edits.
6. Report any relevant staged/unstaged work before touching files that may overlap.

## Project Entry Points

- Human overview: `README.md`
- Project map and operating protocol: `markdowns/main.md`
- Model and experiment assumptions: `markdowns/key assumptions.md`
- Volatile handoff state: `markdowns/chat_handoff.md`
- Active run-generation notebook: `Codes/simulation.ipynb`
- Active analysis notebook: `Codes/analysis.ipynb`
- Reusable modules: `Codes/sourcecode/`

## Hard Rules

- `markdowns/main.md` must be re-read at the start of every new user task before analysis, edits, or command execution.
- Prioritize code and notebook structure over raw data inspection.
- Do not inspect large data/output files unless explicitly requested or required.
- Do not run experiment or data-update tasks directly; provide runnable code for the user to execute.
- When notebook work is requested, implement runnable notebook code cells rather than prose-only instructions.
- Ask before changing the local runtime environment, including package, interpreter, kernel, or toolchain changes.
- Never revert existing user changes unless the user explicitly asks.

## Documentation Maintenance

- Every structural code/notebook change must update `markdowns/main.md` in the same session.
- Update both the relevant structure section and Section 6 change log in `markdowns/main.md`.
- When model assumptions change, update `markdowns/key assumptions.md` in the same session.
- When active workflow state or pending work changes materially, update `markdowns/chat_handoff.md`.
- Keep this file concise and focused on starting future chats.

## Data Safety

Large data and generated-output folders exist in this repo, including `Input data/`, `Output Data/`, `Codes/Output/`, and similar support folders. Prefer directory listings, file names, notebook headings, and Python function summaries. Open raw CSV/XLSX/output files only when the user asks for that data or when the task cannot be completed without it.
