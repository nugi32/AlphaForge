# Prompt to paste at the start of the next session

"I am continuing the project in this zip. Read CHECKPOINT.md, then docs/NOTES.md and docs/ROADMAP.md.
Run `pip install -e .[dev]` and `pytest -q` first. Then do the next item of CHECKPOINT.md (section "Next step": if I give you MetaEditor compiler messages or tester files, fix the generator / run `scripts/check_parity.py` first; otherwise M11, M12 or M10b), add tests, run mutation checks on the new code, update CHECKPOINT.md and stop. Send a zip after every checkpoint (build it at the end of the session, do not wait for me to ask).
Write everything in English. Do not edit the verbatim-copied modules listed in docs/NOTES.md D-002."
