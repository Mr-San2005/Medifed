# Defect Log — MediFed

| ID | Date | Phase | Description | Severity | Found by | Fix | Status |
|---|---|---|---|---|---|---|---|
| DEF-001 | 2026-10-10 | 1 | `verify_data.py` and `make_loader` used global `RAW_DIR` to reconstruct absolute paths from manifest entries. Manifest paths are relative to the hospital-specific raw dir, so all 766 files appeared missing. | Medium | Running `verify_data.py` | Changed both to use `HOSPITAL_RAW_DIRS[hospital_id]` instead of `RAW_DIR` | Fixed |
| DEF-002 | 2026-10-13 | 3 Part 1 | Flower copied the app under the user profile, where local manifests were unavailable when `MEDIFED_ROOT` was unset. | High | Flower deployment startup | Set `MEDIFED_ROOT` in the launcher and added clear client guidance to start with `scripts/start_federation.ps1` when fallback manifests are missing. | Fixed |
| DEF-003 | 2026-10-13 | 3 Part 1 | A stale SuperLink could keep port 9093 occupied and prevent a new federation runtime from starting. | Medium | Flower deployment startup | Added `scripts/stop_federation.ps1` and a 9092-9096 availability check in `scripts/start_federation.ps1`. | Fixed |
| DEF-004 | 2026-10-13 | 3 Part 1 | Flower FAB format version 1 requires a LICENSE file included in the app package. | Medium | Flower app configuration validation | Added the project `LICENSE` file and declared it in `[project].license`. | Fixed |
