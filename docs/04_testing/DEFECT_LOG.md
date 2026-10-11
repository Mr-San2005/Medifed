# Defect Log — MediFed

| ID | Date | Phase | Description | Severity | Found by | Fix | Status |
|---|---|---|---|---|---|---|---|
| DEF-001 | 2026-10-10 | 1 | `verify_data.py` and `make_loader` used global `RAW_DIR` to reconstruct absolute paths from manifest entries. Manifest paths are relative to the hospital-specific raw dir, so all 766 files appeared missing. | Medium | Running `verify_data.py` | Changed both to use `HOSPITAL_RAW_DIRS[hospital_id]` instead of `RAW_DIR` | Fixed |
