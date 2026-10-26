# medical_imaging
FastAPI backend + simple UI for showcases of medical imaging

## Run

```
python -m medical_imaging.ui
```

That single process serves the NiceGUI front-end on http://localhost:8080 and
the model HTTP API under `/api` (e.g. `POST /api/bedsores`, `POST /api/ocr`).
Models load lazily on first use.

Demo logins: `admin / admin`, `clinician / medical`.

### Optional: separate model server

```
python -m medical_imaging.server              # API only, on :8000
BEDSORES_SERVER_URL=http://localhost:8000 python -m medical_imaging.ui
```
