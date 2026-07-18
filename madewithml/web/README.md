# Made With ML Web App

A responsive browser interface for the Made With ML project classifier. It is
served by the same FastAPI and Ray Serve application as the prediction API, so
the page and model always use the same host.

## What the interface does

- Accepts a project title and description.
- Sends the input to `POST /predict/`.
- Shows the predicted category, confidence distribution, request ID, and model
  latency.
- Checks `GET /health/ready` to show whether the model is ready.
- Includes example NLP, computer-vision, and MLOps project inputs.

## Visual design

The styling is written in CSS inside `index.html` so the page can be served as
one fast, self-contained asset.

| Element | Color | Purpose |
|---|---|---|
| Main background | `#F5F7FB` | Calm, clean workspace |
| Brand violet | `#6547E8` | Primary actions and highlights |
| Deep violet | `#4D32C8` | Hover state for the primary action |
| Mint | `#D8F7E9` | Positive model-result accent |
| Ink | `#172033` | Main text and headings |
| Muted gray | `#667085` | Supporting text and metadata |

The layout is responsive: it uses two columns on larger screens and changes to
a single column on phones.

## Run the app

Start the model service from the repository root:

```bash
source venv/bin/activate
export PYTHONPATH="$PWD"
export GITHUB_USERNAME="local"
python madewithml/serve.py --run_id "YOUR_RUN_ID"
```

Open the app locally:

```text
http://127.0.0.1:8000/app
```

If using a Cloudflare tunnel, append `/app` to the tunnel URL:

```text
https://YOUR-TUNNEL.trycloudflare.com/app
```

## Files

- `index.html` — page structure, responsive CSS, and browser-side API calls.
- `../serve.py` — provides the `/app` route alongside the API endpoints.
