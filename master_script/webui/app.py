# master_script/webui/app.py
"""FastAPI server for the CANARY Monitor dashboard.

One process serves the single-page app (webui/static) and the JSON API. The
client routes between Live / Results / Detail / Access / Launch entirely in the
browser, so every server route below is either a static asset or an API read.
"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import api, localguard
from .state import COLLECTION

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX = STATIC_DIR / "index.html"

# Client-side routes: each serves the same SPA shell, so a deep link (or a
# reload on /results/<run_id>) lands on the right view instead of a 404.
ROUTES = ["/", "/results", "/results/{run_id}", "/access", "/launch", "/settings"]

STATE = api.STATE

app = FastAPI(title="CANARY Monitor")
app.include_router(api.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _shell() -> FileResponse:
    return FileResponse(INDEX, headers={"Cache-Control": "no-store"})


@app.get("/")
def index():
    return _shell()


@app.get("/results")
def results_page():
    return _shell()


@app.get("/results/{run_id}")
def detail_page(run_id: str):
    return _shell()


@app.get("/access")
def access_page():
    return _shell()


@app.get("/launch")
def launch_page():
    return _shell()


# The one route that is not served to everyone. A remote viewer gets a plain
# page rather than the shell: the SPA would only load and then 403 on its first
# read, which reads as a bug instead of a boundary.
_DENIED_PAGE = f"""<title>CANARY Monitor — settings unavailable</title>
<div style="font-family:system-ui,sans-serif;background:#0d1014;color:#9aa6b2;min-height:100vh;
            display:flex;align-items:center;justify-content:center;padding:24px;margin:-8px">
  <div style="max-width:460px;text-align:center;line-height:1.65">
    <div style="font-size:15px;font-weight:600;color:#e6ebf0;margin-bottom:10px">Settings are local-only</div>
    <div style="font-size:13px">{localguard.DENIED}</div>
    <a href="/" style="display:inline-block;margin-top:22px;font-size:13px;color:#36c08f;text-decoration:none">← Back to the monitor</a>
  </div>
</div>"""


@app.get("/settings")
def settings_page(request: Request):
    if not localguard.is_local(request):
        return HTMLResponse(_DENIED_PAGE, status_code=403)
    return _shell()


def main(host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn

    try:
        STATE.attach_listener(COLLECTION, lambda: None)
    except Exception as exc:  # degrade gracefully: no credentials, no listener
        print(f"Firestore listener unavailable ({exc}); falling back to polling reads.")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
