"""Main NiceGUI front-end: a login gate and a plugin dashboard.

The dashboard lists the available medical-imaging plugins. Each plugin is a
separate NiceGUI page; selecting one navigates to it. Currently the
:mod:`~medical_imaging.ui.bedsores` pressure-injury classifier is wired up as a
subpage at ``/bedsores`` and the :mod:`~medical_imaging.ui.ocr` document OCR
tool at ``/ocr`` — importing those modules registers their routes.

Launch everything with::

    python -m medical_imaging.ui

The model server (:mod:`medical_imaging.server`) is mounted into this app under
``/api`` (e.g. ``POST /api/ocr``), so a single process serves both the UI and
the HTTP API. Models are loaded lazily on first use.

Log in with one of the demo accounts below (username / password)::

    admin / admin
    clinician / medical
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import RedirectResponse
from nicegui import Client, app, ui
from starlette.middleware.base import BaseHTTPMiddleware

# Importing the modules registers their ``@ui.page(...)`` routes with NiceGUI.
from medical_imaging.ui import bedsores  # noqa: F401
from medical_imaging.ui import ocr  # noqa: F401
from medical_imaging.server import app as model_api

# Expose the model server's HTTP API from the same process.
app.mount("/api", model_api)

# ---------------------------------------------------------------------------
# Demo credentials — replace with a real user store for anything beyond a demo.
# ---------------------------------------------------------------------------
CREDENTIALS: dict[str, str] = {
    "admin": "admin",
    "clinician": "medical",
}

# Routes reachable without being logged in.
UNRESTRICTED_ROUTES: set[str] = {"/login"}


@dataclass(frozen=True)
class Plugin:
    """A single tool exposed on the dashboard."""

    name: str
    route: str
    icon: str
    color: str
    description: str
    available: bool = True


# The plugin registry. Add an entry here (and register its ``@ui.page``) to
# surface a new tool on the dashboard.
PLUGINS: list[Plugin] = [
    Plugin(
        name="Pressure Injury Classifier",
        route="/bedsores",
        icon="healing",
        color="#0891b2",
        description="Stage bedsores from a wound image with a confidence breakdown.",
    ),
    Plugin(
        name="Document OCR",
        route="/ocr",
        icon="document_scanner",
        color="#7c3aed",
        description="Extract text from scanned images or PDF documents.",
    ),
    Plugin(
        name="Wound Segmentation",
        route="/segmentation",
        icon="gesture",
        color="#7c3aed",
        description="Pixel-level wound delineation and area measurement.",
        available=False,
    ),
    Plugin(
        name="Chest X-Ray Triage",
        route="/xray",
        icon="pulmonology",
        color="#0e7490",
        description="Flag common thoracic findings on chest radiographs.",
        available=False,
    ),
    Plugin(
        name="Skin Lesion Screening",
        route="/dermatoscopy",
        icon="coronavirus",
        color="#ea580c",
        description="Melanoma risk estimation from dermatoscopic images.",
        available=False,
    ),
]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect unauthenticated users to the login page.

    Restricts every registered NiceGUI page except those in
    :data:`UNRESTRICTED_ROUTES`.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") or path == "/api":
            return await call_next(request)  # plain HTTP API, no login session
        if not app.storage.user.get("authenticated", False):
            if path in Client.page_routes.values() and path not in UNRESTRICTED_ROUTES:
                app.storage.user["referrer_path"] = path
                return RedirectResponse("/login")
        return await call_next(request)


app.add_middleware(AuthMiddleware)


def _page_head() -> None:
    """Shared page chrome: background + fade-in animation."""
    ui.query("body").style("background: #0f172a;")
    ui.add_head_html(
        "<style>"
        ".fade-in{animation:fadeIn .5s ease}"
        "@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}"
        ".plugin-card{transition:transform .15s ease,box-shadow .15s ease,border-color .15s ease}"
        ".plugin-card:hover{transform:translateY(-3px);box-shadow:0 10px 30px rgba(0,0,0,.45)}"
        "</style>"
    )


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------
@ui.page("/login")
def login_page() -> None:
    _page_head()

    def try_login() -> None:
        if CREDENTIALS.get(username.value) == password.value:
            app.storage.user.update(
                {"username": username.value, "authenticated": True}
            )
            ui.navigate.to(app.storage.user.get("referrer_path", "/"))
        else:
            ui.notify("Invalid username or password", color="negative")

    with ui.column().classes("w-full items-center justify-center gap-6").style(
        "min-height:100vh"
    ):
        with ui.column().classes("items-center gap-1 fade-in"):
            ui.icon("medical_services", size="3rem").style("color:#22d3ee")
            ui.label("Medical Imaging Suite").classes(
                "text-white text-2xl font-bold"
            )
            ui.label("Sign in to access the diagnostic plugins").classes(
                "text-slate-400 text-sm"
            )

        with ui.card().classes("rounded-2xl p-6 gap-4 w-80 fade-in").style(
            "background:#1e293b;border:1px solid #334155"
        ):
            username = (
                ui.input("Username")
                .props("outlined dark color=cyan-6")
                .classes("w-full")
            )
            password = (
                ui.input("Password", password=True, password_toggle_button=True)
                .props("outlined dark color=cyan-6")
                .classes("w-full")
                .on("keydown.enter", try_login)
            )
            ui.button("Sign in", icon="login", on_click=try_login).props(
                "color=cyan-7 unelevated"
            ).classes("w-full")

        ui.label("Demo: admin / admin · clinician / medical").classes(
            "text-slate-600 text-xs italic"
        )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@ui.page("/")
def dashboard_page() -> None:
    _page_head()

    def logout() -> None:
        app.storage.user.clear()
        ui.navigate.to("/login")

    # ---- Header ---------------------------------------------------------
    with ui.row().classes("w-full items-center gap-3 py-6 px-6").style(
        "background:linear-gradient(120deg,#0e7490,#0891b2);"
        "box-shadow:0 4px 24px rgba(0,0,0,.35)"
    ):
        ui.icon("medical_services", size="2.4rem").classes("text-white")
        with ui.column().classes("gap-0"):
            ui.label("Medical Imaging Suite").classes(
                "text-white text-2xl font-bold leading-tight"
            )
            ui.label("Select a plugin to begin").classes("text-cyan-100 text-sm")
        ui.space()
        with ui.row().classes("items-center gap-2"):
            ui.icon("account_circle", size="1.6rem").classes("text-white")
            ui.label(
                f"{app.storage.user.get('username', 'user')}"
            ).classes("text-white text-sm font-medium")
            ui.button(icon="logout", on_click=logout).props(
                "flat round color=white"
            ).tooltip("Sign out")

    # ---- Plugin grid ----------------------------------------------------
    with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-4"):
        ui.label("Available plugins").classes(
            "text-slate-200 text-lg font-semibold"
        )
        with ui.row().classes("w-full gap-6 items-stretch"):
            for plugin in PLUGINS:
                _render_plugin_card(plugin)

        ui.label(
            "For research/education only — not a medical device. "
            "Always confirm with a clinician."
        ).classes("text-slate-500 text-xs italic mt-2")


def _render_plugin_card(plugin: Plugin) -> None:
    card = (
        ui.card()
        .classes("plugin-card rounded-2xl p-5 gap-3 fade-in")
        .style(
            "background:#1e293b;border:1px solid #334155;"
            "width:320px;"
            + ("cursor:pointer" if plugin.available else "opacity:.6")
        )
    )
    with card:
        with ui.row().classes("items-center gap-3 no-wrap w-full"):
            with ui.element("div").classes(
                "flex items-center justify-center rounded-xl"
            ).style(
                f"width:52px;height:52px;background:{plugin.color}26;flex:none"
            ):
                ui.icon(plugin.icon, size="1.8rem").style(f"color:{plugin.color}")
            with ui.column().classes("gap-0 flex-1"):
                ui.label(plugin.name).classes(
                    "text-white text-base font-semibold leading-tight"
                )
                badge_text = "Ready" if plugin.available else "Coming soon"
                badge_color = "#22c55e" if plugin.available else "#64748b"
                ui.label(badge_text).classes("text-xs font-medium").style(
                    f"color:{badge_color}"
                )

        ui.label(plugin.description).classes("text-slate-400 text-sm")

        if plugin.available:
            ui.button(
                "Open",
                icon="arrow_forward",
                on_click=lambda p=plugin: ui.navigate.to(p.route),
            ).props("flat color=cyan-6").classes("self-end")
            card.on("click", lambda p=plugin: ui.navigate.to(p.route))
        else:
            ui.button("Open", icon="lock").props(
                "flat color=grey-7 disable"
            ).classes("self-end")


ui.run(
    title="Medical Imaging Suite",
    port=8013,
    host="localhost",
    reload=False,
    favicon="🩺",
    storage_secret="medical-imaging-suite-secret-change-me",
)
