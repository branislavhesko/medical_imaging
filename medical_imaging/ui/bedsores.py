"""A rich NiceGUI front-end for the bedsores (pressure ulcer) classifier.

Launch with::

    python -m medical_imaging.ui

The model runs in-process (see :mod:`medical_imaging.server`); set
``BEDSORES_SERVER_URL`` to use a separately running model server instead.
The page uploads an image, classifies it and renders the prediction with a
confidence gauge and a per-class probability breakdown.
"""

from __future__ import annotations

import base64
import os

import requests
from nicegui import events, run, ui

from medical_imaging.server import run_bedsores

# Optional: point at a remote model server instead of running in-process.
SERVER_URL = os.environ.get("BEDSORES_SERVER_URL")

# Clinical metadata per class: colour, icon and a short description.
# Ordered from least to most severe so we can render a consistent legend.
CLASS_INFO: dict[str, dict[str, str]] = {
    "Invalid": {
        "color": "#64748b",
        "icon": "help",
        "label": "Invalid / Unclassifiable",
        "desc": "Image could not be assigned to a pressure-injury stage.",
    },
    "SDTI": {
        "color": "#7c3aed",
        "icon": "blur_on",
        "label": "Suspected Deep Tissue Injury",
        "desc": "Localised discolouration of intact skin from pressure/shear damage.",
    },
    "Stage_I": {
        "color": "#eab308",
        "icon": "looks_one",
        "label": "Stage I",
        "desc": "Intact skin with non-blanchable redness of a localised area.",
    },
    "Stage_II": {
        "color": "#f97316",
        "icon": "looks_two",
        "label": "Stage II",
        "desc": "Partial-thickness skin loss with exposed dermis.",
    },
    "Stage_III": {
        "color": "#ef4444",
        "icon": "looks_3",
        "label": "Stage III",
        "desc": "Full-thickness skin loss; subcutaneous fat may be visible.",
    },
    "Stage_IV": {
        "color": "#b91c1c",
        "icon": "looks_4",
        "label": "Stage IV",
        "desc": "Full-thickness skin and tissue loss with exposed muscle/bone.",
    },
}

DEFAULT_INFO = {"color": "#64748b", "icon": "help", "label": "Unknown", "desc": ""}


def info_for(class_name: str) -> dict[str, str]:
    return CLASS_INFO.get(class_name, {**DEFAULT_INFO, "label": class_name})


def call_server(image_bytes: bytes, filename: str) -> dict:
    """Blocking classification call (run off the event loop)."""
    if not SERVER_URL:
        return run_bedsores(image_bytes)
    response = requests.post(
        f"{SERVER_URL}/bedsores",
        files={"image": (filename, image_bytes, "application/octet-stream")},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


@ui.page("/bedsores")
def bedsores_page() -> None:
    ui.query("body").style("background: #0f172a;")
    ui.add_head_html(
        "<style>"
        ".fade-in{animation:fadeIn .5s ease}"
        "@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}"
        "</style>"
    )

    state: dict[str, object] = {"image_data_url": None}

    # ---- Header ---------------------------------------------------------
    with ui.row().classes(
        "w-full items-center gap-3 py-6 px-6"
    ).style(
        "background:linear-gradient(120deg,#0e7490,#0891b2);"
        "box-shadow:0 4px 24px rgba(0,0,0,.35)"
    ):
        ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props(
            "flat round color=white"
        ).tooltip("Back to plugins")
        ui.icon("healing", size="2.4rem").classes("text-white")
        with ui.column().classes("gap-0"):
            ui.label("Pressure Injury Classifier").classes(
                "text-white text-2xl font-bold leading-tight"
            )
            ui.label("AI-assisted bedsore staging · upload an image to analyse").classes(
                "text-cyan-100 text-sm"
            )

    with ui.row().classes("w-full max-w-6xl mx-auto p-6 gap-6 items-stretch no-wrap"):
        # ---- Left: upload + preview ------------------------------------
        with ui.card().classes(
            "flex-1 rounded-2xl p-5 gap-4"
        ).style("background:#1e293b;border:1px solid #334155"):
            ui.label("Input Image").classes("text-slate-200 text-lg font-semibold")

            preview = ui.image().classes(
                "w-full rounded-xl object-contain bg-slate-900"
            ).style("min-height:320px;max-height:420px").props("no-spinner")
            preview.set_visibility(False)

            placeholder = ui.column().classes(
                "w-full items-center justify-center gap-2 rounded-xl"
            ).style(
                "min-height:320px;border:2px dashed #475569;background:#0f172a"
            )
            with placeholder:
                ui.icon("add_photo_alternate", size="3rem").classes("text-slate-500")
                ui.label("Preview will appear here").classes("text-slate-500")

            upload = ui.upload(
                label="Drop or select a wound image",
                auto_upload=True,
                max_files=1,
            ).classes("w-full").props('accept="image/*" color=cyan-7 flat bordered')

        # ---- Right: results --------------------------------------------
        with ui.card().classes(
            "flex-1 rounded-2xl p-5 gap-4"
        ).style("background:#1e293b;border:1px solid #334155"):
            ui.label("Prediction").classes("text-slate-200 text-lg font-semibold")
            results_container = ui.column().classes("w-full gap-4")
            with results_container:
                _render_empty_state()

    # ---- Handlers -------------------------------------------------------
    async def handle_upload(e: events.UploadEventArguments) -> None:
        image_bytes = await e.file.read()
        filename = e.file.name
        mime = e.file.content_type or "image/png"
        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
        state["image_data_url"] = data_url

        placeholder.set_visibility(False)
        preview.set_source(data_url)
        preview.set_visibility(True)

        results_container.clear()
        with results_container:
            with ui.column().classes("w-full items-center justify-center gap-3 py-16"):
                ui.spinner("dots", size="3rem", color="cyan")
                ui.label("Analysing image…").classes("text-slate-400")

        try:
            result = await run.io_bound(call_server, image_bytes, filename)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            results_container.clear()
            with results_container:
                _render_error(str(exc))
            return

        results_container.clear()
        with results_container:
            _render_prediction(result)

    upload.on_upload(handle_upload)


def _render_empty_state() -> None:
    with ui.column().classes("w-full items-center justify-center gap-3 py-16"):
        ui.icon("insights", size="3rem").classes("text-slate-600")
        ui.label("Upload an image to see the classification").classes("text-slate-500")


def _render_error(message: str) -> None:
    with ui.card().classes("w-full rounded-xl p-4 fade-in").style(
        "background:#3f1d1d;border:1px solid #7f1d1d"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("error", size="1.6rem").classes("text-red-400")
            ui.label("Prediction failed").classes("text-red-200 font-semibold")
        ui.label(message).classes("text-red-300 text-sm break-all")
        if SERVER_URL:
            ui.label(
                f"Is the model server running at {SERVER_URL}?"
            ).classes("text-red-400/70 text-xs")


def _render_prediction(result: dict) -> None:
    predicted = result.get("class", "Unknown")
    confidence = float(result.get("confidence", 0.0))
    probabilities: dict[str, float] = result.get("probabilities", {})
    info = info_for(predicted)
    color = info["color"]

    # ---- Headline verdict ----------------------------------------------
    with ui.card().classes("w-full rounded-xl p-4 fade-in").style(
        f"background:{color}1a;border:1px solid {color}"
    ):
        with ui.row().classes("items-center gap-4 no-wrap w-full"):
            ui.icon(info["icon"], size="2.6rem").style(f"color:{color}")
            with ui.column().classes("gap-0 flex-1"):
                ui.label(info["label"]).classes("text-white text-xl font-bold")
                if info["desc"]:
                    ui.label(info["desc"]).classes("text-slate-300 text-sm")
            with ui.column().classes("items-center gap-0"):
                ui.label(f"{confidence * 100:.1f}%").classes(
                    "text-2xl font-bold"
                ).style(f"color:{color}")
                ui.label("confidence").classes("text-slate-400 text-xs uppercase")

    # ---- Confidence gauge ----------------------------------------------
    with ui.column().classes("w-full gap-1 fade-in"):
        ui.label("Model confidence").classes("text-slate-400 text-xs uppercase")
        ui.linear_progress(value=confidence, show_value=False, size="14px").props(
            f"rounded color=cyan-6 track-color=slate-7"
        ).style(f"color:{color}")

    # ---- Per-class probability breakdown -------------------------------
    ui.label("Class probabilities").classes(
        "text-slate-300 text-sm font-semibold mt-2"
    )

    ordered = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    with ui.column().classes("w-full gap-3 fade-in"):
        for class_name, prob in ordered:
            cinfo = info_for(class_name)
            is_top = class_name == predicted
            with ui.column().classes("w-full gap-1"):
                with ui.row().classes("w-full items-center justify-between no-wrap"):
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon(cinfo["icon"], size="1.1rem").style(
                            f"color:{cinfo['color']}"
                        )
                        ui.label(cinfo["label"]).classes(
                            "text-sm " + ("text-white font-semibold" if is_top
                                          else "text-slate-400")
                        )
                    ui.label(f"{prob * 100:.1f}%").classes(
                        "text-sm font-mono " + ("text-white" if is_top
                                                else "text-slate-500")
                    )
                # custom bar so we can colour per-class
                with ui.element("div").classes("w-full rounded-full").style(
                    "height:8px;background:#0f172a;overflow:hidden"
                ):
                    ui.element("div").classes("h-full rounded-full").style(
                        f"width:{max(prob * 100, 1):.1f}%;background:{cinfo['color']};"
                        "transition:width .6s ease"
                    )

    ui.label(
        "For research/education only — not a medical device. Always confirm with a clinician."
    ).classes("text-slate-500 text-xs italic mt-2")


if __name__ == "__main__":
    ui.run(title="Pressure Injury Classifier", port=8080, reload=False, favicon="🩺")
