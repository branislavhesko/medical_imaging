"""NiceGUI front-end for the document OCR service.

Launch with::

    python -m medical_imaging.ui

The model runs in-process (see :mod:`medical_imaging.server`); set
``BEDSORES_SERVER_URL`` to use a separately running model server instead.
The page accepts an image or a PDF and shows the recognised text per page,
rendered as markdown, with copy/download actions.
"""

from __future__ import annotations

import base64
import os

import requests
from nicegui import events, run, ui

from medical_imaging.page_spec import parse_page_spec
from medical_imaging.server import run_ocr

# Optional: point at a remote model server instead of running in-process.
SERVER_URL = os.environ.get("BEDSORES_SERVER_URL")

ACCEPTED_TYPES = "image/*,application/pdf,.pdf"


def call_server(file_bytes: bytes, filename: str, mime: str, pages: str) -> dict:
    """Blocking OCR call (run off the event loop)."""
    if not SERVER_URL:
        return run_ocr(file_bytes, pages)
    response = requests.post(
        f"{SERVER_URL}/ocr",
        files={"file": (filename, file_bytes, mime)},
        data={"pages": pages},
        timeout=600,
    )
    if response.status_code == 422:
        # Validation error (e.g. pages out of range) — show the server's reason.
        detail = response.json().get("detail", response.text)
        raise ValueError(detail if isinstance(detail, str) else str(detail))
    response.raise_for_status()
    return response.json()


def _is_pdf(file_bytes: bytes, filename: str, mime: str) -> bool:
    return (
        file_bytes[:5] == b"%PDF-"
        or mime == "application/pdf"
        or filename.lower().endswith(".pdf")
    )


@ui.page("/ocr")
def ocr_page() -> None:
    ui.query("body").style("background: #0f172a;")
    ui.add_head_html(
        "<style>"
        ".fade-in{animation:fadeIn .5s ease}"
        "@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}"
        ".ocr-markdown{color:#e2e8f0;font-size:1.15rem;line-height:1.75}"
        ".ocr-markdown h1,.ocr-markdown h2,.ocr-markdown h3{color:#fff;font-weight:600;margin:.8em 0 .4em}"
        ".ocr-markdown h1{font-size:2rem}.ocr-markdown h2{font-size:1.6rem}.ocr-markdown h3{font-size:1.3rem}"
        ".ocr-markdown p{margin:.5em 0}"
        ".ocr-markdown table{border-collapse:collapse;margin:.8em 0;width:100%}"
        ".ocr-markdown th,.ocr-markdown td{border:1px solid #334155;padding:.35em .6em;text-align:left}"
        ".ocr-markdown th{background:#1e293b;color:#fff}"
        ".ocr-markdown code,.ocr-markdown pre{background:#1e293b;border-radius:6px;padding:.1em .3em}"
        ".ocr-markdown pre{padding:.6em;overflow:auto}"
        ".ocr-markdown ul,.ocr-markdown ol{padding-left:1.5em;margin:.5em 0}"
        ".ocr-markdown a{color:#c4b5fd}"
        "</style>"
    )

    state: dict[str, object] = {
        "text": "",
        "filename": None,
        "file_bytes": None,
        "mime": None,
        "is_pdf": False,
    }

    # ---- Header ---------------------------------------------------------
    with ui.row().classes(
        "w-full items-center gap-3 py-6 px-6"
    ).style(
        "background:linear-gradient(120deg,#4c1d95,#7c3aed);"
        "box-shadow:0 4px 24px rgba(0,0,0,.35)"
    ):
        ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props(
            "flat round color=white"
        ).tooltip("Back to plugins")
        ui.icon("document_scanner", size="2.4rem").classes("text-white")
        with ui.column().classes("gap-0"):
            ui.label("Document OCR").classes(
                "text-white text-2xl font-bold leading-tight"
            )
            ui.label("Extract text from scanned images or PDF documents").classes(
                "text-violet-100 text-sm"
            )

    with ui.row().classes("w-full max-w-[1600px] mx-auto p-6 gap-6 items-start no-wrap"):
        # ---- Left: upload + preview ------------------------------------
        with ui.card().classes(
            "rounded-2xl p-5 gap-4"
        ).style("background:#1e293b;border:1px solid #334155;flex:1 1 0;min-width:0"):
            ui.label("Input Document").classes("text-slate-200 text-lg font-semibold")

            preview_container = ui.column().classes("w-full gap-2")
            with preview_container:
                _render_placeholder()

            def validate_pages(value: str) -> str | None:
                try:
                    parse_page_spec(value)
                except ValueError as exc:
                    return str(exc)
                return None

            pages_input = ui.input(
                "Pages (PDF only)",
                placeholder="all — or e.g. 3, 5-8, 12",
                validation=validate_pages,
            ).props("outlined dense dark color=purple-6 clearable").classes("w-full")
            with pages_input.add_slot("hint"):
                ui.label(
                    "Leave empty for all pages. Comma-separated numbers and ranges."
                )

            upload = ui.upload(
                label="Drop or select an image or PDF",
                auto_upload=True,
                max_files=1,
            ).classes("w-full").props(
                f'accept="{ACCEPTED_TYPES}" color=purple-7 flat bordered'
            )

            start_btn = ui.button(
                "Start conversion", icon="play_arrow"
            ).props("color=purple-7 unelevated no-caps").classes("w-full")
            start_btn.disable()

        # ---- Right: results --------------------------------------------
        with ui.card().classes(
            "rounded-2xl p-5 gap-4"
        ).style("background:#1e293b;border:1px solid #334155;flex:2 1 0;min-width:0"):
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                ui.label("Recognised Text").classes(
                    "text-slate-200 text-lg font-semibold"
                )
                with ui.row().classes("gap-1"):
                    copy_btn = ui.button(
                        icon="content_copy",
                        on_click=lambda: _copy_text(state),
                    ).props("flat round color=purple-4").tooltip("Copy to clipboard")
                    download_btn = ui.button(
                        icon="download",
                        on_click=lambda: _download_text(state),
                    ).props("flat round color=purple-4").tooltip("Download as .md")
            copy_btn.set_visibility(False)
            download_btn.set_visibility(False)
            results_container = ui.column().classes("w-full gap-4")
            with results_container:
                _render_empty_state()

    # ---- Handlers -------------------------------------------------------
    async def handle_upload(e: events.UploadEventArguments) -> None:
        """Stage the file and show a preview; OCR starts on the button."""
        file_bytes = await e.file.read()
        filename = e.file.name
        mime = e.file.content_type or "application/octet-stream"
        is_pdf = _is_pdf(file_bytes, filename, mime)
        if is_pdf:
            mime = "application/pdf"
        elif not mime.startswith("image/"):
            mime = "image/png"

        state.update(
            {
                "filename": filename,
                "file_bytes": file_bytes,
                "mime": mime,
                "is_pdf": is_pdf,
                "text": "",
            }
        )
        copy_btn.set_visibility(False)
        download_btn.set_visibility(False)

        preview_container.clear()
        with preview_container:
            _render_preview(file_bytes, filename, mime, is_pdf)

        results_container.clear()
        with results_container:
            _render_ready_state(is_pdf)

        start_btn.enable()

    async def start_conversion() -> None:
        file_bytes = state["file_bytes"]
        if file_bytes is None:
            ui.notify("Upload a document first", color="warning")
            return

        pages_spec = (pages_input.value or "").strip() if state["is_pdf"] else ""
        try:
            parse_page_spec(pages_spec)
        except ValueError as exc:
            ui.notify(f"Fix the page selection first: {exc}", color="negative")
            return

        filename = str(state["filename"])
        mime = str(state["mime"])
        is_pdf = bool(state["is_pdf"])

        state["text"] = ""
        copy_btn.set_visibility(False)
        download_btn.set_visibility(False)
        start_btn.disable()
        start_btn.props("loading")

        results_container.clear()
        with results_container:
            with ui.column().classes("w-full items-center justify-center gap-3 py-16"):
                ui.spinner("dots", size="3rem", color="purple")
                ui.label(
                    "Running OCR… PDFs are processed page by page, this can take a while."
                    if is_pdf else "Running OCR…"
                ).classes("text-slate-400 text-center")

        try:
            result = await run.io_bound(call_server, file_bytes, filename, mime, pages_spec)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            results_container.clear()
            with results_container:
                _render_error(str(exc))
            return
        finally:
            start_btn.props(remove="loading")
            start_btn.enable()

        pages: list[str] = result.get("pages") or [result.get("text", "")]
        page_numbers: list[int] = result.get("page_numbers") or list(range(1, len(pages) + 1))
        state["text"] = "\n\n".join(pages)
        copy_btn.set_visibility(True)
        download_btn.set_visibility(True)
        results_container.clear()
        with results_container:
            _render_result(pages, page_numbers)

    upload.on_upload(handle_upload)
    start_btn.on_click(start_conversion)
    upload.on(
        "failed",
        lambda: ui.notify(
            "Upload failed — the file may exceed the server's size limit.",
            color="negative",
            timeout=8000,
        ),
        args=[],
    )


def _render_ready_state(is_pdf: bool) -> None:
    with ui.column().classes("w-full items-center justify-center gap-3 py-16"):
        ui.icon("play_circle", size="3rem").classes("text-purple-400")
        ui.label("Document loaded").classes("text-slate-300 font-semibold")
        ui.label(
            "Choose the pages to process, then press “Start conversion”."
            if is_pdf else "Press “Start conversion” to run OCR."
        ).classes("text-slate-500 text-center")


def _render_placeholder() -> None:
    with ui.column().classes(
        "w-full items-center justify-center gap-2 rounded-xl"
    ).style("min-height:320px;border:2px dashed #475569;background:#0f172a"):
        ui.icon("upload_file", size="3rem").classes("text-slate-500")
        ui.label("Preview will appear here").classes("text-slate-500")
        ui.label("PNG · JPEG · TIFF · PDF").classes("text-slate-600 text-xs")


def _render_preview(file_bytes: bytes, filename: str, mime: str, is_pdf: bool) -> None:
    data_url = f"data:{mime};base64,{base64.b64encode(file_bytes).decode()}"
    with ui.row().classes("items-center gap-2 no-wrap w-full"):
        ui.icon("picture_as_pdf" if is_pdf else "image", size="1.2rem").classes(
            "text-purple-300"
        )
        ui.label(filename).classes("text-slate-300 text-sm truncate")
        ui.space()
        ui.label(f"{len(file_bytes) / 1024:.0f} KB").classes("text-slate-500 text-xs")

    if is_pdf:
        # Browsers render data-URL PDFs inside an iframe (Chrome, Firefox, Edge).
        ui.html(
            f'<iframe src="{data_url}" '
            'style="width:100%;height:420px;border:0;border-radius:12px;background:#0f172a">'
            "</iframe>"
        ).classes("w-full")
    else:
        ui.image(data_url).classes(
            "w-full rounded-xl object-contain bg-slate-900"
        ).style("min-height:320px;max-height:420px").props("no-spinner")


def _render_empty_state() -> None:
    with ui.column().classes("w-full items-center justify-center gap-3 py-16"):
        ui.icon("text_snippet", size="3rem").classes("text-slate-600")
        ui.label("Upload a document to extract its text").classes("text-slate-500")


def _render_error(message: str) -> None:
    with ui.card().classes("w-full rounded-xl p-4 fade-in").style(
        "background:#3f1d1d;border:1px solid #7f1d1d"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("error", size="1.6rem").classes("text-red-400")
            ui.label("OCR failed").classes("text-red-200 font-semibold")
        ui.label(message).classes("text-red-300 text-sm break-all")
        if SERVER_URL:
            ui.label(
                f"Is the model server running at {SERVER_URL}?"
            ).classes("text-red-400/70 text-xs")


def _render_result(pages: list[str], page_numbers: list[int]) -> None:
    text = "\n\n".join(pages)
    if not text.strip():
        with ui.card().classes("w-full rounded-xl p-4 fade-in").style(
            "background:#1e293b;border:1px solid #475569"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("search_off", size="1.6rem").classes("text-slate-400")
                ui.label("No text was recognised in this document.").classes(
                    "text-slate-300"
                )
        return

    words = len(text.split())
    with ui.row().classes("gap-4 fade-in"):
        stats = ((len(pages), "pages"), (words, "words"), (len(text), "characters"))
        for value, unit in stats:
            with ui.column().classes("gap-0"):
                ui.label(f"{value:,}").classes("text-white text-lg font-bold")
                ui.label(unit).classes("text-slate-400 text-xs uppercase")

    # One tab per page; each page can be viewed rendered or as editable source.
    with ui.tabs().classes("w-full fade-in").props(
        "dense dark active-color=purple-4 indicator-color=purple-4 align=left "
        "outside-arrows mobile-arrows"
    ) as tabs:
        page_tabs = [ui.tab(f"page-{n}", label=f"Page {n}") for n in page_numbers]
    with ui.tab_panels(tabs, value=page_tabs[0]).classes("w-full fade-in").props(
        "animated"
    ).style("background:transparent"):
        for tab, page_text in zip(page_tabs, pages):
            with ui.tab_panel(tab).classes("p-0 pt-3"):
                _render_page(page_text)

    ui.label(
        "For research/education only — verify extracted text against the original document."
    ).classes("text-slate-500 text-xs italic mt-2")


def _render_page(page_text: str) -> None:
    """Rendered markdown with a toggle to view/edit the raw source."""
    with ui.row().classes("w-full items-center justify-end"):
        mode = ui.toggle(
            {"rendered": "Rendered", "source": "Source"}, value="rendered"
        ).props("dense no-caps unelevated toggle-color=purple-7 color=slate-8 text-color=white")

    rendered_box = ui.element("div").classes(
        "w-full rounded-xl p-6 overflow-auto ocr-markdown"
    ).style("background:#0f172a;border:1px solid #334155;min-height:60vh;max-height:80vh")
    with rendered_box:
        rendered = ui.markdown(
            page_text or "*(empty page)*", extras=["tables", "fenced-code-blocks"]
        ).classes("text-slate-200")

    source = ui.textarea(value=page_text).props(
        "outlined dark color=purple-6 input-class=font-mono"
    ).classes("w-full text-base").style("min-height:60vh")
    source.set_visibility(False)

    def switch_mode() -> None:
        show_source = mode.value == "source"
        source.set_visibility(show_source)
        rendered_box.set_visibility(not show_source)
        if not show_source:
            # Re-render edits made in the source view.
            rendered.set_content(source.value or "*(empty page)*")

    mode.on_value_change(switch_mode)


async def _copy_text(state: dict[str, object]) -> None:
    await ui.clipboard.write(str(state["text"]))
    ui.notify("Text copied to clipboard", color="positive")


def _download_text(state: dict[str, object]) -> None:
    stem = os.path.splitext(str(state.get("filename") or "document"))[0]
    ui.download.content(str(state["text"]).encode("utf-8"), f"{stem}.md")


if __name__ == "__main__":
    ui.run(title="Document OCR", port=8080, reload=False, favicon="🩺")
