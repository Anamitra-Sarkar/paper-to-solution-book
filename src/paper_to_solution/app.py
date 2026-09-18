"""Demo web UI + JSON API (FastAPI). Upload image -> see extracted questions -> answers."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from paper_to_solution.extractor import ExtractionError, extract_questions_from_image
from paper_to_solution.graph import run_graph
from paper_to_solution.image_io import ImageError

app = FastAPI(title="Paper to Solution Book (prototype)")

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Paper to Solution Book</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
.card{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}
.q{font-weight:600}.meta{color:#555;font-size:.9em}pre{white-space:pre-wrap;background:#f6f6f6;padding:.75rem;border-radius:6px}
.err{color:#a00}</style></head><body>
<h1>Paper to Solution Book (prototype)</h1>
<p>Upload a question-paper image (PNG/JPEG/WebP). Vision: <code>qwen/qwen3.8-27b</code> via Groq.</p>
<form id="f"><input type="file" name="file" accept=".png,.jpg,.jpeg,.webp" required>
<button>Extract</button> <label><input type="checkbox" id="solve" checked> also solve via LangGraph</label></form>
<div id="out"></div>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const solve = document.getElementById('solve').checked;
  document.getElementById('out').innerHTML = '<p>Working…</p>';
  const r = await fetch('/api/extract' + (solve ? '?solve=true' : ''), {method:'POST', body:fd});
  const j = await r.json();
  if (!r.ok) { document.getElementById('out').innerHTML = '<p class="err">Error: ' + (j.detail||r.status) + '</p>'; return; }
  let h = '<h2>Extracted questions (' + j.questions.length + ')</h2>';
  for (const q of j.questions) {
    h += '<div class="card"><div class="q">Q' + q.question_number + ' <span class="meta">[' +
      (q.marks ?? '?') + ' marks · ' + q.question_type + ' · page ' + q.source_page +
      ' · conf ' + q.confidence + ']</span></div><p>' + q.question_text + '</p>';
    if (q.options && q.options.length) h += '<ul>' + q.options.map(o=>'<li>'+o+'</li>').join('') + '</ul>';
    if (q.subquestions && q.subquestions.length) h += '<ul>' + q.subquestions.map(s=>'<li>(' + s.label + ') ' + s.text + '</li>').join('') + '</ul>';
    if (q.extraction_notes) h += '<div class="meta">Note: ' + q.extraction_notes + '</div>';
    const s = (j.solutions||[]).find(x=>x.question_number===q.question_number);
    if (s) h += '<pre>' + s.answer + '</pre>';
    h += '</div>';
  }
  if (j.errors && j.errors.length) h += '<p class="err">' + j.errors.join('<br>') + '</p>';
  document.getElementById('out').innerHTML = h;
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/api/extract")
async def api_extract(file: UploadFile = File(...), solve: bool = False):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        return JSONResponse({"detail": f"Unsupported type '{suffix}'. Use PNG/JPEG/WebP."}, status_code=400)
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        extraction = extract_questions_from_image(tmp_path)
    except (ImageError, ExtractionError, RuntimeError) as e:
        return JSONResponse({"detail": str(e)}, status_code=502 if isinstance(e, ExtractionError) else 400)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
    out = {"questions": [q.model_dump() for q in extraction.questions], "errors": []}
    if solve and extraction.questions:
        final = run_graph(extraction.questions)
        out["solutions"] = final["solutions"]
        out["errors"] = final["errors"]
    return JSONResponse(out)
