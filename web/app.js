// Modal endpoint (set after `modal deploy modal_app/app.py`). Override with ?api=<base-url>.
const DEFAULT_API = "https://WORKSPACE--deepfake-detector-detector-web.modal.run";
const API = (new URLSearchParams(location.search).get("api") || DEFAULT_API).replace(/\/+$/, "");
const MAX_BYTES = 10 * 1024 * 1024;
const $ = (id) => document.getElementById(id);
const fileIn = $("file"), drop = $("drop"), preview = $("preview"), dropText = $("dropText");
const runBtn = $("run"), status = $("status"), model = $("model");
let file = null;

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  file = f;
  preview.src = URL.createObjectURL(f);
  preview.hidden = false;
  dropText.hidden = true;
  runBtn.disabled = false;
  $("result").hidden = true;
  status.textContent = "";
}

fileIn.addEventListener("change", () => setFile(fileIn.files[0]));
["dragenter", "dragover"].forEach((e) =>
  drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) =>
  drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => setFile(ev.dataTransfer.files[0]));

function show(r) {
  const p = Math.max(0, Math.min(1, Number(r.p_fake)));
  const pct = (p * 100).toFixed(1);
  const lbl = $("label");
  lbl.textContent = r.label;
  lbl.className = "pill " + (r.label === "fake" ? "fake" : "real");
  $("pct").textContent = `${pct}% likely AI-generated`;
  const fill = $("fill");
  fill.style.width = `${pct}%`;
  fill.className = "fill " + (r.label === "fake" ? "fake" : "real");
  fill.parentElement.setAttribute("aria-valuenow", pct);
  $("modelId").textContent = r.model_id;
  $("version").textContent = r.version;
  $("notes").textContent = r.notes;
  $("result").hidden = false;
}

runBtn.addEventListener("click", async () => {
  if (!file) return;
  runBtn.disabled = true;
  runBtn.classList.add("loading");
  status.textContent = "Analysing… if the server is asleep, the first request can take about a minute while it wakes.";
  try {
    if (file.size > MAX_BYTES) throw new Error("Image is larger than 10 MB");
    const body = new FormData();
    body.append("file", file, file.name || "image");
    body.append("ensemble", model.value);
    const resp = await fetch(`${API}/detect`, { method: "POST", body });
    let r = null;
    try { r = await resp.json(); } catch { /* non-JSON error page */ }
    if (!resp.ok) throw new Error(r?.detail || `Server error ${resp.status}`);
    if (!r || r.error) throw new Error(r?.error || "Empty response");
    show(r);
    status.textContent = "";
  } catch (e) {
    console.error(e);
    status.textContent = "Something went wrong: " + (e?.message || e) + ". Try again in a minute.";
  } finally {
    runBtn.disabled = false;
    runBtn.classList.remove("loading");
  }
});
