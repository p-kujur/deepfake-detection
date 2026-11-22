import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client@1.19.1/dist/index.min.js";

const SPACE = new URLSearchParams(location.search).get("space") || "ezra9426/deepfake-detector";
const $ = (id) => document.getElementById(id);
const fileIn = $("file"), drop = $("drop"), preview = $("preview"), dropText = $("dropText");
const runBtn = $("run"), status = $("status"), model = $("model");
let file = null;
let clientPromise = null;

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

function getClient() {
  if (!clientPromise) {
    clientPromise = Client.connect(SPACE, {
      status_callback: (s) => {
        if (s.status === "sleeping" || s.status === "starting" || s.stage === "pending")
          status.textContent = "Waking the Space up… the first request can take about a minute.";
      },
    }).catch((e) => { clientPromise = null; throw e; });
  }
  return clientPromise;
}

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
  status.textContent = "Connecting… if the Space is asleep, the first request can take about a minute.";
  try {
    const app = await getClient();
    status.textContent = "Analysing image…";
    const res = await app.predict("/predict", { image: file, ensemble: model.value });
    const r = res.data[0];
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
