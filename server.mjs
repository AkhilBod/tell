import { execFile } from "node:child_process";
import http from "node:http";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(ROOT, "public");
const RECORDINGS = path.join(ROOT, "recordings");
const CLIPS = path.join(ROOT, "clips");
const FRAMES = path.join(CLIPS, "frames");
const run = promisify(execFile);
const PORT = Number(process.env.PORT) || 5199;
const MAX_BODY = 8 * 1024 * 1024;
const MAX_CLIP = 20 * 1024 * 1024;
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".webm": "video/webm",
};

function send(res, status, text) {
  res.writeHead(status, { "content-type": "text/plain; charset=utf-8" });
  res.end(text);
}

function sendJson(res, data) {
  res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
  res.end(JSON.stringify(data));
}

// Serves a file from dir, refusing anything that resolves outside it.
async function serveFrom(dir, rel, req, res) {
  const file = path.normalize(path.join(dir, rel));
  if (!file.startsWith(dir + path.sep)) return send(res, 403, "Forbidden");
  try {
    const body = await readFile(file);
    res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream", "cache-control": "no-store" });
    res.end(req.method === "HEAD" ? undefined : body);
  } catch {
    send(res, 404, "Not found");
  }
}

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error("Body too large"));
        req.destroy();
      } else chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

// Saves one round: recordings/<session>/round-NNN.json (frame data) and .jpg (contact sheet).
async function saveRound(req, res) {
  let data;
  try {
    data = JSON.parse((await readBody(req, MAX_BODY)).toString("utf8"));
  } catch {
    return send(res, 400, "Expected a JSON body under 8 MB");
  }
  const session = String(data.session ?? "").replace(/[^0-9A-Za-z-]/g, "").slice(0, 40);
  const round = Number.parseInt(data.round, 10);
  if (!session || !Number.isInteger(round) || round < 1) return send(res, 400, "Missing session or round");

  const dir = path.join(RECORDINGS, session);
  const name = `round-${String(round).padStart(3, "0")}`;
  await mkdir(dir, { recursive: true });
  await writeFile(path.join(dir, `${name}.json`), JSON.stringify(data.log ?? {}));
  const sheet = /^data:image\/jpeg;base64,([A-Za-z0-9+/=]+)$/.exec(data.sheet ?? "");
  if (sheet) await writeFile(path.join(dir, `${name}.jpg`), Buffer.from(sheet[1], "base64"));
  sendJson(res, { saved: path.relative(ROOT, path.join(dir, name)) });
}

// Saves an unlabeled opponent clip: clips/<id>.webm plus <id>.json with where Shoot falls in the video.
async function saveClip(req, res) {
  const url = new URL(req.url, "http://localhost");
  const shoot = Number(url.searchParams.get("shoot"));
  if (!Number.isFinite(shoot) || shoot < 0 || shoot > 10000) return send(res, 400, "Missing shoot offset");
  let body;
  try {
    body = await readBody(req, MAX_CLIP);
  } catch {
    return send(res, 400, "Clip must be under 20 MB");
  }
  if (!body.length) return send(res, 400, "Empty clip");
  await mkdir(CLIPS, { recursive: true });
  const id = `clip-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
  await writeFile(path.join(CLIPS, `${id}.webm`), body);
  await writeFile(path.join(CLIPS, `${id}.json`), JSON.stringify({ shootOffset: Math.round(shoot), createdAt: new Date().toISOString() }));
  sendJson(res, { id });
}

// Every frame of a clip as a JPEG plus its exact timestamp, extracted once with ffmpeg and cached.
// The clip test reads these instead of seeking a <video>, which browsers stop redrawing in background tabs.
async function clipFrames(res, id) {
  if (!CLIP_ID.test(id)) return send(res, 400, "Bad clip id");
  const dir = path.join(FRAMES, id);
  const index = path.join(dir, "index.json");
  try {
    return sendJson(res, JSON.parse(await readFile(index, "utf8")));
  } catch {}
  await mkdir(dir, { recursive: true });
  let stderr;
  try {
    ({ stderr } = await run("ffmpeg", ["-v", "info", "-y", "-i", path.join(CLIPS, `${id}.webm`), "-vf", "showinfo,scale=640:-2", "-fps_mode", "passthrough", "-q:v", "4", path.join(dir, "%04d.jpg")], { maxBuffer: 64 << 20 }));
  } catch {
    return send(res, 500, "Couldn't extract frames. The clip test needs ffmpeg installed.");
  }
  const frames = [...stderr.matchAll(/pts_time:\s*([\d.]+)/g)].map((m, i) => ({ t: Math.round(Number.parseFloat(m[1]) * 1000), url: `/clip-frames/${id}/${String(i + 1).padStart(4, "0")}.jpg` }));
  await writeFile(index, JSON.stringify(frames));
  sendJson(res, frames);
}

const CLIP_ID = /^clip-[a-z0-9-]+$/;

// Stores what the judge saw in a clip (its throw, and its clearest fist before opening). For a Paper or Scissors clip
// it also makes a Rock version: the same pump, then frozen on that fist instead of opening.
async function labelClip(req, res) {
  let data;
  try {
    data = JSON.parse((await readBody(req, 64 * 1024)).toString("utf8"));
  } catch {
    return send(res, 400, "Expected JSON");
  }
  const { id, judged, freezeAt } = data;
  if (!CLIP_ID.test(id ?? "") || ![0, 1, 2, null].includes(judged)) return send(res, 400, "Bad label");
  const metaPath = path.join(CLIPS, `${id}.json`);
  const meta = JSON.parse(await readFile(metaPath, "utf8"));
  meta.judged = judged;
  if (Number.isFinite(freezeAt)) meta.freezeAt = Math.round(freezeAt);
  await writeFile(metaPath, JSON.stringify(meta));

  let rock = null;
  if (!meta.generated && (judged === 1 || judged === 2) && Number.isFinite(freezeAt) && freezeAt > 300) {
    rock = `clip-r-${id.slice("clip-".length)}`;
    const hold = (meta.shootOffset + 1200 - freezeAt) / 1000;
    try {
      await run("ffmpeg", ["-v", "error", "-y", "-i", path.join(CLIPS, `${id}.webm`),
        "-vf", `trim=end=${((freezeAt + 20) / 1000).toFixed(3)},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=${hold.toFixed(3)}`,
        "-an", "-c:v", "libvpx", "-b:v", "2500k", path.join(CLIPS, `${rock}.webm`)]);
      await writeFile(path.join(CLIPS, `${rock}.json`), JSON.stringify({ shootOffset: meta.shootOffset, judged: 0, generated: true, from: id, createdAt: new Date().toISOString() }));
    } catch {
      rock = null;
    }
  }
  sendJson(res, { id, judged, rock });
}

async function listClips(res) {
  let files = [];
  try {
    files = await readdir(CLIPS);
  } catch {}
  const clips = [];
  for (const f of files.filter((f) => f.endsWith(".json")).sort()) {
    const id = f.slice(0, -5);
    if (!files.includes(`${id}.webm`)) continue;
    try {
      const meta = JSON.parse(await readFile(path.join(CLIPS, f), "utf8"));
      clips.push({ id, shootOffset: meta.shootOffset, judged: meta.judged, generated: !!meta.generated });
    } catch {}
  }
  sendJson(res, clips);
}

const server = http.createServer((req, res) => {
  const fail = () => send(res, 500, "Server error");
  const pathname = req.url.split("?")[0];
  if (req.method === "POST" && pathname === "/api/record") return void saveRound(req, res).catch(fail);
  if (req.method === "POST" && pathname === "/api/clip") return void saveClip(req, res).catch(fail);
  if (req.method === "GET" && pathname === "/api/clips") return void listClips(res).catch(fail);
  if (req.method === "POST" && pathname === "/api/clip-label") return void labelClip(req, res).catch(fail);
  if (req.method === "GET" && pathname === "/api/clip-frames") return void clipFrames(res, new URL(req.url, "http://localhost").searchParams.get("id") ?? "").catch(fail);
  if (req.method !== "GET" && req.method !== "HEAD") return send(res, 405, "Method not allowed");

  let rel;
  try {
    rel = decodeURIComponent(pathname);
  } catch {
    return send(res, 400, "Bad path");
  }
  if (rel.startsWith("/clips/") && rel.endsWith(".webm")) return void serveFrom(CLIPS, rel.slice("/clips/".length), req, res);
  if (rel.startsWith("/clip-frames/") && rel.endsWith(".jpg")) return void serveFrom(FRAMES, rel.slice("/clip-frames/".length), req, res);
  if (rel.endsWith("/")) rel += "index.html";
  serveFrom(PUBLIC, rel, req, res);
});

server.listen(PORT, "127.0.0.1", () => console.log(`Tell is running at http://localhost:${PORT}`));
