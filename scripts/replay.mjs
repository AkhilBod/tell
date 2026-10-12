// Reruns recorded rounds through the current reader.js + brain.js and compares with what happened live.
// Usage: npm run replay -- recordings/<session>   (defaults to the newest session)
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { NAMES, createEngine, outcome } from "../public/brain.js";
import { createReader } from "../public/reader.js";

const root = path.resolve(import.meta.dirname, "..");
let dir = process.argv[2];
if (!dir) {
  const sessions = (await readdir(path.join(root, "recordings"))).sort();
  dir = path.join("recordings", sessions.at(-1));
}
const files = (await readdir(path.resolve(root, dir))).filter((f) => f.endsWith(".json")).sort();
const logs = [];
for (const file of files) {
  const log = JSON.parse(await readFile(path.resolve(root, dir, file), "utf8"));
  if ((log.mode ?? "read") === "read") logs.push(log);
}

// Seeded so two replays differ only by the code change, not by the pattern engine's random tie-breaks.
const seeds = Number(process.env.SEEDS) || 50;
const rngFor = (seed) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const N = (m) => (m == null ? "-" : NAMES[m][0]);
const res = (o) => (o > 0 ? "win " : o < 0 ? "loss" : "draw");

function run(seed) {
  const reader = createReader();
  const brain = createEngine();
  const rng = rngFor(seed);
  const tally = { live: [0, 0, 0], now: [0, 0, 0], sources: {} };
  const lines = [];

  for (const log of logs) {
    const r = reader.round();
    const pattern = brain.decide(rng);
    let lastHand = false;
    for (const f of log.frames) {
      if (f.t + f.lag >= 0) break;
      reader.see(r, f.t, f.e);
      lastHand = !!f.e;
    }
    const d = reader.decide(r, pattern, lastHand);
    const final = log.final;
    if (final == null) continue;

    const was = outcome(log.decision.play, final);
    const now = outcome(d.play, final);
    tally.live[1 - was]++;
    tally.now[1 - now]++;
    const src = `${d.source}:${d.kind}`;
    tally.sources[src] ??= [0, 0];
    tally.sources[src][1]++;
    if (d.guess === final) tally.sources[src][0]++;
    reader.learn(r, final);
    brain.record(final, d.play);

    const mark = now > was ? "better" : now < was ? "WORSE" : "";
    const tell = r.tell ? `${N(r.tell.guess)}@${r.tell.seenAt}` : "-";
    lines.push(`#${String(log.round).padStart(3)}  held ${N(r.held)}  tell ${tell.padEnd(6)} ${src.padEnd(16)} you ${N(final)}  live ${N(log.decision.play)} ${res(was)}  now ${N(d.play)} ${res(now)}  ${mark}`);
  }
  return { tally, lines };
}

const fmt = ([w, d, l]) => `${w} wins, ${d} draws, ${l} losses (machine takes ${Math.round((100 * w) / Math.max(1, w + l))}% of decided rounds)`;
const first = run(1);
const totals = [0, 0, 0];
for (let seed = 1; seed <= seeds; seed++) run(seed).tally.now.forEach((v, k) => (totals[k] += v));

console.log(first.lines.join("\n"));
console.log(`\nLive:   ${fmt(first.tally.live)}`);
console.log(`Replay: ${fmt(first.tally.now)}  (seed 1)`);
console.log(`Replay: ${fmt(totals.map((v) => Math.round((v / seeds) * 10) / 10))}  (average of ${seeds} seeds)`);
console.log("Reads that matched your throw (seed 1):", Object.entries(first.tally.sources).map(([k, [ok, n]]) => `${k} ${ok}/${n}`).join(", "));
