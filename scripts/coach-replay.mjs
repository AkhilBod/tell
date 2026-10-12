// Reruns the coach over the finger traces saved by "Test my reads on every clip", so coach settings can be tuned
// without replaying video. Usage: npm run coach-replay [-- recordings/coach-test-<time>] [-- key=value ...]
// e.g. npm run coach-replay -- MOVE_T=0.25 CONFIRM=3
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { COACH, createCoach } from "../public/coach.js";

const root = path.resolve(import.meta.dirname, "..");
const args = process.argv.slice(2);
let dir = args.find((a) => !a.includes("="));
if (!dir) dir = path.join("recordings", (await readdir(path.join(root, "recordings"))).filter((d) => d.startsWith("coach-test-")).sort().at(-1));
const overrides = Object.fromEntries(args.filter((a) => a.includes("=")).map((a) => [a.split("=")[0], Number(a.split("=")[1])]));
const { rows } = JSON.parse(await readFile(path.resolve(root, dir, "round-001.json"), "utf8"));
if (!rows[0]?.trace) throw new Error("This test predates saved traces. Run the clip test again.");

const coach = createCoach(undefined, { ...COACH, ...overrides });
const N = ["Rock", "Paper", "Scissors"];
const lag = 25;
let saidS = [];
const falseS = [];
let missed = 0;
for (const row of rows) {
  if (row.final == null) continue;
  const r = coach.round();
  for (const [t, e] of row.trace) coach.see(r, t, t + lag, e);
  const said = r.read?.opp === 1;
  if (row.final === 1) said ? saidS.push(r.read.at) : missed++;
  else if (said) falseS.push(`${row.clip} (${N[row.final]}) at ${r.read.at} ms`);
}
saidS = saidS.sort((a, b) => a - b);
const papers = saidS.length + missed;
console.log(`settings: ${JSON.stringify({ ...COACH, ...overrides })}`);
console.log(`vs Paper: said "s" on ${saidS.length} of ${papers}, median ${saidS[saidS.length >> 1] ?? "-"} ms, 100+ ms early on ${saidS.filter((x) => x <= -100).length}`);
console.log(`false "s" on ${falseS.length} of ${rows.filter((r) => r.final != null && r.final !== 1).length} Rock/Scissors clips${falseS.length ? `: ${falseS.join(", ")}` : ""}`);
