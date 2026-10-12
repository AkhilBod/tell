// Decides my throw from what the camera saw before Shoot. No DOM here, so scripts/replay.mjs can rerun recordings through it.
import { beats } from "./brain.js";

export const ROCK = 0;
export const PAPER = 1;
export const SCISSORS = 2;

export const SETTINGS = {
  ARM: -320,       // start reading tells here; the 200 ms before it set the pose you're holding
  MOVE_T: 0.25,    // how far your hand must travel toward a new shape before it counts as a tell
  ALIGN_T: 0.8,    // how cleanly that motion has to point at the shape
  RETREAT: 0.4,    // pulling back below this share of the furthest you got toward a shape withdraws the tell
  HOLD_T: 0.45,    // confidence for reading a held pose or your final throw
  STREAK: 2,       // frames a tell must hold before I trust it
  TELL_PRIOR: 2,   // trust in a tell that starts from a fist, before I've seen whether you fake them
  BAIT_PRIOR: 1,   // trust in a tell that starts from a held Paper or Scissors, which is usually bait
};

// Finger openness [index, middle, ring, pinky] of a textbook Rock, Paper and Scissors.
const PROTO = [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]];

export const avg = (rows) => rows[0].map((_, k) => rows.reduce((s, r) => s + r[k], 0) / rows.length);

// How far you've gone from the held pose b toward a shape: 0 = still at b, 1 = all the way there.
function progress(b, e, move) {
  const dir = PROTO[move].map((v, k) => v - b[k]);
  const n2 = dir.reduce((s, v) => s + v * v, 0) || 1e-6;
  return e.reduce((s, v, k) => s + (v - b[k]) * dir[k], 0) / n2;
}

export function shapeOf(e) {
  const [i, m, r, p] = e;
  const scores = [
    Math.min(1 - i, 1 - m, 1 - r, 1 - p),
    Math.min(i, m, r, p),
    Math.min(i, m, 1 - r, 1 - p),
  ];
  let move = 0;
  for (let k = 1; k < 3; k++) if (scores[k] > scores[move]) move = k;
  return { move, conf: scores[move] };
}

// Which shape your hand is moving toward, measured from the pose you held going into the last beat.
// Holding still is not a tell, and neither is tightening into the shape you already hold.
export function heading(b, e, held, S = SETTINGS) {
  const d = e.map((v, k) => v - b[k]);
  const dn = Math.hypot(...d);
  if (dn < 0.12) return null;
  let best = null;
  PROTO.forEach((p, move) => {
    if (move === held) return;
    const dir = p.map((v, k) => v - b[k]);
    const n = Math.hypot(...dir);
    if (n < 0.3) return;
    const dot = d.reduce((s, v, k) => s + v * dir[k], 0);
    const align = dot / (dn * n);
    if (!best || align > best.align) best = { move, align, progress: dot / (n * n) };
  });
  return best && best.align >= S.ALIGN_T && best.progress >= S.MOVE_T ? best : null;
}

function heldOf(base, S) {
  if (!base.length) return null;
  const s = shapeOf(avg(base));
  return s.conf >= S.HOLD_T ? s.move : null;
}

export function createReader(S = SETTINGS) {
  // What you actually threw after each kind of pre-Shoot evidence, e.g. habits["held:1"] = [rock, paper, scissors] counts.
  const habits = {};

  return {
    habits,

    round() {
      return { base: [], held: undefined, streak: null, retreat: 0, tell: null, lastUsed: null, key: null };
    },

    // One camera frame I saw before Shoot. t: capture time relative to Shoot. e: finger openness, or null with no hand.
    see(r, t, e) {
      r.lastUsed = Math.round(t);
      if (t < S.ARM) {
        if (e && t >= S.ARM - 200) r.base.push(e.slice());
        return;
      }
      if (r.held === undefined) r.held = heldOf(r.base, S);
      if (!e) {
        r.streak = null;
        return;
      }
      if (!r.base.length) r.base.push(e.slice());
      const b = avg(r.base);
      if (r.tell) {
        const now = progress(b, e, r.tell.guess);
        r.tell.peak = Math.max(r.tell.peak, now);
        r.retreat = now < S.RETREAT * r.tell.peak ? r.retreat + 1 : 0;
        if (r.retreat >= S.STREAK) {
          r.tell = null;
          r.retreat = 0;
        }
      }
      const h = heading(b, e, r.held, S);
      if (!h) {
        r.streak = null;
        return;
      }
      r.streak = r.streak?.guess === h.move ? { guess: h.move, n: r.streak.n + 1 } : { guess: h.move, n: 1 };
      if (r.streak.n >= S.STREAK && r.tell?.guess !== h.move) r.tell = { guess: h.move, seenAt: Math.round(t), peak: h.progress };
    },

    // Called on the Shoot beat. pattern: the brain's pick for this round. handVisible: whether I can see your hand now.
    decide(r, pattern, handVisible) {
      if (r.held === undefined) r.held = heldOf(r.base, S);
      const held = r.held;
      r.key = r.tell ? `tell:${held ?? "x"}>${r.tell.guess}` : held !== null && held !== ROCK ? `held:${held}` : null;
      const seen = r.key ? (habits[r.key] ?? [0, 0, 0]).slice() : null;
      const base = { key: r.key, held, seen, lastUsed: r.lastUsed };

      if (r.tell) {
        const odds = seen.slice();
        odds[r.tell.guess] += held === ROCK ? S.TELL_PRIOR : S.BAIT_PRIOR;
        const top = Math.max(...odds);
        const guess = odds[r.tell.guess] >= top ? r.tell.guess : odds.indexOf(top);
        return { ...base, ...r.tell, guess, tellGuess: r.tell.guess, play: beats(guess), source: "camera", kind: guess === r.tell.guess ? "tell" : "fake" };
      }
      if (r.key) {
        const total = seen[0] + seen[1] + seen[2];
        if (total >= 2) {
          const guess = seen.indexOf(Math.max(...seen));
          return { ...base, guess, play: beats(guess), source: "habit", kind: "habit" };
        }
        // You're holding a shape still, so you'll probably switch off it. Throwing what it beats wins or ties against both other shapes.
        return { ...base, guess: (held + 1) % 3, play: (held + 2) % 3, source: "habit", kind: "bait" };
      }
      return { ...base, guess: pattern.assumed, play: pattern.play, source: "pattern", kind: handVisible ? "fist" : "blind", why: pattern.why, label: pattern.label };
    },

    learn(r, final) {
      if (r.key && final !== null) (habits[r.key] ??= [0, 0, 0])[final]++;
    },
  };
}
