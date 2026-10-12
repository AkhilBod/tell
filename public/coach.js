// Reads an opponent's hand frame by frame and decides whether to tip you off. It only ever sees frames up to "now",
// never a label, so it works the same on a recorded clip as on a real person.
//
// Strategy: you hold a fist. A fist already beats Scissors and ties Rock, so the only throw you need warning about is
// Paper, and Paper is also the one that shows earliest (all four fingers opening is a big movement). So I commit to my
// first confident read of their throw, and only a Paper read produces a cue: "s", throw Scissors.
import { PAPER, ROCK, SETTINGS, avg, heading, shapeOf } from "./reader.js";

export const COACH = {
  // Tuned with npm run coach-replay on your clips and practice rounds: most of the early Paper calls, almost no false "s".
  MOVE_T: 0.2,     // how far their hand must open toward a shape before it counts as a tell
  CONFIRM: 2,      // frames a read must hold before I commit to it
  RISE: 0.1,       // and each confirming frame must open further: real throws keep opening, a loose fist stalls
  SHAPE_T: 0.55,   // confidence for a formed shape after Shoot
  ROCK_AT: 150,    // a fist still held this long after Shoot is a Rock
};

export function createCoach(S = SETTINGS, C = COACH) {
  const settings = { ...S, MOVE_T: C.MOVE_T };
  return {
    round() {
      return { base: [], held: undefined, streak: null, read: null, cues: [] };
    },

    // One opponent frame. t: when it was captured, relative to Shoot. now: when I saw it. e: finger openness or null.
    // Returns a cue { opp, why, t, at } the moment I commit to reading Paper, otherwise null.
    see(r, t, now, e) {
      if (r.read) return null;
      if (!e) {
        r.streak = null;
        return null;
      }
      if (t < settings.ARM) {
        if (t >= settings.ARM - 200) r.base.push(e.slice());
        return null;
      }
      if (r.held === undefined) {
        const s = r.base.length ? shapeOf(avg(r.base)) : null;
        r.held = s && s.conf >= settings.HOLD_T ? s.move : null;
      }
      if (!r.base.length) r.base.push(e.slice());

      let guess = null;
      let why = null;
      let progress = null;
      const s = shapeOf(e);
      if (t >= 0 && s.conf >= C.SHAPE_T && (s.move !== ROCK || t >= C.ROCK_AT)) {
        guess = s.move;
        why = "shape";
      } else {
        const h = heading(avg(r.base), e, r.held, settings);
        if (h) {
          guess = h.move;
          why = "tell";
          progress = h.progress;
        }
      }
      if (guess === null) {
        r.streak = null;
        return null;
      }
      const rising = progress === null || r.streak?.progress == null || progress >= r.streak.progress + C.RISE;
      r.streak = r.streak?.guess === guess && rising ? { guess, n: r.streak.n + 1, progress } : { guess, n: 1, progress };
      if (r.streak.n < C.CONFIRM) return null;
      r.read = { opp: guess, why, t: Math.round(t), at: Math.round(now) };
      if (guess !== PAPER) return null;
      r.cues.push(r.read);
      return r.read;
    },
  };
}
