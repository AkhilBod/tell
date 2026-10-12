import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { NAMES, beats, createEngine, outcome } from "./brain.js";
import { createCoach } from "./coach.js";
import { PAPER, ROCK, SCISSORS, SETTINGS, avg, createReader, shapeOf } from "./reader.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Times are in ms relative to SHOOT (t = 0). The tell-reading settings live in reader.js.
const BEAT = 480;       // gap between Rock, Paper, Scissors, Shoot
const LEAD_IN = 600;    // pause before the first beat
const READ_FROM = 150;  // your throw is whatever you hold in this window
const READ_TO = 350;
const LATE_TO = 600;    // keep watching until here to catch late switches

// "You read me" mode: how early my cue comes. It moves later each time you win and earlier each time you don't,
// which settles where you beat me about 70% of the time.
const LEAD_START = 800;
const LEAD_MIN = 60;
const LEAD_MAX = 1400;
const STEP_WIN = 30;
const STEP_MISS = 70;
const MOVE_FROM_CUE = 0.25;  // finger travel that counts as you starting to move
const HUMAN_FASTEST = 100;   // reactions quicker than this are guesses, not reactions
const LETTERS = ["R", "P", "S"];  // every cue names the throw to make: Rock, Paper or Scissors
const { ARM, HOLD_T } = SETTINGS;

// "I coach you" mode.
const CLIP_BEFORE = 3 * BEAT + 400;  // clips start 400 ms before the first beat
const CLIP_AFTER = 900;              // and run this long past Shoot
const CLIPS_PER_BATCH = 10;
const PLAN_WINDOW = 12;   // their last 12 throws decide their favorite
const PLAN_MIN = 3;       // throws I need to see before I trust a favorite
const PLAN_P = 0.05;      // a favorite has to be this unlikely to happen by chance before I act on it
const HUMAN_SHIFT = 120;  // a generated opponent throws up to this many ms early or late
const HUMAN_SPEED = 0.1;  // and moves up to 10% faster or slower
const PLAN_AT = -1200;    // the pre-round letter comes 1.2 s before Shoot, between the Rock and Paper beats

const FINGERS = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"];
const PALM = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17]];
const SHEET_TIMES = [-640, -480, -400, -320, -240, -180, -120, -60, 0, 60, 120, 180, 250, 350, 450, 600];
const THUMB_W = 320;

const SVG = (inner) => `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${inner}</svg>`;
const ICONS = [
  SVG('<path d="M12 30 9 21 15 12 26 9 36 13 40 23 37 33 27 38 16 37Z"/><path d="M15 12 20 22 9 21M20 22 30 24 36 13M30 24 37 33M20 22 16 37M30 24 27 38M20 22 12 30" stroke-width="1.6" opacity=".55"/>'),
  SVG('<path d="M13 6h16l7 7v29H13Z"/><path d="M29 6v7h7"/><path d="M18 20h13M18 26h13M18 32h9" stroke-width="1.8" opacity=".55"/>'),
  SVG('<circle cx="15" cy="36" r="5.5"/><circle cx="33" cy="36" r="5.5"/><path d="M18.5 31.5 34 8M29.5 31.5 14 8"/><circle cx="24" cy="23.2" r="1.5" fill="currentColor" stroke="none"/>'),
];
const EYE = SVG('<path d="M4 24s7-12 20-12 20 12 20 12-7 12-20 12S4 24 4 24Z"/><circle cx="24" cy="24" r="5.5"/>');

const $ = (id) => document.getElementById(id);
const el = Object.fromEntries(
  ["video", "overlay", "stage", "gate", "startCam", "err", "seen", "machineThrow", "call", "banner", "play", "auto", "sound", "record", "recStatus",
   "mine", "fingers", "chart", "story", "lastNo", "sM", "sD", "sH", "stReads", "stCam", "stHabit", "stPat", "stLag",
   "modeRead", "modeReact", "cueType", "cueFlash", "cues", "stLead", "stBeat", "stRt", "stBest",
   "modeCoach", "oppVideo", "oppOverlay", "sides", "sideL", "sideR", "cueDot", "oppSource", "oppSide", "oppSideWrap", "coachCue",
   "judgeView", "clipCount", "recClips", "testClips", "clipStatus", "testResult", "stCalls", "stFirst", "stOnTime", "stLate", "stReact",
   "sMLab", "sHLab", "title", "stSeen", "newOpp"].map((id) => [id, $(id)])
);
const css = getComputedStyle(document.documentElement);
const tok = (name) => css.getPropertyValue(name).trim();
const COLOR = {
  ink: tok("--ink"), muted: tok("--muted"), line: tok("--line"), panel2: tok("--panel-2"), bg: tok("--bg"),
  machine: tok("--machine"), human: tok("--human"),
  fingers: ["--f-index", "--f-middle", "--f-ring", "--f-pinky"].map(tok),
};

const clamp = (x, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
const median = (xs) => (xs.length ? xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)] : null);
const r3 = (x) => Math.round(x * 1000) / 1000;
const when = (t) => (t < 0 ? `${Math.round(-t)} ms before Shoot` : `${Math.round(t)} ms after Shoot`);
const SESSION = new Date().toISOString().slice(0, 19).replace(/:/g, "-");

const brain = createEngine();
const reader = createReader();
const coach = createCoach();
let oppLandmarker = null;
let oppSmooth = null;
let opp = null;
let clips = [];
let lastClipId = null;
let busy = null;
let stopRecording = false;
const voice = [];
let landmarker = null;
let audio = null;
let smooth = null;
let live = null;
let liveLm = null;
let round = null;
let roundNo = 0;
let lastRound = null;
let autoTimer = 0;
let lastVideoTime = -1;
const lags = [];
let mode = "read";
let lead = LEAD_START;
function newOpponent() {
  return { m: 0, d: 0, h: 0, played: 0, calls: [0, 0], firsts: [], onTime: [0, 0], late: [0, 0], rts: [], seen: [] };
}

const tallies = {
  read: { m: 0, d: 0, h: 0, played: 0, right: 0, camera: [0, 0], habit: [0, 0], pattern: [0, 0] },
  react: { m: 0, d: 0, h: 0, played: 0, rts: [], best: null },
  coach: newOpponent(),
};

// How straight each finger is, 0 (curled into the palm) to 1 (fully extended).
// Blends the bend at the middle knuckle with how far the fingertip reaches from the wrist.
function openness(lm, w, h) {
  const P = (i) => [lm[i].x * w, lm[i].y * h, lm[i].z * w];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const len = (v) => Math.hypot(v[0], v[1], v[2]) || 1e-6;
  const wrist = P(0);
  return FINGERS.map(([m, p, , t]) => {
    const M = P(m);
    const Pp = P(p);
    const T = P(t);
    const a = sub(Pp, M);
    const b = sub(T, Pp);
    const cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (len(a) * len(b));
    const bend = clamp((cos + 0.3) / 1.2);
    const reach = clamp((len(sub(T, wrist)) / len(sub(M, wrist)) - 1.05) / 0.75);
    return (bend + reach) / 2;
  });
}

async function startCamera() {
  el.startCam.disabled = true;
  el.startCam.textContent = "Starting camera…";
  el.err.hidden = true;
  ensureAudio();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 60 } },
    });
    el.video.srcObject = stream;
    await el.video.play();
  } catch (e) {
    fail(e?.name === "NotAllowedError"
      ? "Camera access was blocked. Allow the camera for this page in your browser's site settings, then reload."
      : `Couldn't start the camera (${e?.name || e?.message || "unknown error"}). Close other apps using it and try again.`);
    return;
  }
  el.startCam.textContent = "Loading hand tracking…";
  try {
    landmarker = await createLandmarker();
  } catch {
    fail("Couldn't load the hand-tracking model. Check your internet connection and reload.");
    return;
  }
  el.stage.style.aspectRatio = `${el.video.videoWidth} / ${el.video.videoHeight}`;
  el.stage.style.setProperty("--cam-ar", `${el.video.videoWidth} / ${el.video.videoHeight}`);
  el.gate.hidden = true;
  el.seen.hidden = false;
  el.play.disabled = false;
  setMine("idle");
  pump();
}

function ensureAudio() {
  try {
    audio ??= new (window.AudioContext || window.webkitAudioContext)();
    audio.resume?.();
    return loadVoice();
  } catch {
    return Promise.resolve();
  }
}

// Spoken R, P and S, recorded once with the Mac's built-in voice and trimmed so the letter starts instantly.
function loadVoice() {
  voice.ready ??= (async () => {
    try {
      for (const [m, letter] of LETTERS.entries()) {
        const res = await fetch(`sounds/${letter}.wav`);
        voice[m] = await audio.decodeAudioData(await res.arrayBuffer());
      }
    } catch {
      voice.ready = null;
    }
  })();
  return voice.ready;
}

function fail(msg) {
  el.err.textContent = msg;
  el.err.hidden = false;
  el.startCam.disabled = false;
  el.startCam.textContent = "Try again";
}

async function createLandmarker(numHands = 2) {
  const files = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(files, options("GPU"));
  } catch {
    return HandLandmarker.createFromOptions(files, options("CPU"));
  }
}

function pump() {
  const rvfc = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
  const next = () => (rvfc ? el.video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick));
  const tick = (_now, meta) => {
    if (el.video.currentTime !== lastVideoTime) {
      lastVideoTime = el.video.currentTime;
      const ts = performance.now();
      const res = landmarker.detectForVideo(el.video, ts);
      const cap = meta?.captureTime;
      onFrame(res, cap && cap <= ts && cap > ts - 1000 ? cap : ts, performance.now());
    }
    next();
  };
  next();
}

// The biggest hand in view is yours. In live coaching, the hand on the opponent's side of the screen is theirs.
function splitHands(hands) {
  if (!(mode === "coach" && el.oppSource.value === "live")) {
    const area = (lm) => {
      const xs = lm.map((p) => p.x);
      const ys = lm.map((p) => p.y);
      return (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
    };
    return { self: hands.reduce((a, b) => (!a || area(b) > area(a) ? b : a), null), other: null };
  }
  const oppLeft = el.oppSide.value === "left";
  let self = null;
  let other = null;
  for (const lm of hands) {
    const onLeft = 1 - lm[0].x < 0.5;
    if (onLeft === oppLeft) other ??= lm;
    else self ??= lm;
  }
  return { self, other };
}

function onFrame(res, capturedAt, seenAt) {
  const hands = res.landmarks ?? [];
  const { self, other } = splitHands(hands);
  liveLm = self;
  opp = other ? { e: (oppSmooth = smoothInto(oppSmooth, openness(other, el.video.videoWidth, el.video.videoHeight))) } : null;
  if (!other) oppSmooth = null;
  if (liveLm) {
    const e = openness(liveLm, el.video.videoWidth, el.video.videoHeight);
    smooth = smooth ? smooth.map((v, k) => v + 0.65 * (e[k] - v)) : e;
    live = { e: smooth, s: shapeOf(smooth) };
  } else {
    smooth = null;
    live = null;
  }
  lags.push(seenAt - capturedAt);
  if (lags.length > 300) lags.shift();
  drawHands(el.overlay, el.video, hands);
  updateLive();
  if (round?.phase !== "count") return;
  const t = capturedAt - round.shootAt;
  const now = seenAt - round.shootAt;
  if (round.mode === "coach") {
    if (round.source === "live") seeOpponent(round, t, now, opp ? opp.e : null);
    stepCoachUser(round, t, now);
  } else if (round.mode === "read" || round.mode === "react") step(round, t, now);
}

const smoothInto = (prev, e) => (prev ? prev.map((v, k) => v + 0.65 * (e[k] - v)) : e);

// t: when this frame was captured. now: when I got to see it. Both relative to Shoot.
// Only frames I saw before Shoot can shape my throw.
function step(r, t, now) {
  const f = { t: Math.round(t), lag: Math.round(now - t), e: live ? live.e.map(r3) : null, s: live ? { move: live.s.move, conf: r3(live.s.conf) } : null };
  if (el.record.checked && t >= Math.min(-2 * BEAT, (r.cueAt ?? 0) - 200)) f.thumb = thumbnail();
  r.frames.push(f);
  if (now < 0 && r.mode === "read") reader.see(r.read, t, live ? live.e : null);
  if (t >= LATE_TO) finish(r);
}

// Fires on the Shoot beat. My throw is fixed here from what I saw before this instant.
function shoot(r) {
  if (r !== round || r.shot) return;
  r.shot = true;
  const firedAt = Math.round(performance.now() - r.shootAt);
  r.decision = r.mode === "react"
    ? { play: r.machine, source: "cue", kind: "cue", firedAt, lastUsed: null }
    : { ...reader.decide(r.read, r.brain, !!liveLm), firedAt };
  const p = r.decision.play;
  el.machineThrow.innerHTML = `<span class="mt-lab">Machine</span>${ICONS[p]}<span class="mt-name">${NAMES[p]}</span>`;
  el.machineThrow.hidden = false;
  el.machineThrow.classList.remove("drop");
  void el.machineThrow.offsetWidth;
  el.machineThrow.classList.add("drop");
  setMine("reveal", r.decision);
}

function majority(frames, from, to) {
  const votes = [0, 0, 0];
  for (const f of frames) if (f.s && f.s.conf >= HOLD_T && f.t >= from && f.t <= to) votes[f.s.move]++;
  const top = Math.max(...votes);
  return top ? { move: votes.indexOf(top), n: top } : null;
}

function finish(r) {
  if (r !== round || r.phase !== "count") return;
  if (!r.shot) shoot(r);
  r.phase = "done";
  clearTimeout(r.safety);
  clearTimeout(r.shootTimer);
  el.call.textContent = "";

  r.final = majority(r.frames, READ_FROM, READ_TO)?.move ?? null;
  if (r.final !== null) {
    for (let k = r.frames.length - 1; k >= 0; k--) {
      const f = r.frames[k];
      if (f.t > READ_TO || !f.s) continue;
      if (f.t < (r.mode === "react" ? r.cueAt : ARM)) break;
      if (f.s.move === r.final && f.s.conf >= HOLD_T) r.formed = f.t;
      else break;
    }
    const late = majority(r.frames, READ_TO + 1, LATE_TO);
    if (late && late.move !== r.final && late.n >= 3) r.late = late.move;

    r.o = outcome(r.decision.play, r.final);
    const tally = tallies[r.mode];
    tally.played++;
    if (r.o > 0) tally.m++;
    else if (r.o < 0) tally.h++;
    else tally.d++;
    if (r.mode === "read") {
      const right = r.decision.guess === r.final;
      if (right) tally.right++;
      tally[r.decision.source][0] += right ? 1 : 0;
      tally[r.decision.source][1]++;
      reader.learn(r.read, r.final);
      brain.record(r.final, r.decision.play);
    } else {
      r.onset = onsetAfter(r.frames, r.cueAt);
      r.rt = r.onset == null ? null : r.onset - r.cueAt;
      if (r.rt != null && r.rt >= HUMAN_FASTEST) tally.rts.push(r.rt);
      // Beating my Scissors only takes staying in your fist, so those wins don't tighten the timing.
      r.free = r.machine === SCISSORS && r.o < 0;
      if (r.o < 0 && !r.free) {
        tally.best = tally.best == null ? r.lead : Math.min(tally.best, r.lead);
        lead = Math.max(LEAD_MIN, lead - STEP_WIN);
      } else if (r.o >= 0) lead = Math.min(LEAD_MAX, lead + STEP_MISS);
    }
  }

  lastRound = r;
  showBanner(r);
  renderStats();
  el.lastNo.textContent = `Round ${r.no}`;
  el.story.textContent = narrative(r);
  drawChart(r);
  if (el.record.checked) saveRecording(r);
  r.frames.forEach((f) => delete f.thumb);
  el.play.disabled = false;
  setModeLock(false);
  if (el.auto.checked) autoTimer = setTimeout(startRound, 2200);
}

function startRound() {
  if (!landmarker) return;
  if (busy) return notice({ recording: "Recording clips", testing: "Testing clips", preparing: "Getting opponents ready" }[busy], "Wait for it to finish, then play.");
  if (round && round.phase !== "done") return;
  if (mode === "coach") return void startCoachRound();
  clearTimeout(autoTimer);
  audio?.resume?.();
  const shootAt = performance.now() + LEAD_IN + 3 * BEAT;
  const r = { no: ++roundNo, mode, phase: "count", shootAt, frames: [], read: reader.round(), decision: null, shot: false };
  if (mode === "read") r.brain = brain.decide();
  else {
    r.machine = crypto.getRandomValues(new Uint32Array(1))[0] % 3;
    r.lead = lead;
    r.cueAt = -lead;
    r.cueType = el.cueType.value;
    if (r.cueType !== "flash") cueSound(beats(r.machine), shootAt - lead);
    if (r.cueType !== "sound") r.flashTimer = setTimeout(() => showCueFlash(beats(r.machine)), shootAt - lead - performance.now());
  }
  round = r;
  r.shootTimer = setTimeout(() => shoot(r), shootAt - performance.now());
  r.safety = setTimeout(() => finish(r), LEAD_IN + 3 * BEAT + LATE_TO + 600);
  for (let k = 3; k >= 0; k--) beep(shootAt - k * BEAT, k ? 523 : 1046, k ? 0.07 : 0.12);
  setMine("watch");
  el.banner.hidden = true;
  el.machineThrow.hidden = true;
  el.play.disabled = true;
  setModeLock(true);
}

function beep(at, freq, dur, always = false) {
  if (!audio || (!el.sound.checked && !always)) return;
  const t = audio.currentTime + Math.max(0, (at - performance.now()) / 1000);
  const osc = audio.createOscillator();
  const gain = audio.createGain();
  osc.type = "triangle";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  osc.connect(gain).connect(audio.destination);
  osc.start(t);
  osc.stop(t + dur + 0.02);
}

// Says the letter of the throw to make, at time `at` on the performance clock.
function cueSound(move, at) {
  if (!audio) return;
  if (voice[move]) {
    const src = audio.createBufferSource();
    src.buffer = voice[move];
    src.connect(audio.destination);
    src.start(audio.currentTime + Math.max(0, (at - performance.now()) / 1000));
    return;
  }
  // Lowercase, or the system voice says "capital P".
  setTimeout(() => window.speechSynthesis?.speak(new SpeechSynthesisUtterance(LETTERS[move].toLowerCase())), Math.max(0, at - performance.now()));
}

function showCueFlash(move) {
  el.cueFlash.textContent = LETTERS[move];
  el.cueFlash.hidden = false;
  setTimeout(() => (el.cueFlash.hidden = true), 300);
}

// When your fingers started moving after a cue at time `at`: the first of two frames in a row that left the pose you held then.
function onsetAfter(frames, at) {
  const before = frames.filter((f) => f.e && f.t <= at && f.t >= at - 150);
  const pose = before.length ? avg(before.map((f) => f.e)) : null;
  if (!pose) return null;
  const after = frames.filter((f) => f.e && f.t > at && f.t <= LATE_TO);
  const far = (f) => Math.hypot(...f.e.map((v, k) => v - pose[k])) >= MOVE_FROM_CUE;
  for (let i = 0; i < after.length; i++) if (far(after[i]) && (!after[i + 1] || far(after[i + 1]))) return after[i].t;
  return null;
}

function renderCues() {
  el.cues.innerHTML = LETTERS.map((letter, m) => `
    <div class="cue">
      <span class="letter">${letter}</span>
      <span><b>Throw ${NAMES[m]}</b><small>You'll hear or see "${letter}"</small></span>
      <button class="hear" type="button" data-cue="${m}">Hear it</button>
    </div>`).join("");
}

function setMode(next) {
  if (busy || (round && round.phase !== "done")) return;
  mode = next;
  document.body.dataset.mode = mode;
  if (mode !== "coach") setJudgeView(false);
  updateStage();
  el.banner.hidden = true;
  el.machineThrow.hidden = true;
  setMine("idle");
  renderStats();
}

function setModeLock(locked) {
  for (const input of [el.modeRead, el.modeReact, el.modeCoach, el.oppSource, el.oppSide]) input.disabled = locked;
  el.recClips.disabled = locked && busy !== "recording";
  el.testClips.disabled = locked;
}

// ---- "I coach you": read the opponent's hand and tip you off -------------------------------------------------

function sideNames() {
  if (!(mode === "coach" && el.oppSource.value === "live")) return ["Them", "You"];
  if (!document.body.classList.contains("judge")) return ["Them", "You"];
  return el.oppSide.value === "left" ? ["Left", "Right"] : ["Right", "Left"];
}

function updateStage() {
  const clipView = mode === "coach" && (el.oppSource.value !== "live" || busy === "testing") && busy !== "recording";
  el.stage.dataset.view = clipView ? "clip" : "cam";
  el.oppVideo.hidden = !clipView;
  el.oppOverlay.hidden = !clipView;
  const liveCoach = mode === "coach" && el.oppSource.value === "live" && busy !== "recording";
  el.sides.hidden = !liveCoach;
  el.oppSideWrap.hidden = !(mode === "coach" && el.oppSource.value === "live");
  const [l, r] = document.body.classList.contains("judge") ? ["Left player", "Right player"] : el.oppSide.value === "left" ? ["Opponent", "You"] : ["You", "Opponent"];
  el.sideL.textContent = l;
  el.sideR.textContent = r;
  renderStats();
}

function resetOpponent() {
  if (round && round.phase !== "done") return;
  tallies.coach = newOpponent();
  el.banner.hidden = true;
  renderStats();
  notice("New opponent", "Scores and their throw history are cleared.");
}

function setJudgeView(on) {
  el.judgeView.checked = on;
  document.body.classList.toggle("judge", on);
  el.title.textContent = on ? "Judge" : "Tell";
  updateStage();
}

// Tip-offs name the throw you should make, same as the cues in "You read me", so the practice carries over.
function tipOff(move, style, early = false) {
  if (style === "voice" || (style === "beep" && early)) return cueSound(move, performance.now());
  if (style === "beep") return beep(performance.now(), 1400, 0.06, true);
  if (style === "big") return showCueFlash(move);
  el.cueDot.textContent = LETTERS[move];
  el.cueDot.hidden = false;
  setTimeout(() => (el.cueDot.hidden = true), early ? 1600 : 700);
}

async function loadClips() {
  try {
    const res = await fetch("/api/clips", { cache: "no-store" });
    clips = res.ok ? await res.json() : [];
  } catch {
    clips = [];
  }
  el.clipCount.textContent = clips.length ? `${clips.length} saved` : "none yet";
}

// Explains on the stage, where you're looking, why a round can't start.
function notice(head, text) {
  el.banner.innerHTML = `<strong class="k-d">${head}</strong><span>${text}</span>`;
  el.banner.hidden = false;
}

function clipStatus(text) {
  el.clipStatus.textContent = text;
  el.clipStatus.hidden = !text;
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));
const random = () => crypto.getRandomValues(new Uint32Array(1))[0] / 2 ** 32;
const pick = (xs) => xs[Math.floor(random() * xs.length)];

// Plays a clip as the opponent. Shoot lands where it fell when the clip was recorded, and every frame goes through
// the hand tracker as the video plays, exactly as a live camera would deliver it.
async function loadClip(v, clip) {
  oppLandmarker ??= await createLandmarker(1);
  oppSmooth = null;
  v.src = `/clips/${clip.id}.webm`;
  await new Promise((resolve, reject) => {
    v.onloadeddata = resolve;
    v.onerror = () => reject(new Error(`Couldn't load ${clip.id}`));
  });
}

function readClipFrame(v, mediaMs, clip) {
  const lm = oppLandmarker.detectForVideo(v, performance.now()).landmarks?.[0] ?? null;
  drawHands(el.oppOverlay, v, lm ? [lm] : []);
  const e = lm ? (oppSmooth = smoothInto(oppSmooth, openness(lm, v.videoWidth, v.videoHeight))) : (oppSmooth = null);
  return { t: mediaMs - clip.shootOffset, e };
}

// For the blind test: reads every frame of the clip as a still image (extracted by the server), in order, so it gives
// the same answer whether or not this tab is in front. Each frame counts as seen one camera-lag after it was captured.
async function analyzeClip(clip, onFrame) {
  oppLandmarker ??= await createLandmarker(1);
  oppSmooth = null;
  const res = await fetch(`/api/clip-frames?id=${encodeURIComponent(clip.id)}`);
  if (!res.ok) throw new Error(await res.text());
  const frames = await res.json();
  const lag = median(lags) ?? 25;
  for (const f of frames) {
    const t = f.t - clip.shootOffset;
    if (t < ARM - 260) continue;
    if (t > LATE_TO + 60) break;
    // createImageBitmap decodes even when the tab is hidden; <img>.decode() waits for the tab to be shown.
    const bmp = await createImageBitmap(await (await fetch(f.url)).blob());
    const lm = oppLandmarker.detectForVideo(bmp, performance.now()).landmarks?.[0] ?? null;
    const e = lm ? (oppSmooth = smoothInto(oppSmooth, openness(lm, bmp.width, bmp.height))) : (oppSmooth = null);
    bmp.close();
    onFrame(t, t + lag, e);
  }
}

// rate and shift make a generated opponent move faster or slower and throw early (negative) or late (positive).
async function playClip(clip, { onStart, onFrame, rate = 1, shift = 0 }) {
  const v = el.oppVideo;
  await loadClip(v, clip);
  v.playbackRate = rate;
  return new Promise((resolve) => {
    let shootAt = null;
    let done = false;
    const end = () => {
      if (done) return;
      done = true;
      v.pause();
      resolve();
    };
    const tick = (now, meta) => {
      if (done) return;
      if (shootAt == null) {
        shootAt = (meta.expectedDisplayTime ?? now) + (clip.shootOffset - meta.mediaTime * 1000) / rate - shift;
        onStart(shootAt);
      }
      const frame = readClipFrame(v, meta.mediaTime * 1000, clip);
      const t = frame.t / rate + shift;
      onFrame(t, performance.now() - shootAt, frame.e);
      if (t >= LATE_TO + 60) end();
      else v.requestVideoFrameCallback(tick);
    };
    v.onended = end;
    v.currentTime = 0;
    v.requestVideoFrameCallback(tick);
    v.play().catch(end);
    // Browsers stop delivering video frames to background tabs.
    setTimeout(() => shootAt == null && end(), 1500);
  });
}

// Before the countdown: if they have a clear favorite lately, I tell you its counter while there's still plenty of time.
// With no clear favorite you hold your fist, which only loses to Paper, and I watch their hand for Paper.
function planFor() {
  const recent = tallies.coach.seen.slice(-PLAN_WINDOW);
  const counts = [0, 0, 0];
  for (const m of recent) counts[m]++;
  const top = Math.max(...counts);
  const favs = [0, 1, 2].filter((m) => counts[m] === top);
  const n = recent.length;
  const chance = n ? Math.min(1, 3 * binomTail(n, top, 1 / 3)) : 1;
  const fav = n >= PLAN_MIN && favs.length === 1 && chance < PLAN_P ? favs[0] : null;
  return { counts, n, fav, chance: Math.round(chance * 1000) / 1000, play: fav === null ? ROCK : beats(fav) };
}

// Chance of k or more hits in n tries at probability p: how often a "favorite" this strong shows up by pure luck.
function binomTail(n, k, p) {
  let total = 0;
  for (let i = k; i <= n; i++) {
    let c = 1;
    for (let j = 0; j < i; j++) c = (c * (n - j)) / (j + 1);
    total += c * p ** i * (1 - p) ** (n - i);
  }
  return total;
}

function beginCount(r, shootAt) {
  r.shootAt = shootAt;
  r.phase = "count";
  r.plan = planFor();
  if (r.plan.play !== ROCK) r.planTimer = setTimeout(() => tipOff(r.plan.play, r.cueStyle, true), shootAt + PLAN_AT - performance.now());
  for (let k = 3; k >= 0; k--) beep(shootAt - k * BEAT, k ? 523 : 1046, k ? 0.07 : 0.12);
  r.safety = setTimeout(() => finishCoach(r), shootAt - performance.now() + LATE_TO + 900);
}

function seeOpponent(r, t, now, e) {
  r.opp.frames.push({ t: Math.round(t), e: e ? e.map(r3) : null, s: e ? { move: shapeOf(e).move, conf: r3(shapeOf(e).conf) } : null });
  if (r.phase !== "count") return;
  const cue = coach.see(r.opp.state, t, now, e);
  if (cue && r.plan?.play === ROCK) tipOff(beats(cue.opp), r.cueStyle);
  if (r.source === "clip" && t >= LATE_TO) {
    r.oppDone = true;
    if (r.userDone) finishCoach(r);
  }
}

function stepCoachUser(r, t, now) {
  const f = { t: Math.round(t), lag: Math.round(now - t), e: live ? live.e.map(r3) : null, s: live ? { move: live.s.move, conf: r3(live.s.conf) } : null };
  if (el.record.checked && t >= -2 * BEAT) f.thumb = thumbnail();
  r.frames.push(f);
  if (t >= LATE_TO) {
    r.userDone = true;
    if (r.source === "live" || r.oppDone) finishCoach(r);
  }
}

// Where to freeze a Rock version of a clip: the clearest fist (in ms into the clip) before the fingers start opening.
function fistFrame(frames, clip) {
  const base = frames.filter((f) => f.e && f.t >= ARM - 200 && f.t < ARM).map((f) => f.e);
  if (!base.length) return null;
  const b = avg(base);
  const opens = frames.find((f) => f.e && f.t >= ARM && Math.hypot(...f.e.map((v, i) => v - b[i])) >= 0.2)?.t ?? READ_FROM;
  let best = null;
  for (const f of frames) {
    if (!f.s || f.t >= opens || f.s.move !== ROCK) continue;
    if (!best || f.s.conf > best.s.conf) best = f;
  }
  return best && best.s.conf >= 0.6 ? best.t + clip.shootOffset : null;
}

// Judges every clip that hasn't been judged yet, the same way I judge a live throw, and asks the server for a Rock
// version of each Paper and Scissors clip. Returns the clips grouped by throw.
async function preparePool() {
  await loadClips();
  const todo = clips.filter((c) => c.judged === undefined);
  for (const [i, clip] of todo.entries()) {
    notice("Getting opponents ready", `Reading clip ${i + 1} of ${todo.length}. This happens once.`);
    const frames = [];
    await analyzeClip(clip, (t, now, e) => frames.push({ t, e, s: e ? shapeOf(e) : null }));
    const judged = majority(frames, READ_FROM, READ_TO)?.move ?? null;
    await fetch("/api/clip-label", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: clip.id, judged, freezeAt: judged === PAPER || judged === SCISSORS ? fistFrame(frames, clip) : null }),
    });
  }
  if (todo.length) await loadClips();
  el.banner.hidden = true;
  const pool = [[], [], []];
  for (const c of clips) if (c.judged != null) pool[c.judged].push(c);
  return pool;
}

async function startCoachRound() {
  const source = el.oppSource.value;
  if (source !== "live" && !clips.length) await loadClips();
  if (source !== "live" && !clips.length) {
    return notice("No opponent clips yet", "Click Record 10 clips in the side panel first, or set Opponent to Live person on camera.");
  }
  let human = null;
  if (source === "random") {
    let pool;
    busy = "preparing";
    el.play.disabled = true;
    setModeLock(true);
    try {
      pool = await preparePool();
    } catch {
      pool = null;
    }
    busy = null;
    el.play.disabled = false;
    setModeLock(false);
    if (!pool) return notice("Couldn't read the clips", "Check that npm start is still running, then try again.");
    const missing = [0, 1, 2].filter((m) => !pool[m].length);
    if (missing.length) return notice(`No ${missing.map((m) => NAMES[m]).join(" or ")} clips`, "Record more clips, throwing each of Rock, Paper and Scissors at least once.");
    const thrown = Math.floor(random() * 3);
    human = { throw: thrown, clip: pick(pool[thrown]), shift: Math.round((random() * 2 - 1) * HUMAN_SHIFT), rate: 1 + (random() * 2 - 1) * HUMAN_SPEED };
  }
  clearTimeout(autoTimer);
  audio?.resume?.();
  const r = { no: ++roundNo, mode: "coach", source, phase: "load", frames: [], opp: { frames: [], state: coach.round() }, cueStyle: el.coachCue.value };
  round = r;
  setMine("watch");
  el.banner.hidden = true;
  el.machineThrow.hidden = true;
  el.play.disabled = true;
  setModeLock(true);
  if (source === "live") return beginCount(r, performance.now() + LEAD_IN + 3 * BEAT);

  let clip;
  if (human) {
    clip = human.clip;
    r.human = { throw: human.throw, shift: human.shift, rate: human.rate };
  } else {
    const originals = clips.filter((c) => !c.generated);
    const pool = originals.length > 1 ? originals.filter((c) => c.id !== lastClipId) : originals;
    clip = pick(pool);
  }
  lastClipId = clip.id;
  r.clip = clip.id;
  try {
    await playClip(clip, {
      rate: human?.rate ?? 1,
      shift: human?.shift ?? 0,
      onStart: (shootAt) => beginCount(r, shootAt),
      onFrame: (t, now, e) => seeOpponent(r, t, now, e),
    });
  } catch {
    notice("Couldn't load the clip", "Check that npm start is still running, then try again.");
  }
  r.oppDone = true;
  if (r.phase === "load") {
    notice("The clip didn't play", "Keep this tab in front and try again.");
    r.phase = "done";
    el.play.disabled = false;
    setModeLock(false);
  } else if (r.userDone) finishCoach(r);
}

function finishCoach(r) {
  if (r !== round || r.phase !== "count") return;
  r.phase = "done";
  clearTimeout(r.safety);
  el.call.textContent = "";
  const t = tallies.coach;
  const cues = r.opp.state.cues;
  r.oppFinal = majority(r.opp.frames, READ_FROM, READ_TO)?.move ?? null;
  r.final = majority(r.frames, READ_FROM, READ_TO)?.move ?? null;
  r.endedOn = majority(r.frames, READ_TO + 50, LATE_TO)?.move ?? r.final;
  const read = r.opp.state.read;
  clearTimeout(r.planTimer);
  if (r.oppFinal !== null) {
    t.seen.push(r.oppFinal);
    t.calls[1]++;
    if (read?.opp === r.oppFinal) {
      t.calls[0]++;
      t.firsts.push(read.at);
    }
  }
  if (r.oppFinal !== null && r.final !== null) {
    r.o = outcome(r.final, r.oppFinal);
    r.oLate = outcome(r.endedOn, r.oppFinal);
    t.played++;
    if (r.o > 0) t.h++;
    else if (r.o < 0) t.m++;
    else t.d++;
    t.onTime[1]++;
    t.late[1]++;
    if (r.o > 0) t.onTime[0]++;
    if (r.oLate > 0) t.late[0]++;
  }
  if (cues.length && r.plan.play === ROCK) {
    r.onset = onsetAfter(r.frames, cues[0].at);
    r.rt = r.onset == null ? null : r.onset - cues[0].at;
    if (r.rt != null && r.rt >= HUMAN_FASTEST) t.rts.push(r.rt);
  }

  lastRound = r;
  const [them, you] = sideNames();
  if (r.oppFinal === null || r.final === null) {
    el.banner.innerHTML = `<strong class="k-d">No result</strong><span>I couldn't see ${r.oppFinal === null ? (r.source === "clip" ? "the clip's" : "their") : "your"} throw just after Shoot.</span>`;
  } else {
    const head = r.o > 0 ? (you === "You" ? "You win" : `${you} wins`) : r.o < 0 ? (them === "Them" ? "They win" : `${them} wins`) : "Draw";
    const cls = r.o > 0 ? "k-h" : r.o < 0 ? "k-m" : "k-d";
    el.banner.innerHTML = `<strong class="${cls}">${head}</strong><span>${NAMES[r.final]} vs ${NAMES[r.oppFinal]}</span>`;
  }
  el.banner.hidden = false;
  setMine("coached", r);
  renderStats();
  el.lastNo.textContent = `Round ${r.no}`;
  el.story.textContent = coachNarrative(r);
  drawChart(r);
  if (el.record.checked) saveRecording(r);
  r.frames.forEach((f) => delete f.thumb);
  el.play.disabled = false;
  setModeLock(false);
  if (el.auto.checked) autoTimer = setTimeout(startRound, 2200);
}

function coachNarrative(r) {
  const read = r.opp.state.read;
  const who = r.source === "clip" ? "The clip" : "They";
  const p = r.plan;
  const spoke = p.play !== ROCK;
  const history = p.fav !== null ? `${who === "They" ? "They've" : "The clips have"} thrown ${NAMES[p.fav]} ${p.counts[p.fav]} of the last ${p.n} times` : null;
  let s = "";
  if (spoke) s += `${history}, so before the countdown I said "${LETTERS[p.play].toLowerCase()}". `;
  else {
    s += history ? `${history}, so you held your fist, which beats it, and I watched for Paper. `
      : `${p.n < PLAN_MIN ? "Too few throws to see a favorite yet" : "No favorite in their last throws stronger than chance"}, so you held your fist and I watched for Paper. `;
    if (!read) s += "I never got a clear read on their hand, so I stayed quiet.";
  }
  if (read && !spoke) {
    const how = read.why === "tell" ? "their fingers started moving toward it" : "their hand had formed it";
    s += read.opp === PAPER
      ? `At ${when(read.at)} I read Paper (${how}) and said "s". `
      : `At ${when(read.at)} I read ${NAMES[read.opp]} (${how}), so I stayed quiet: your fist ${read.opp === SCISSORS ? "beats" : "ties"} it. `;
  }
  if (read && spoke) s += `The camera read ${NAMES[read.opp]} ${when(read.at)}. `;
  if (r.oppFinal === null) return `${s} Then I couldn't see their throw, so this round doesn't count.`;
  if (r.human) s += `(This opponent picked ${NAMES[r.human.throw]} at random and threw ${Math.abs(r.human.shift)} ms ${r.human.shift < 0 ? "early" : "late"} at ${r.human.rate.toFixed(2)}x speed.) `;
  s += `${who} threw ${NAMES[r.oppFinal]}${read ? (read.opp === r.oppFinal ? ", so my read was right." : ", so I misread them.") : "."}`;
  if (r.final === null) return `${s} I couldn't see your throw.`;
  if (r.rt != null && !spoke) s += ` Your fingers started moving ${r.rt} ms after I said "s".`;
  s += ` On time you showed ${NAMES[r.final]}`;
  if (r.endedOn !== r.final) s += ` and ended on ${NAMES[r.endedOn]}, which ${r.oLate > 0 ? "would have won" : r.oLate < 0 ? "would have lost" : "would have tied"} if late throws counted`;
  return `${s}.`;
}

async function recordOneClip() {
  const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  const rec = new MediaRecorder(el.video.srcObject, { mimeType: types.find((m) => MediaRecorder.isTypeSupported(m)), videoBitsPerSecond: 2_500_000 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const startedAt = await new Promise((resolve) => {
    rec.onstart = () => resolve(performance.now());
    rec.start();
  });
  const shootAt = startedAt + CLIP_BEFORE;
  round = { mode: "rec", phase: "count", shootAt };
  for (let k = 3; k >= 0; k--) beep(shootAt - k * BEAT, k ? 523 : 1046, k ? 0.07 : 0.12);
  await wait(shootAt + CLIP_AFTER - performance.now());
  const stopped = new Promise((resolve) => (rec.onstop = resolve));
  rec.stop();
  await stopped;
  round = null;
  el.call.textContent = "";
  const res = await fetch(`/api/clip?shoot=${Math.round(shootAt - startedAt)}`, { method: "POST", body: new Blob(chunks, { type: "video/webm" }) });
  if (!res.ok) throw new Error(await res.text());
}

async function recordClips() {
  if (busy === "recording") {
    stopRecording = true;
    el.recClips.textContent = "Stopping after this clip…";
    return;
  }
  if (!landmarker || busy || (round && round.phase !== "done")) return;
  busy = "recording";
  stopRecording = false;
  ensureAudio();
  setModeLock(true);
  el.play.disabled = true;
  el.recClips.textContent = "Stop recording";
  updateStage();
  let saved = 0;
  try {
    for (let i = 1; i <= CLIPS_PER_BATCH && !stopRecording; i++) {
      clipStatus(`Clip ${i} of ${CLIPS_PER_BATCH}: pump and throw anything on Shoot. Mix it up.`);
      await recordOneClip();
      saved++;
      await wait(900);
    }
    clipStatus(`Saved ${saved} clip${saved === 1 ? "" : "s"} to clips/.`);
  } catch {
    clipStatus("Couldn't save the clip. Run the app with npm start so the local server can write to clips/.");
  }
  busy = null;
  round = null;
  el.recClips.textContent = `Record ${CLIPS_PER_BATCH} clips`;
  setModeLock(false);
  el.play.disabled = false;
  await loadClips();
  updateStage();
}

// Plays every clip through the coach with nobody playing, to measure how often and how early it reads them.
async function testClips() {
  if (!landmarker || busy || (round && round.phase !== "done")) return;
  await loadClips();
  if (!clips.length) return clipStatus("Record some opponent clips first.");
  busy = "testing";
  setModeLock(true);
  el.play.disabled = true;
  updateStage();
  const rows = [];
  for (const [i, clip] of clips.entries()) {
    clipStatus(`Testing clip ${i + 1} of ${clips.length}…`);
    const state = coach.round();
    const frames = [];
    try {
      await analyzeClip(clip, (t, now, e) => {
        frames.push({ t: Math.round(t), e: e ? e.map(r3) : null, s: e ? shapeOf(e) : null });
        coach.see(state, t, now, e);
      });
    } catch {
      continue;
    }
    const final = majority(frames, READ_FROM, READ_TO)?.move ?? null;
    const read = state.read;
    const trace = frames.map((f) => [f.t, f.e]);
    rows.push({ clip: clip.id, final, frames: frames.length, handFrames: frames.filter((f) => f.e).length, read, right: final !== null && read?.opp === final, trace });
  }
  const judged = rows.filter((r) => r.final !== null);
  const right = judged.filter((r) => r.right).length;
  const papers = judged.filter((r) => r.final === PAPER);
  const saidS = papers.filter((r) => r.read?.opp === PAPER).map((r) => r.read.at);
  const falseS = judged.filter((r) => r.final !== PAPER && r.read?.opp === PAPER).length;
  const mid = median(saidS);
  el.testResult.innerHTML = [
    `<b>Read ${right} of ${judged.length} clips right. Against Paper I said "s" on ${saidS.length} of ${papers.length}, median ${mid == null ? "–" : when(mid)}, ${saidS.filter((x) => x <= -100).length} of them 100+ ms early. False "s": ${falseS}.</b>`,
    ...rows.map((r, i) => `Clip ${i + 1}: threw ${NAMES[r.final] ?? "?"} · ${r.read ? `I read ${NAMES[r.read.opp]} ${when(r.read.at)}${r.right ? "" : " (wrong)"}` : "no read"}`),
  ].join("<br>");
  clipStatus(`Tested ${rows.length} clips.`);
  try {
    await fetch("/api/record", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session: `coach-test-${SESSION}`, round: 1, log: { settings: { ...SETTINGS }, rows } }),
    });
  } catch {}
  busy = null;
  setModeLock(false);
  el.play.disabled = false;
  updateStage();
}

function countdown() {
  if (round?.phase === "count") {
    const t = performance.now() - round.shootAt;
    if (t >= 0 && !round.shot && (round.mode === "read" || round.mode === "react")) shoot(round);
    const word = t < -3 * BEAT ? "Ready" : t < -2 * BEAT ? "Rock" : t < -BEAT ? "Paper" : t < 0 ? "Scissors" : "Shoot!";
    if (el.call.textContent !== word) {
      el.call.textContent = word;
      el.call.dataset.shoot = String(t >= 0);
      el.call.classList.remove("hit");
      void el.call.offsetWidth;
      el.call.classList.add("hit");
    }
  }
  requestAnimationFrame(countdown);
}

function narrative(r) {
  if (r.mode === "react") return reactNarrative(r);
  const d = r.decision;
  const play = NAMES[d.play];
  const from = d.held !== null ? `from ${NAMES[d.held]} ` : "";
  const record = (move) => {
    const n = d.seen ? d.seen[move] : 0;
    const total = d.seen ? d.seen[0] + d.seen[1] + d.seen[2] : 0;
    return `${n} of ${total} times`;
  };
  let s;
  if (d.kind === "tell") {
    s = `At ${when(d.seenAt)}, your hand started moving ${from}toward ${NAMES[d.guess]}, so I threw ${play} on Shoot.`;
  } else if (d.kind === "fake") {
    s = `At ${when(d.seenAt)}, your hand started moving ${from}toward ${NAMES[d.tellGuess]}. But when you've shown me that before Shoot, you've ended on ${NAMES[d.guess]} ${record(d.guess)}, so I threw ${play}.`;
  } else if (d.kind === "habit") {
    s = `You held ${NAMES[d.held]} still through the countdown. When you've done that before, you ended on ${NAMES[d.guess]} ${record(d.guess)}, so I threw ${play} on Shoot.`;
  } else if (d.kind === "bait") {
    s = `You held ${NAMES[d.held]} still through the countdown, which looks like bait. If you switch, you'll throw ${NAMES[(d.held + 1) % 3]} or ${NAMES[(d.held + 2) % 3]}. ${play} beats one and ties the other, so I threw ${play} on Shoot.`;
  } else {
    s = `${d.kind === "fist" ? "Your hand was a still fist when I had to throw, so the camera had nothing to go on." : "I couldn't see your hand before Shoot."} I went on your history instead. ${d.why}`;
  }
  if (r.final === null) return `${s} Then I couldn't make out your throw between ${READ_FROM} and ${READ_TO} ms after Shoot, so this round doesn't count.`;
  s += r.final === d.guess ? ` You threw ${NAMES[r.final]}, so I read you right.` : ` You threw ${NAMES[r.final]}, so I misread you.`;
  if (r.formed != null && r.final !== ROCK) s += ` Your hand finished forming ${when(r.formed)}.`;
  if (r.late != null) s += ` You switched to ${NAMES[r.late]} after that, which was too late to count.`;
  return s;
}

function reactNarrative(r) {
  const how = { sound: "spoken letter", flash: "letter on screen", both: "spoken and on-screen letter" }[r.cueType];
  let s = `${r.lead} ms before Shoot, a ${how} told you to throw ${NAMES[beats(r.machine)]} against my ${NAMES[r.machine]}.`;
  if (r.final === null) return `${s} Then I couldn't make out your throw between ${READ_FROM} and ${READ_TO} ms after Shoot, so this round doesn't count.`;
  if (r.rt == null && r.final === ROCK) s += " You stayed in your fist.";
  else if (r.rt == null) s += " I didn't see your fingers move after the cue.";
  else if (r.rt < HUMAN_FASTEST) s += ` Your fingers moved ${r.rt} ms after it. That's faster than a human can react, so you were already moving or guessing.`;
  else s += ` Your fingers started moving ${r.rt} ms after it.`;
  if (r.formed != null && r.final !== ROCK) s += ` Your ${NAMES[r.final]} finished forming ${when(r.formed)}.`;
  s += r.o < 0 ? ` You beat my ${NAMES[r.machine]}.` : r.o === 0 ? ` That ties my ${NAMES[r.machine]}.` : ` That loses to my ${NAMES[r.machine]}.`;
  if (r.free) s += " Staying in a fist needs no reaction, so this round doesn't change the timing.";
  s += ` Next cue: ${lead} ms before Shoot.`;
  return s;
}

function showBanner(r) {
  if (r.final === null) {
    el.banner.innerHTML = `<strong class="k-d">No throw</strong><span>I couldn't see a clear throw just after Shoot. Keep your hand in frame and throw on the beat.</span>`;
  } else {
    const [cls, head] = { 1: ["k-m", "I win"], "-1": ["k-h", "You win"], 0: ["k-d", "Draw"] }[r.o];
    el.banner.innerHTML = `<strong class="${cls}">${head}</strong><span>My ${NAMES[r.decision.play]} vs your ${NAMES[r.final]}</span>`;
  }
  el.banner.hidden = false;
}

function setMine(state, d) {
  el.mine.dataset.state = state;
  const parts = {
    idle: [EYE, "Ready", landmarker ? "Press Space or Play round" : "Turn on the camera to play"],
    watch: mode === "coach" ? [EYE, "Watching them", "I'll tip you off the moment I read their hand"]
      : mode === "react" ? [EYE, "Listen", "My cue comes before Shoot"] : [EYE, "Watching", "I throw on Shoot, same as you"],
    coached: [d && d.oppFinal != null ? ICONS[d.oppFinal] : EYE, d && d.oppFinal != null ? `They: ${NAMES[d.oppFinal]}` : "No read",
      d?.opp ? (d.opp.state.read ? `I read ${NAMES[d.opp.state.read.opp]} ${when(d.opp.state.read.at)}` : "I didn't get a read in time") : ""],
    reveal: [d && ICONS[d.play], d && NAMES[d.play], d && `Thrown on Shoot · ${{ camera: "camera read", habit: "held-shape read", pattern: "pattern read", cue: "cued early" }[d.source]}`],
  }[state];
  el.mine.innerHTML = `<span class="mine-ic">${parts[0]}</span><span class="mine-word">${parts[1]}</span><span class="mine-sub">${parts[2]}</span>`;
}

const bars = FINGER_NAMES.map((name, k) => {
  const row = document.createElement("div");
  row.className = "finger";
  row.innerHTML = `<span>${name}</span><span class="track"><span class="fill" style="background:${COLOR.fingers[k]}"></span><span class="base" hidden></span></span><span class="val">–</span>`;
  el.fingers.append(row);
  return { fill: row.querySelector(".fill"), base: row.querySelector(".base"), val: row.querySelector(".val") };
});

function updateLive() {
  const base = round?.phase === "count" && round.read?.base.length ? avg(round.read.base) : null;
  bars.forEach((b, k) => {
    const v = live ? live.e[k] : 0;
    b.fill.style.width = `${v * 100}%`;
    b.val.textContent = live ? Math.round(v * 100) : "–";
    b.base.hidden = !base;
    if (base) b.base.style.left = `calc(${base[k] * 100}% - 1px)`;
  });
  el.seen.dataset.on = String(!!live);
  el.seen.textContent = !live ? "No hand in view" : live.s.conf < HOLD_T ? "I see your hand moving" : `I see ${live.s.move === ROCK ? "a fist" : NAMES[live.s.move]}`;
}

function drawHands(canvas, video, hands) {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  for (const lm of hands) drawHand(canvas, lm, false, lm === liveLm ? live?.e : null);
}

function drawHand(canvas, lm, mirror = false, e = null) {
  const g = canvas.getContext("2d");
  if (!lm) return;
  const W = canvas.width;
  const H = canvas.height;
  const P = (i) => [(mirror ? 1 - lm[i].x : lm[i].x) * W, lm[i].y * H];
  const lw = Math.max(1.5, W / 200);
  g.lineCap = "round";
  g.lineWidth = lw;
  g.strokeStyle = COLOR.ink;
  g.globalAlpha = 0.5;
  for (const [a, b] of PALM) seg(g, P(a), P(b));
  g.globalAlpha = 1;
  FINGERS.forEach((f, k) => {
    g.strokeStyle = COLOR.fingers[k];
    g.lineWidth = lw * (1 + (e ? e[k] : 0.4));
    for (let j = 0; j < 3; j++) seg(g, P(f[j]), P(f[j + 1]));
  });
  g.fillStyle = COLOR.ink;
  for (let i = 0; i < 21; i++) {
    const [x, y] = P(i);
    g.beginPath();
    g.arc(x, y, lw * 0.9, 0, Math.PI * 2);
    g.fill();
  }
}

function seg(g, a, b) {
  g.beginPath();
  g.moveTo(a[0], a[1]);
  g.lineTo(b[0], b[1]);
  g.stroke();
}

function drawChart(r) {
  const c = el.chart;
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth;
  const H = c.clientHeight;
  c.width = Math.round(W * dpr);
  c.height = Math.round(H * dpr);
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  g.font = `11px ${tok("--mono")}`;
  if (!r) return;

  const frames = r.mode === "coach" ? r.opp.frames : r.frames;
  const t0 = Math.min(-3 * BEAT, r.cueAt ?? 0) - 120;
  const t1 = LATE_TO;
  const L = 34, R = 12, T = 22, B = 24;
  const X = (t) => L + ((clamp(t, t0, t1) - t0) / (t1 - t0)) * (W - L - R);
  const Y = (v) => T + (1 - v) * (H - T - B);

  g.fillStyle = COLOR.panel2;
  g.fillRect(X(READ_FROM), T, X(READ_TO) - X(READ_FROM), H - T - B);

  g.strokeStyle = COLOR.line;
  g.lineWidth = 1;
  g.fillStyle = COLOR.muted;
  g.textAlign = "right";
  for (const v of [0, 0.5, 1]) {
    seg(g, [L, Y(v)], [W - R, Y(v)]);
    g.fillText(String(v * 100), L - 6, Y(v) + 4);
  }
  g.textAlign = "center";
  ["Rock", "Paper", "Scissors", "Shoot"].forEach((word, k) => {
    const x = X((k - 3) * BEAT);
    g.strokeStyle = COLOR.line;
    seg(g, [x, T], [x, H - B]);
    g.fillStyle = k === 3 ? COLOR.machine : COLOR.muted;
    g.fillText(word, x, 14);
  });
  g.fillStyle = COLOR.muted;
  g.fillText("your throw", (X(READ_FROM) + X(READ_TO)) / 2, H - 8);

  g.lineWidth = 2;
  g.lineJoin = "round";
  FINGERS.forEach((_, k) => {
    g.strokeStyle = COLOR.fingers[k];
    g.beginPath();
    let pen = false;
    for (const f of frames) {
      if (f.t < t0 || f.t > t1 || !f.e) {
        pen = false;
        continue;
      }
      const x = X(f.t);
      const y = Y(f.e[k]);
      if (pen) g.lineTo(x, y);
      else g.moveTo(x, y);
      pen = true;
    }
    g.stroke();
  });

  g.setLineDash([4, 4]);
  if (r.formed != null && r.final !== ROCK) {
    g.strokeStyle = COLOR.human;
    seg(g, [X(r.formed), T], [X(r.formed), H - B]);
  }
  const marks = r.mode === "coach" ? r.opp.state.cues.map((c) => c.at) : [r.mode === "react" ? r.cueAt : r.decision?.lastUsed];
  g.strokeStyle = COLOR.machine;
  for (const mark of marks) if (mark != null) seg(g, [X(mark), T], [X(mark), H - B]);
  g.setLineDash([]);
  if (r.onset != null) {
    g.strokeStyle = COLOR.human;
    g.lineWidth = 1.5;
    seg(g, [X(r.onset), T], [X(r.onset), H - B]);
  }
  if (r.mode !== "coach") {
    g.strokeStyle = COLOR.machine;
    g.lineWidth = 2.5;
    seg(g, [X(0), T], [X(0), H - B]);
  }
}

function renderStats() {
  const tally = tallies[mode];
  el.sM.textContent = tally.m;
  el.sD.textContent = tally.d;
  el.sH.textContent = tally.h;
  const [them, you] = sideNames();
  el.sMLab.textContent = mode === "coach" ? them : "Machine";
  el.sHLab.textContent = mode === "coach" ? you : "You";
  if (mode === "coach") {
    const share = ([w, n]) => (n ? `${w} of ${n}` : "–");
    const first = median(tally.firsts);
    const rt = median(tally.rts);
    el.stCalls.textContent = share(tally.calls);
    el.stFirst.textContent = first == null ? "–" : when(first);
    el.stOnTime.textContent = share(tally.onTime);
    el.stLate.textContent = share(tally.late);
    el.stReact.textContent = rt == null ? "–" : `${rt} ms`;
    const c = [0, 0, 0];
    for (const m of tally.seen) c[m]++;
    el.stSeen.textContent = tally.seen.length ? `R ${c[0]} · P ${c[1]} · S ${c[2]}` : "–";
    return;
  }
  if (mode === "react") {
    const rt = median(tally.rts);
    el.stLead.textContent = `${lead} ms before Shoot`;
    el.stBeat.textContent = tally.played ? `${tally.h} of ${tally.played}` : "–";
    el.stRt.textContent = rt == null ? "–" : `${rt} ms`;
    el.stBest.textContent = tally.best == null ? "–" : `${tally.best} ms`;
    return;
  }
  if (!tally.played) return;
  el.stReads.textContent = `${tally.right} / ${tally.played}`;
  const fmt = ([right, n]) => (n ? `${right} right of ${n}` : "none yet");
  el.stCam.textContent = fmt(tally.camera);
  el.stHabit.textContent = fmt(tally.habit);
  el.stPat.textContent = fmt(tally.pattern);
  const lag = median(lags);
  el.stLag.textContent = lag == null ? "–" : `${Math.round(lag)} ms`;
}

// Recording: a contact sheet of camera frames around Shoot plus the raw finger data, saved by server.mjs.
function thumbnail() {
  const vw = el.video.videoWidth;
  const vh = el.video.videoHeight;
  const c = document.createElement("canvas");
  c.width = THUMB_W;
  c.height = Math.round((THUMB_W * vh) / vw);
  const g = c.getContext("2d");
  g.save();
  g.translate(c.width, 0);
  g.scale(-1, 1);
  g.drawImage(el.video, 0, 0, c.width, c.height);
  g.restore();
  drawHand(c, liveLm, true, live?.e);
  return c;
}

function contactSheet(r) {
  const withThumbs = r.frames.filter((f) => f.thumb);
  if (!withThumbs.length) return null;
  const tw = withThumbs[0].thumb.width;
  const th = withThumbs[0].thumb.height;
  const cols = 4;
  const times = r.mode === "react"
    ? SHEET_TIMES.map((_, i) => Math.round(r.cueAt - 120 + ((LATE_TO - r.cueAt + 120) * i) / (SHEET_TIMES.length - 1)))
    : SHEET_TIMES;
  const rows = Math.ceil(times.length / cols);
  const head = 64;
  const c = document.createElement("canvas");
  c.width = cols * tw;
  c.height = head + rows * th;
  const g = c.getContext("2d");
  g.fillStyle = COLOR.bg;
  g.fillRect(0, 0, c.width, c.height);

  const d = r.decision ?? {};
  const result = r.final == null || r.o == null ? "no throw read"
    : r.mode === "coach" ? `you threw ${NAMES[r.final]} vs their ${NAMES[r.oppFinal]} · ${{ 1: "you win", "-1": "they win", 0: "draw" }[r.o]}`
    : `you threw ${NAMES[r.final]} · ${{ 1: "I win", "-1": "you win", 0: "draw" }[r.o]}`;
  const source = r.mode === "coach"
    ? `coach · their throw ${NAMES[r.oppFinal] ?? "–"} · my cues ${r.opp.state.cues.map((c) => `${NAMES[c.opp]}@${c.at}`).join(", ") || "none"} · you moved at ${r.onset ?? "–"} ms · ended on ${NAMES[r.endedOn] ?? "–"}`
    : r.mode === "react"
    ? `react mode · cue ${NAMES[r.machine]} (${r.cueType}) at ${r.cueAt} ms · you moved at ${r.onset ?? "–"} ms · reaction ${r.rt ?? "–"} ms`
    : d.source === "camera" ? `camera ${d.kind}: ${NAMES[d.guess]} seen at ${d.seenAt} ms` : `pattern read (${d.kind}): expected ${NAMES[d.guess]}`;
  g.fillStyle = COLOR.ink;
  g.font = `bold 20px ${tok("--body")}`;
  g.fillText(r.mode === "coach" ? `Round ${r.no} · coaching · ${result}` : `Round ${r.no} · I threw ${NAMES[d.play]} on Shoot · ${result}`, 12, 26);
  g.fillStyle = COLOR.muted;
  g.font = `14px ${tok("--mono")}`;
  g.fillText(`${source} · last frame used ${d.lastUsed} ms · formed ${r.formed ?? "–"} ms · lag ${median(r.frames.map((f) => f.lag))} ms`, 12, 50);

  const used = new Set();
  const markAt = r.mode === "coach" ? r.opp.state.cues[0]?.at : r.cueAt;
  const markT = r.mode !== "read" && markAt != null ? withThumbs.reduce((a, f) => (Math.abs(f.t - markAt) < Math.abs(a.t - markAt) ? f : a)).t : d.lastUsed;
  times.forEach((target, i) => {
    let best = null;
    for (const f of withThumbs) if (!used.has(f) && (!best || Math.abs(f.t - target) < Math.abs(best.t - target))) best = f;
    if (!best) return;
    used.add(best);
    const x = (i % cols) * tw;
    const y = head + Math.floor(i / cols) * th;
    g.drawImage(best.thumb, x, y);
    let border = null;
    if (best.t === markT) border = COLOR.machine;
    else if (best.t === r.formed) border = COLOR.human;
    if (border) {
      g.strokeStyle = border;
      g.lineWidth = 4;
      g.strokeRect(x + 2, y + 2, tw - 4, th - 4);
    }
    g.fillStyle = "rgba(0,0,0,.65)";
    g.fillRect(x, y, tw, 20);
    g.fillRect(x, y + th - 20, tw, 20);
    g.font = `12px ${tok("--mono")}`;
    g.fillStyle = best.t < 0 ? COLOR.ink : COLOR.human;
    g.fillText(`t ${best.t > 0 ? "+" : ""}${best.t} ms`, x + 6, y + 14);
    g.fillStyle = COLOR.ink;
    const e = best.e ? best.e.map((v) => Math.round(v * 100)).join(" ") : "no hand";
    const s = best.s ? `${NAMES[best.s.move]} ${best.s.conf.toFixed(2)}` : "";
    g.fillText(`I M R P ${e}  ${s}`, x + 6, y + th - 6);
  });
  return c.toDataURL("image/jpeg", 0.82);
}

async function saveRecording(r) {
  const log = {
    round: r.no,
    session: SESSION,
    mode: r.mode,
    settings: { BEAT, READ_FROM, READ_TO, LATE_TO, ...SETTINGS },
    final: r.final ?? null,
    formed: r.formed ?? null,
    late: r.late ?? null,
    outcome: r.o ?? null,
    frames: r.frames.map(({ thumb, ...f }) => f),
  };
  if (r.mode === "read") {
    Object.assign(log, {
      baseline: r.read.base.length ? avg(r.read.base).map(r3) : null,
      held: r.read.held ?? null,
      tell: r.read.tell,
      habits: structuredClone(reader.habits),
      brain: { assumed: r.brain.assumed, play: r.brain.play, label: r.brain.label },
      decision: r.decision,
    });
  } else if (r.mode === "react") {
    Object.assign(log, { decision: r.decision, cue: { move: r.machine, lead: r.lead, at: r.cueAt, type: r.cueType, onset: r.onset ?? null, rt: r.rt ?? null } });
  } else {
    Object.assign(log, {
      source: r.source,
      clip: r.clip ?? null,
      plan: r.plan,
      human: r.human ?? null,
      tipOff: r.cueStyle,
      opponent: { final: r.oppFinal, held: r.opp.state.held ?? null, frames: r.opp.frames },
      cues: r.opp.state.cues,
      endedOn: r.endedOn ?? null,
      outcomeIfLate: r.oLate ?? null,
      onset: r.onset ?? null,
      rt: r.rt ?? null,
    });
  }
  showRec(`Saving round ${r.no}…`);
  try {
    const res = await fetch("/api/record", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session: SESSION, round: r.no, log, sheet: contactSheet(r) }),
    });
    if (!res.ok) throw new Error(await res.text());
    const { saved } = await res.json();
    showRec(`Saved ${saved}.json and .jpg`);
  } catch {
    showRec("Couldn't save the recording. Run the app with npm start so the local server can write to recordings/.", true);
  }
}

function showRec(text, bad = false) {
  el.recStatus.textContent = text;
  el.recStatus.dataset.bad = String(bad);
  el.recStatus.hidden = false;
}

el.startCam.addEventListener("click", startCamera);
el.modeRead.addEventListener("change", () => setMode("read"));
el.modeReact.addEventListener("change", () => setMode("react"));
el.modeCoach.addEventListener("change", () => setMode("coach"));
el.oppSource.addEventListener("change", updateStage);
el.oppSide.addEventListener("change", updateStage);
el.judgeView.addEventListener("change", () => setJudgeView(el.judgeView.checked));
el.recClips.addEventListener("click", recordClips);
el.newOpp.addEventListener("click", resetOpponent);
el.testClips.addEventListener("click", testClips);
el.cues.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-cue]");
  if (!b) return;
  await ensureAudio();
  cueSound(+b.dataset.cue, performance.now() + 20);
});
el.play.addEventListener("click", startRound);
el.auto.addEventListener("change", () => {
  if (el.auto.checked && landmarker && round?.phase !== "count") startRound();
  if (!el.auto.checked) clearTimeout(autoTimer);
});
el.record.addEventListener("change", () => {
  if (el.record.checked) showRec(`Recording on. Each round saves to recordings/${SESSION}/`);
  else el.recStatus.hidden = true;
});
document.addEventListener("keydown", (e) => {
  if (mode === "coach" && !e.repeat && !e.metaKey && !e.ctrlKey && !e.target.closest?.("input, select")) {
    if (e.code === "KeyJ") return void setJudgeView(!document.body.classList.contains("judge"));
    if (e.code === "KeyN") return void resetOpponent();
  }
  if (e.code !== "Space" || e.repeat || e.target.closest?.("button, input, select")) return;
  e.preventDefault();
  startRound();
});
let resizeTimer = 0;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => drawChart(lastRound), 120);
});

renderCues();
loadClips();
setMine("idle");
drawChart(null);
countdown();
