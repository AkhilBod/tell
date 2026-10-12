// Pattern engine from Sealed Throw. When the camera has no tell before Shoot, this picks the throw from your history.
export const NAMES = ["Rock", "Paper", "Scissors"];
export const beats = (m) => (m + 1) % 3;
export const outcome = (a, u) => (a === u ? 0 : a === beats(u) ? 1 : -1);

export function createEngine(opts = {}) {
  const DECAY = opts.decay ?? 0.85;
  const BETA = opts.beta ?? 1;
  const MODE = opts.mode ?? "best";
  const WINDOW = 40;
  const DEPTH = 300;
  const U = [];
  const A = [];
  const N = (m) => NAMES[m];
  const ofTimes = (c, t) => (t === 1 ? "the one time so far" : `${c} of ${t} times`);

  function tally(n, from, match, value) {
    const c = [0, 0, 0];
    const last = [-1, -1, -1];
    for (let i = Math.max(0, from); i < n; i++) {
      if (!match(i)) continue;
      const v = value(i);
      c[v]++;
      last[v] = i;
    }
    const t = c[0] + c[1] + c[2];
    if (!t) return null;
    let m = 0;
    for (let k = 1; k < 3; k++) if (c[k] > c[m] || (c[k] === c[m] && last[k] > last[m])) m = k;
    return { m, c: c[m], t };
  }

  function findRun(seq, minL, maxL) {
    const n = seq.length;
    const floor = Math.max(0, n - DEPTH);
    for (let L = Math.min(maxL, n - 1); L >= minL; L--) {
      for (let j = n - L - 1; j >= floor; j--) {
        let k = 0;
        while (k < L && seq[j + k] === seq[n - L + k]) k++;
        if (k === L) return { L, next: seq[j + L] };
      }
    }
    return null;
  }

  const PREDICTORS = [
    { id: "fav", name: "Favorite", run(n) {
      if (n < 1) return null;
      const from = Math.max(0, n - 12);
      const r = tally(n, from, () => true, (i) => U[i]);
      const k = n - from;
      return { move: r.m, why: k === 1 ? `Your only throw so far was ${N(r.m)}.` : `${N(r.m)} is your most common throw lately (${r.c} of your last ${k}).` };
    } },
    { id: "repeat", name: "Repeat", run(n) {
      if (n < 1) return null;
      return { move: U[n - 1], why: `You threw ${N(U[n - 1])} last round, and this read bets on a repeat.` };
    } },
    { id: "copy", name: "Copy me", run(n) {
      if (n < 1) return null;
      return { move: A[n - 1], why: `I threw ${N(A[n - 1])} last round, and this read bets you'll copy it.` };
    } },
    { id: "after1", name: "After one", run(n) {
      if (n < 2) return null;
      const x = U[n - 1];
      const r = tally(n, n - WINDOW, (i) => i >= 1 && U[i - 1] === x, (i) => U[i]);
      return r && { move: r.m, why: `After you throw ${N(x)}, you've followed with ${N(r.m)} ${ofTimes(r.c, r.t)}.` };
    } },
    { id: "after2", name: "After two", run(n) {
      if (n < 3) return null;
      const x = U[n - 2];
      const y = U[n - 1];
      const r = tally(n, n - WINDOW, (i) => i >= 2 && U[i - 2] === x && U[i - 1] === y, (i) => U[i]);
      return r && { move: r.m, why: `After ${N(x)} then ${N(y)}, you've followed with ${N(r.m)} ${ofTimes(r.c, r.t)}.` };
    } },
    { id: "matchup", name: "After the matchup", run(n) {
      if (n < 2) return null;
      const x = U[n - 1];
      const y = A[n - 1];
      const r = tally(n, n - WINDOW, (i) => i >= 1 && U[i - 1] === x && A[i - 1] === y, (i) => U[i]);
      return r && { move: r.m, why: `When your ${N(x)} meets my ${N(y)}, you've gone ${N(r.m)} next ${ofTimes(r.c, r.t)}.` };
    } },
    { id: "react", name: "Win-stay, lose-shift", run(n) {
      if (n < 2) return null;
      const res = (i) => -outcome(A[i], U[i]);
      const o = res(n - 1);
      const r = tally(n, n - WINDOW, (i) => i >= 1 && res(i - 1) === o, (i) => (U[i] - U[i - 1] + 3) % 3);
      if (!r) return null;
      const move = (U[n - 1] + r.m) % 3;
      const after = o === 1 ? "a win" : o === -1 ? "a loss" : "a draw";
      const did = ["kept the same throw", "switched to the throw that beats your last one", "switched to the throw your last one beats"][r.m];
      return { move, why: `After ${after}, you've ${did} ${ofTimes(r.c, r.t)}. That points to ${N(move)}.` };
    } },
    { id: "replay", name: "Your replay", run(n) {
      const r = findRun(U, 2, 6);
      return r && { move: r.next, why: `Your last ${r.L} throws (${U.slice(n - r.L).map(N).join(", ")}) match an earlier run, and that time you threw ${N(r.next)} next.` };
    } },
    { id: "gameReplay", name: "Game replay", run() {
      const r = findRun(U.map((u, i) => u * 3 + A[i]), 2, 6);
      if (!r) return null;
      const move = Math.floor(r.next / 3);
      return { move, why: `Our last ${r.L} rounds match an earlier stretch of the game, and that time you threw ${N(move)} next.` };
    } },
    { id: "myFav", name: "Counter my favorite", run(n) {
      if (n < 1) return null;
      const r = tally(n, n - 12, () => true, (i) => A[i]);
      const move = beats(r.m);
      return { move, why: `I've thrown ${N(r.m)} most lately. If you've noticed, you'd counter with ${N(move)}.` };
    } },
    { id: "myHabit", name: "Counter my habit", run(n) {
      if (n < 2) return null;
      const x = A[n - 1];
      const r = tally(n, n - WINDOW, (i) => i >= 1 && A[i - 1] === x, (i) => A[i]);
      if (!r) return null;
      const move = beats(r.m);
      return { move, why: `After I throw ${N(x)}, I've usually gone ${N(r.m)} next. If you've noticed, you'd counter with ${N(move)}.` };
    } },
    { id: "myReplay", name: "Counter my replay", run() {
      const r = findRun(A, 2, 6);
      if (!r) return null;
      const move = beats(r.next);
      return { move, why: `My last ${r.L} throws match an earlier run that I followed with ${N(r.next)}. If you've spotted it, you'd counter with ${N(move)}.` };
    } },
  ];

  const VARIANTS = ["straight", "shifted up", "shifted down"];
  const strats = [];
  for (const p of PREDICTORS) {
    for (let r = 0; r < 3; r++) strats.push({ key: `${p.id}:${r}`, p, r, name: p.name, variant: VARIANTS[r], score: 0, hits: [], assumed: null, out: null });
  }
  let pending = null;

  function pickBest(list, rng) {
    let top = -Infinity;
    for (const s of list) top = Math.max(top, s.score);
    const ties = list.filter((s) => s.score >= top - 1e-9);
    return ties[Math.floor(rng() * ties.length)];
  }

  function explain(s) {
    let t = s.out.why;
    if (s.r === 1) t += ` But lately you've been throwing whatever beats that, which is ${N(s.assumed)}.`;
    if (s.r === 2) t += ` But lately you've been throwing whatever that beats, which is ${N(s.assumed)}.`;
    if (s.hits.length) t += ` This read has called ${s.hits.filter(Boolean).length} of the last ${s.hits.length} rounds.`;
    return `${t} So I expected ${N(s.assumed)} and threw ${N(beats(s.assumed))}.`;
  }

  function pack(assumed, dist, why, s, kind) {
    const label = s ? `${s.name}, ${s.variant}` : { opening: "Opening book", gut: "Gut read", random: "Random throw" }[kind];
    return { play: beats(assumed), assumed, dist, why, kind, key: s ? s.key : null, label };
  }

  function decide(rng = Math.random) {
    const n = U.length;
    const outs = {};
    for (const p of PREDICTORS) outs[p.id] = p.run(n);
    const live = [];
    for (const s of strats) {
      s.out = outs[s.p.id];
      s.assumed = s.out ? (s.out.move + s.r) % 3 : null;
      if (s.out) live.push(s);
    }
    pending = live.map((s) => [s, s.assumed]);

    const w = [0, 0, 0];
    for (const s of live) w[s.assumed] += Math.exp(BETA * s.score);
    const tot = w[0] + w[1] + w[2];
    const dist = tot ? w.map((x) => x / tot) : [1 / 3, 1 / 3, 1 / 3];

    if (n === 0) {
      return pack(0, [0.4, 0.3, 0.3], "First round, so I have nothing on you yet. Rock is a popular opener, so I expected Rock and threw Paper.", null, "opening");
    }

    const best = live.length ? pickBest(live, rng) : null;
    if (best && best.score > 0) {
      let s = best;
      if (MODE === "vote") {
        const top = Math.max(...dist);
        const ms = [0, 1, 2].filter((m) => dist[m] >= top - 1e-9);
        const m = ms[Math.floor(rng() * ms.length)];
        s = pickBest(live.filter((x) => x.assumed === m), rng);
      }
      return pack(s.assumed, dist, explain(s), s, "read");
    }

    const lastU = U[n - 1];
    const res = -outcome(A[n - 1], lastU);
    if (n < 4 && res === 1) {
      return pack(lastU, dist, `You won last round, and winners tend to stick with the throw that worked. So I expected ${N(lastU)} again and threw ${N(beats(lastU))}.`, null, "gut");
    }
    if (n < 4 && res === -1) {
      const m = beats(lastU);
      return pack(m, dist, `You lost last round, and people who lose tend to switch to the throw that beats their last one. So I expected ${N(m)} and threw ${N(beats(m))}.`, null, "gut");
    }
    const m = Math.floor(rng() * 3);
    return pack(m, dist, `None of my reads has a winning record right now, so I threw ${N(beats(m))} at random. Nobody can beat random on average, which makes it a safe throw while I look for your next pattern.`, null, "random");
  }

  function record(u, a) {
    const touched = new Set();
    for (const [s, m] of pending || []) {
      s.score = s.score * DECAY + outcome(beats(m), u);
      s.hits.push(m === u);
      if (s.hits.length > 10) s.hits.shift();
      touched.add(s);
    }
    for (const s of strats) if (!touched.has(s)) s.score *= DECAY;
    pending = null;
    U.push(u);
    A.push(a);
  }

  function board(k) {
    return strats
      .slice()
      .sort((x, y) => y.score - x.score)
      .slice(0, k)
      .map((s) => ({ key: s.key, name: s.name, variant: s.variant, score: s.score, h: s.hits.filter(Boolean).length, t: s.hits.length }));
  }

  return { decide, record, board, size: strats.length, max: 1 / (1 - DECAY) };
}
