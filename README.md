# Tell

Rock paper scissors with a webcam that reads your hand before you finish throwing. Write-up: [Trying to never lose at rock paper scissors](https://akhilbod.github.io/writing/tell.html).

Three modes:

- **I read you.** The computer watches your hand and throws on Shoot, using only camera frames it saw before Shoot. When your hand gives nothing away, it guesses from your habits.
- **You read me.** The computer says a letter (r, p or s) before Shoot telling you what beats its throw. The warning gets later each time you win, so it finds the shortest one you can react to.
- **I coach you.** The computer watches an opponent's hand and tells you what to throw. Hold a fist by default; it says "s" when it sees Paper coming, and before the countdown it names the counter to the opponent's favorite throw if they have one that's stronger than chance. Practice against recorded clips, a generated random opponent, or a live person, with a judge view that only shows the countdown and score.

`sealed-throw.html` is the first version: no camera, just the habit model, with a SHA-256 commitment so you can check it locked its throw before you picked.

## Run it

```bash
npm start
```

Then open http://localhost:5199 in Chrome and allow the camera. Hand tracking is MediaPipe's hand landmarker, loaded from a CDN and run in the browser; video never leaves the tab.

The clip tools in "I coach you" save to `clips/` and `recordings/` through the local server and need `ffmpeg` installed for frame extraction and generated Rock clips.

## Files

- `public/reader.js` reads tells from finger movement and decides the computer's throw. No DOM, so it runs in Node too.
- `public/coach.js` decides when to tip you off about an opponent.
- `public/brain.js` is the habit model: 12 predictors, each tried three ways, scored every round.
- `public/app.js` is the camera, countdown, cues and UI.
- `server.mjs` serves the app and saves recordings and clips. No dependencies.
- `npm run replay` reruns recorded rounds through the current `reader.js`.
- `npm run coach-replay` reruns the coach over the finger traces saved by the clip test, with optional overrides like `MOVE_T=0.3`.
