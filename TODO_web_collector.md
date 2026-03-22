# Web HCI Collector — TODO

Issues identified from a code review of `src/collectors/web_hci_collector/`.

---

## Video Recording (Memory / OOM)

- [ ] **Replace FileReader.readAsDataURL with fetch POST of raw Blob** — `ondataavailable` currently base64-encodes each chunk via `FileReader`, inflating memory ~33%. Multiple readers run concurrently with no backpressure. Peak memory for a 30-min session: ~2.5 GB. Fix: POST the `Blob` directly to the existing stub at `POST /api/session/{id}/save-video`, passing metadata as query params. The browser streams the Blob without decoding it into JS heap (~125 KB per chunk instead of ~462 KB).
  - `participate.js:1363-1384` — replace `ondataavailable` with `fetch()` call
  - `participate.js:1386-1391` — simplify `onstop` (no `pendingVideoReaders` needed)
  - `participate.js:38` — remove `pendingVideoReaders` variable
  - `server.py:486-493` — implement the `save-video` POST endpoint (read `request.body()`, append to `recording.webm`)
- [ ] **Add video chunk size limit on server** — `server.py:1093-1107` decodes unbounded base64 payloads with no max size check. Add a per-chunk size limit (e.g. 2 MB) and per-session disk quota.
- [ ] **Add FileReader error handler** — `participate.js:1363-1384` has no `reader.onerror`, so if `readAsDataURL` fails (OOM), `pendingVideoReaders` never decrements and `sendVideoComplete` never fires.
- [ ] **Fix stopScreenRecording race** — `participate.js:1425-1429` sets `mediaRecorder = null` before tracks fully stop. The async version (`1431-1444`) uses an arbitrary 200ms `setTimeout` that may not be enough.

---

## Camera & Media Streams

- [ ] **Prevent multiple camera streams** — `participate.js:942-1040`: if WebGazer is inactive, `initFaceMeshFallback()` opens a second camera stream. This doubles CPU/GPU usage and can cause camera lock on macOS.
- [ ] **Show user-facing error on webcam failure** — `participate.js:791-895`: if camera permission is denied, error is only logged to console. No modal/banner shown. Participant unknowingly collects mouse-position data labeled as "gaze."

---

## Memory Leaks

- [ ] **Remove event listeners on cleanup** — `participate.js:1156-1216`: mouse/keyboard listeners added with anonymous functions and never removed. Store references and call `removeEventListener` in `cleanupOnUnload`.
- [ ] **Clear calibration DOM nodes on restart** — `participate.js:315-487`: `calPointElements` array grows on each recalibration (9 → 18 → 27 nodes). `restartCalibration()` should clear the array and remove old DOM nodes.
- [ ] **Bound dashboard timeline arrays** — `dashboard.js:91-104`: `timelineData` arrays (`gaze`, `mouse`, `faceMesh`, etc.) grow unboundedly. For a 30-min session: ~100K entries, ~138 MB. Use circular buffers or window to last N minutes for live view.

---

## CPU / GPU Usage

- [ ] **Cancel RAF loops on session end** — `dashboard.js:258-273`: three `requestAnimationFrame` loops (`renderFaceMesh`, `renderTrails`, `timelineLoop`) run forever, even after the session ends. Store RAF IDs and call `cancelAnimationFrame` in cleanup.
- [ ] **Pause rendering when tab is hidden** — `dashboard.js`: add `document.addEventListener('visibilitychange')` to pause RAF/setInterval when the tab is backgrounded.
- [ ] **Debounce MutationObserver** — `participate.js:1305-1318`: observer watches entire iframe DOM with `subtree: true`. On dynamic SPAs, this fires on every DOM change. Batch with `requestAnimationFrame`.
- [ ] **Throttle iframe resize handler** — `participate.js:1242`: `window.addEventListener('resize')` recalculates iframe rect on every event with no debounce.

---

## Error Handling

- [ ] **Log FaceMesh processing errors** — `participate.js:993-998`: `faceMesh.send()` errors are silently swallowed with `catch (e) { /* ignore */ }`. At minimum log to console.
- [ ] **Validate WebSocket JSON payloads** — `server.py:970-1001`: incoming WS messages have no schema validation (no pydantic models). Malformed payloads can raise unhandled exceptions.
- [ ] **Use structured logging** — `server.py`, `data_processor.py`: currently uses `print()` throughout. Replace with Python `logging` module for levels, formatting, and log rotation.

---

## Security

- [ ] **Add rate limiting** — no rate limiting on WebSocket or POST endpoints. A malicious client can flood the server with video chunks or event data.
- [ ] **Add authentication** — no auth on any endpoint. Assumes trusted LAN. Add at least token-based auth for non-local deployments.
- [ ] **Validate postMessage origin** — `participate.js:1232` and iframe-tracker use `'*'` as target origin. Specify the actual origin to prevent cross-origin message injection.

---

## Resource Management

- [ ] **Add session TTL / max buffer size** — `data_processor.py`: `DataBuffer` keeps all session data in memory with no TTL or cap. Long sessions can exhaust RAM. Clear buffer after periodic flush or enforce a max size.
- [ ] **Guard against double-start** — `participate.js:275-307`: `beginStudy()` has no `if (isCollecting) return` guard. Double-clicking can initialize WebGazer twice and create duplicate listeners.
- [ ] **Fix video finalization race condition** — `server.py:893-911, 455-484`: both WebSocket disconnect and beacon POST can call `_finalize_video()`. Guarded by `_finalized_sessions` set (max 200 entries), but overflows are possible at scale.

---

## Data Integrity

- [ ] **Fix emotion timestamp alignment** — `data_processor.py:236-244`: emotion detector uses `time.time()` (Unix seconds) while gaze uses `performance.now()` (ms). Misalignment affects cross-stream correlation analysis.
- [ ] **Warn on CSV bad lines** — `data_processor.py:227`: `pd.read_csv(..., on_bad_lines='skip')` silently drops malformed rows. Log a warning with the count of dropped rows.
