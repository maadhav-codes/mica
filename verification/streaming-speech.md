# Sentence speech streaming — 9 October 2026

Voice responses now enqueue completed sentence chunks during the existing LLM streaming loop. The CLI thread retains generation, terminal output and Conversation completion; a speech thread consumes a bounded four-sentence queue, using the existing isolated local Kokoro process and blocking playback adapter. Speech starts before the answer finishes. The final text fragment is submitted only after successful generation. Markdown filtering applies only to speech; raw terminal output and saved text are preserved. Fenced code is buffered until the closing fence and replaced with a spoken omission notice.

Queue backpressure bounds pending sentences and can slow generation when playback falls behind. Punctuation/newline sentence boundaries are heuristic. Concurrent MLX processes share GPU resources, so early audio does not imply higher total generation throughput.

Generation interruption/error cancels speech, aborts device buffers, terminates speech inference and joins the speech thread. Queued/unsubmitted fragments are discarded; already played sentences cannot be recalled. The existing incomplete-turn history/cache policy remains. Speech errors stop further audio while allowing text generation to complete. Interrupting speech after text completion preserves completed text. Recording never overlaps generation/playback, and mode switches occur after speech finishes or is cancelled.

Validation:

- Full automated suite: **132 passed in 10.93s**. Focused voice tests passed again after adding speech-thread startup failure fallback.
- Added fake-adapter tests demonstrate audio before generation ends, sentence/token boundaries, fenced-code filtering, playback order, queue bounds, cancelled inference result discard, continued text production after speech failure, and CLI interrupted-turn persistence.
- Real offline overlap check on Python 3.12.13 / Apple M3 using staged Llama 3.2 3B and Kokoro bf16: first synthesized sentence audio at **1.932 seconds**, LLM generation completed at **18.848 seconds**, 11 synthesized audio chunks. This one short run verifies overlap; it is not a sustained performance benchmark.
- The check used an in-memory playback sink, so no microphone or speaker device was opened. Real audible playback and physical device cancellation still require manual verification.
- Repeat with `uv run --locked --extra voice python scripts/voice_stream_smoke.py` after preparing the default assets.
