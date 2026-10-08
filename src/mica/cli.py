from dataclasses import asdict
from functools import wraps
from enum import Enum
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shlex
import time

import typer
from rich.console import Console
from rich.table import Table

from .config import ChatError, Settings, check_platform, home, offline_environment
from .conversation import Conversation
from .inference import Engine
from .models import PRESETS, inspect_remote, resolve

app = typer.Typer(no_args_is_help=True, help="Mica — Local conversations, powered by MLX.")
console = Console()


def explain(e):
    text = str(e)
    name = type(e).__name__
    if isinstance(e, MemoryError) or any(s in text.lower() for s in ("out of memory", "allocation", "insufficient memory")):
        return "Memory allocation failed. Close other applications, choose the 3B model, or reduce context/output limits. No backend fallback is used."
    if "Gated" in name or "401" in text or "403" in text:
        return "Model access denied. Accept its license on Hugging Face and authenticate using `uv run hf auth login`, or set HF_TOKEN."
    if isinstance(e, OSError) and e.errno == 28:
        return "Disk is full. Free space and rerun the same download; Hugging Face reuses its cache."
    if any(s in name for s in ("HTTP", "Connection", "Timeout", "Download", "RepositoryNotFound")):
        return f"Model metadata/download failed: {text}. Check repository ID, network and authentication; rerun to resume cached downloads or use --offline with a downloaded model."
    return text or name


def guarded(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            console.print("\nInterrupted. Retry the command to resume any cached download.")
            raise typer.Exit(130)
        except Exception as e:
            console.print(f"Error: {explain(e)}", style="red", markup=False, soft_wrap=True)
            raise typer.Exit(1)
    return run


def setup(offline):
    if offline:
        offline_environment()
    check_platform()


def available(offline):
    found = []
    for identity, label in PRESETS:
        try:
            if offline:
                spec = resolve(identity, offline=True)
            else:
                spec = inspect_remote(identity)
            found.append((spec, label))
        except Exception as e:
            console.print(f"Omitted {identity}: {explain(e)}", markup=False)
    return found


def picker(offline):
    found = available(offline)
    for i, (spec, label) in enumerate(found, 1):
        console.print(f"{i}. {label}: {spec.identity} ({spec.size / 1024**3:.2f} GiB weights)", markup=False)
    answer = typer.prompt("Choose number, repository ID, or local directory", default="1" if found else "")
    if answer.isdigit():
        i = int(answer) - 1
        if not 0 <= i < len(found):
            raise ChatError("Invalid model selection.")
        return found[i][0].identity
    return answer


@app.command()
@guarded
def models(offline: bool = typer.Option(False, help="Inspect cached presets only; no network requests.")):
    """List presets only after validating native quantization and architecture."""
    setup(offline)
    table = Table("Model", "Choice", "Weight GiB", "Context", "Packed layers", "Revision")
    for spec, label in available(offline):
        table.add_row(spec.identity, label, f"{spec.size / 1024**3:.2f}", str(spec.context), str(spec.quantized_layers), spec.revision or "local")
    console.print(table)


@app.command()
@guarded
def download(model: str = typer.Option(...), offline: bool = False):
    """Download only the selected, validated native MLX 4-bit model."""
    setup(offline)
    spec = resolve(model, offline, announce=lambda s: console.print(s, markup=False))
    from .assets import stage_model
    if not Path(model).expanduser().is_dir():
        spec.path = stage_model(spec.identity, spec.path, spec.revision)
    console.print(f"Validated {spec.identity}\n{spec.path}", markup=False)


@app.command("voice-download")
@guarded
def voice_download(offline: bool = False):
    """Cache configured speech models/voices and validate phonemizer assets."""
    setup(offline)
    from .voice import VoiceSettings, LocalSpeech, resolve_assets
    settings = VoiceSettings(**Settings.load().voice)
    from .assets import stage_speech_resources
    stage_speech_resources()
    paths = resolve_assets(settings, offline, prepare=True)
    speech = LocalSpeech(settings, paths)
    try:
        for _ in speech.synthesize("Speech assets are ready."):
            pass
        console.print("Speech models, configured voice and phonemizer validated for offline use.")
    finally:
        speech.close()


@app.command()
@guarded
def doctor():
    """Check native Python, Metal, dependencies, and configured memory headroom."""
    check_platform()
    import mlx.core as mx
    console.print(f"macOS {platform.mac_ver()[0]}, {platform.machine()}, Python {platform.python_version()}")
    for package in ("mlx", "mlx-lm", "huggingface-hub", "typer", "rich", "prompt-toolkit"):
        console.print(f"{package}: {version(package)}")
    console.print(mx.device_info())
    console.print(f"Settings: {home() / 'config.json'}")
    console.print(asdict(Settings.load()))
    console.print("MLX memory limit is a guideline. Other apps, KV cache and swap affect usable memory; no system settings are changed.")


class IOMode(str, Enum):
    text = "text"
    voice = "voice"


HELP = """/mode [text|voice] — show/switch input and output mode
/help — commands
/model [repository-or-directory] — pick/switch, releasing the previous model
/new — clear conversation and KV cache
/save [path] — save JSON session (default: Mica sessions/last.json)
/load [path] — restore model, settings and messages
/system [instructions] — replace system instructions (no argument clears them)
/stats — most recent generation measurements
/exit — exit (EOF also exits)
Text: Enter sends; Esc+Enter inserts a newline.
Voice: Enter starts recording; Enter again stops/submits. Slash commands work before recording.
Ctrl+C cancels the active operation or clears input.
Interrupted turns are saved with incomplete=true and removed as a pair before the next turn.
"""


@app.command()
@guarded
def chat(
    model: str | None = typer.Option(None, help="Repository ID or local directory; otherwise last selection/picker."),
    offline: bool = False,
    system: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    context_budget: int | None = None,
    max_tokens: int | None = None,
    memory_gb: float | None = None,
    io_mode: IOMode = typer.Option(IOMode.text, help="Text or local half-duplex voice I/O."),
):
    """Stream a local conversation; never use remote inference."""
    setup(offline)
    console.print("Mica — Local conversations, powered by MLX.")
    settings = Settings.load()
    for key, value in {"system": system, "temperature": temperature, "top_p": top_p, "seed": seed, "context_budget": context_budget, "max_tokens": max_tokens, "memory_gb": memory_gb}.items():
        if value is not None:
            setattr(settings, key, value)
    settings.validate()
    identity = model or settings.last_model or picker(offline)
    engine = Engine()
    voice_io = None
    active_mode = "text"

    def set_mode(mode):
        nonlocal voice_io, active_mode
        if mode not in ("text", "voice"):
            raise ChatError("Use /mode text or /mode voice.")
        if mode == active_mode:
            return
        if mode == "voice":
            from .voice import VoiceSettings, create_voice
            candidate = create_voice(VoiceSettings(**settings.voice), offline)
            voice_io = candidate
        else:
            if voice_io is not None:
                voice_io.close()
            voice_io = None
        active_mode = mode

    def switch(identity, selected_settings):
        spec = resolve(identity, offline, announce=lambda s: console.print(s, markup=False))
        engine.load(spec, selected_settings)
        selected_settings.last_model = spec.identity
        selected_settings.save()
        console.print(f"Loaded {spec.identity} in {engine.load_seconds:.2f}s. Context budget: {min(spec.context, selected_settings.context_budget)}.", markup=False)
        return spec.identity

    try:
        identity = switch(identity, settings)
        conversation = Conversation(identity, settings)
        try:
            set_mode(io_mode.value)
        except KeyboardInterrupt:
            console.print("Voice setup cancelled; using text mode.")
        except Exception as e:
            console.print(f"Voice setup failed: {explain(e)}. Using text mode.", style="red", markup=False)
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        home().mkdir(parents=True, exist_ok=True, mode=0o700)
        history_path = home() / "terminal-history"
        history_path.touch(mode=0o600, exist_ok=True)
        keys = KeyBindings()

        @keys.add("enter")
        def send(event):
            event.current_buffer.validate_and_handle()

        @keys.add("escape", "enter")
        def newline(event):
            event.current_buffer.insert_text("\n")

        session = PromptSession(history=FileHistory(str(history_path)), multiline=True, key_bindings=keys)
        console.print(HELP, markup=False)
        default_session = home() / "sessions" / "last.json"
        def wait_for_stop(full):
            # prompt_toolkit owns stdin; timeout also submits at the duration cap.
            import asyncio
            async def wait():
                task = asyncio.create_task(session.prompt_async("Recording> "))
                try:
                    while not task.done() and not full.is_set():
                        await asyncio.sleep(0.05)
                    if task.done():
                        task.result()
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            asyncio.run(wait())

        while True:
            from_voice = False
            try:
                text = session.prompt("Voice> Enter records; /command> " if active_mode == "voice" else "You> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                continue
            if active_mode == "voice" and not text.strip():
                try:
                    text = voice_io.listen(wait_for_stop, lambda state: console.print(state, markup=False))
                    if not text.strip():
                        console.print("No speech detected.")
                        continue
                    from_voice = True
                    console.print(f"You (voice)> {text}", markup=False, highlight=False)
                except KeyboardInterrupt:
                    console.print("Voice input cancelled.")
                    continue
                except Exception as e:
                    console.print(explain(e), style="red", markup=False)
                    continue
            if not text.strip():
                continue
            if active_mode == "voice" and not from_voice and not text.startswith("/"):
                console.print("Press Enter to record, or use /mode text for typed messages.")
                continue
            if text.startswith("/") and not from_voice:
                command, _, argument = text.partition(" ")
                try:
                    if command == "/mode":
                        if argument.strip():
                            set_mode(argument.strip())
                        console.print(f"I/O mode: {active_mode}", markup=False)
                    elif command == "/exit":
                        break
                    elif command == "/help":
                        console.print(HELP, markup=False)
                    elif command == "/stats":
                        console.print(engine.stats)
                    elif command == "/new":
                        conversation.messages.clear()
                        engine.invalidate()
                    elif command == "/system":
                        settings.system = argument
                        settings.save()
                        engine.invalidate()
                    elif command in ("/save", "/load"):
                        path = Path(shlex.split(argument)[0]).expanduser() if argument else default_session
                        if command == "/save":
                            conversation.save(path)
                            console.print(f"Saved {path}", markup=False)
                        else:
                            restored = Conversation.load(path)
                            switch(restored.model, restored.settings)
                            previous_mode = active_mode
                            set_mode("text")
                            conversation, settings = restored, restored.settings
                            engine.invalidate()
                            set_mode(previous_mode)
                    elif command == "/model":
                        selected = argument.strip() or picker(offline)
                        identity = switch(selected, settings)
                        conversation = Conversation(identity, settings)
                    else:
                        console.print("Unknown command; use /help.")
                except KeyboardInterrupt:
                    console.print("Command cancelled.")
                except Exception as e:
                    console.print(explain(e), style="red", markup=False)
                if engine.model is None:
                    console.print("No model remains loaded after switching failed. Restart chat to select a model.")
                    break
                continue
            try:
                was_incomplete = bool(conversation.messages and conversation.messages[-1].get("incomplete"))
                tokens, trimmed = conversation.prepare(text, engine.tokenizer, engine.spec.context)
                if trimmed:
                    engine.invalidate()
                if was_incomplete:
                    console.print("Removed the previously interrupted turn from the active context.")
            except ChatError as e:
                console.print(str(e), style="red", markup=False)
                continue
            if from_voice:
                console.print("Generating…", markup=False)
            console.print("Assistant> ", end="")
            output = []
            incomplete = False
            stream = engine.stream(tokens, settings)
            try:
                for piece in stream:
                    output.append(piece)
                    console.print(piece, end="", markup=False, highlight=False, soft_wrap=True)
            except KeyboardInterrupt:
                incomplete = True
                console.print("\n[cancelled]", markup=False)
            except Exception as e:
                incomplete = True
                console.print(f"\n{explain(e)}", style="red", markup=False)
            finally:
                stream.close()
                conversation.finish("".join(output), incomplete)
                if incomplete:
                    engine.invalidate()
                console.print()
            if from_voice and not incomplete:
                try:
                    voice_io.speak("".join(output), lambda state: console.print(state, markup=False))
                except KeyboardInterrupt:
                    console.print("Speech cancelled; completed text retained.")
                except Exception as e:
                    console.print(f"Speech failed: {explain(e)}. Completed text retained.", style="red", markup=False)
    finally:
        try:
            if voice_io is not None:
                voice_io.close()
        finally:
            engine.close()


@app.command()
@guarded
def benchmark(
    model: str = typer.Option(...), offline: bool = False,
    runs: int = typer.Option(3, min=1, max=10), max_tokens: int = typer.Option(128, min=2, max=512),
    output: Path | None = None,
):
    """Measure cold model load separately from warm, repeatable generation."""
    setup(offline)
    settings = Settings(temperature=0.0, top_p=1.0, seed=42, max_tokens=max_tokens)
    spec = resolve(model, offline, announce=lambda s: console.print(s, markup=False))
    engine = Engine()
    records = []
    prompt = "Explain in six short sentences why the sky appears blue during the day."
    try:
        engine.load(spec, settings)
        load_seconds = engine.load_seconds
        for i in range(runs + 1):
            # Uncached prompt each run: measure warm weights, not a prefix-cache hit.
            engine.invalidate()
            state = Conversation(spec.identity, settings)
            tokens, _ = state.prepare(prompt, engine.tokenizer, spec.context)
            for _ in engine.stream(tokens, settings):
                pass
            records.append({"phase": "first_generation" if i == 0 else "warm_generation", "run": i, **engine.stats})
        import mlx.core as mx
        result = {"model": spec.identity, "revision": spec.revision, "device": mx.device_info(), "versions": {p: version(p) for p in ("mlx", "mlx-lm", "huggingface-hub")}, "settings": asdict(settings), "prompt": prompt, "cold_load_seconds": load_seconds, "cold_load_peak_mlx_bytes": engine.load_peak_mlx_bytes, "cold_load_note": "First load in this process; OS filesystem caches were not flushed. Validation/download excluded.", "runs": records}
        console.print_json(json.dumps(result))
        if output:
            from .config import atomic_json
            atomic_json(output, result)
    finally:
        engine.close()
