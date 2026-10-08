import argparse
import os

from . import ChatError, Mica
from .server import serve


def main():
    parser = argparse.ArgumentParser(description="Mica — Local conversations, powered by MLX. Optional local HTTP API.")
    parser.add_argument("--model", required=True, help="Cached repository ID or local native MLX model directory")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--online", action="store_true", help="Allow metadata checks/downloads; default is offline")
    parser.add_argument("--voice", action="store_true", help="Enable local speech endpoints (requires voice extra and prepared assets)")
    parser.add_argument("--allow-origin", action="append", default=[], help="Allow an exact loopback browser origin, e.g. http://localhost:5173")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        with Mica(args.model, offline=not args.online) as backend:
            print(f"Mica API listening at http://127.0.0.1:{args.port}")
            from ..config import Settings
            serve(backend, port=args.port, token=os.environ.get("MICA_API_TOKEN"),
                  audio=args.voice, audio_settings=Settings.load().voice if args.voice else None,
                  allowed_origins=args.allow_origin)
    except KeyboardInterrupt:
        pass
    except (ChatError, OSError, RuntimeError, ImportError, ValueError) as e:
        parser.exit(1, f"Mica API error: {e}\n")


if __name__ == "__main__":
    main()
