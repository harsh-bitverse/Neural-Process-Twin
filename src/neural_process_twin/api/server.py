"""Command-line entrypoint to launch the Neural Process Twin API server."""

from __future__ import annotations

import argparse

import uvicorn

from neural_process_twin.api.app import create_app


def main() -> None:
    """Parse CLI options and start the Uvicorn ASGI server."""
    parser = argparse.ArgumentParser(description="Neural Process Twin API Server")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument(
        "--static-dir", default="frontend/dist", help="Path to compiled frontend static files"
    )
    args = parser.parse_args()

    app = create_app(static_dir=args.static_dir)
    print(f"Starting Neural Process Twin API on http://{args.host}:{args.port}")
    print(f"Interactive API documentation: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
