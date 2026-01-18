import argparse

from aiohttp import web

from proxy.server import create_app, UPSTREAM_URL


def main() -> None:
    parser = argparse.ArgumentParser(description="API format proxy server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()

    print(f"Proxy: {args.host}:{args.port} -> {UPSTREAM_URL}")
    print("Routes:")
    print("  POST /v1/responses  (Responses API -> Chat Completions)")
    print("  POST /v1/messages   (Anthropic API -> Chat Completions)")
    print("  GET  /health")

    web.run_app(create_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
