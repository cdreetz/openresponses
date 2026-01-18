import argparse
from aiohttp import web
from .server import create_app, UPSTREAM_URL

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--host", default="0.0.0.0")
args = parser.parse_args()

print(f"Proxy listening on {args.host}:{args.port} -> {UPSTREAM_URL}")
web.run_app(create_app(), host=args.host, port=args.port)
