"""mcp/ — a tiny client for the Model Context Protocol.

What is MCP? It's a standard way for an LLM app (the *host*/client) to discover
and call tools that live in a separate *server* process — a filesystem server, a
GitHub server, a database server, whatever. Before MCP, every app hand-wrote glue
for every integration. MCP fixes that the way LSP fixed editor/language plugins:
one protocol, so any compliant client can talk to any compliant server. Write a
tool server once and every MCP-aware app can use it.

The wire format is **JSON-RPC 2.0** — a decades-old, dead-simple RPC convention:
each request is `{"jsonrpc":"2.0","id":<n>,"method":<str>,"params":{...}}` and the
reply is `{"jsonrpc":"2.0","id":<n>,"result":{...}}` (or `"error":{...}`). The
`id` is what pairs a reply to its request.

The transport here is **stdio**: we launch the server as a subprocess and speak
newline-delimited JSON over its stdin/stdout. (MCP also defines an HTTP/SSE
transport for remote servers; the message shapes are identical, only the pipe
changes.) One handshake, three verbs — that's the whole surface a client needs:
  - `initialize`  — announce protocol version + capabilities, learn the server's.
  - `tools/list`  — ask what tools exist (names, descriptions, JSON-Schema args).
  - `tools/call`  — invoke one by name with arguments; get back content blocks.

Results are **content blocks**: a list like [{"type":"text","text":"..."}], not a
bare string — the same idea as Anthropic's message content, so an image or other
block type can ride the same channel. We flatten text blocks for convenience.

Note the shape mapping: MCP's tool schema uses `inputSchema`, while Musaeus's own
tool registry uses `parameters` (see CONTRACT.md). `list_tools()` returns the raw
MCP dicts; a caller bridging MCP tools into an Agent would rename that one key.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

from .. import __version__

# The MCP revision our handshake advertises. Revisions are date-stamped by the
# spec ("2026-07-28" superseded "2025-11-25"); during `initialize` a server that
# does not speak ours replies with the newest revision it does, and the verbs we
# use (tools/list, tools/call over stdio) are stable across all of them.
PROTOCOL_VERSION = "2026-07-28"


class MCPError(RuntimeError):
    """Raised for transport failures and JSON-RPC error responses.

    We use one clear exception type so callers can degrade gracefully — catch it,
    tell the user "no MCP server", and carry on — instead of drowning in raw
    BrokenPipe / JSONDecode errors from the plumbing.
    """


class MCPClient:
    """Speaks JSON-RPC 2.0 to one MCP server subprocess over stdio.

    Usage:
        client = MCPClient(["python", "-m", "some_mcp_server"])
        client.initialize()
        for tool in client.list_tools():
            print(tool["name"], "-", tool.get("description"))
        print(client.call_tool("read_file", {"path": "README.md"}))
        client.close()

    Or as a context manager, which starts the process and always closes it:
        with MCPClient([...]) as client:
            client.initialize()
    """

    def __init__(self, command: list[str], *, timeout: float = 30.0):
        self.command = command
        self.timeout = timeout
        self._id = 0            # monotonically increasing JSON-RPC request id
        self._proc: subprocess.Popen[str] | None = None

    # -- process lifecycle --------------------------------------------------
    def start(self) -> None:
        """Launch the server subprocess with stdin/stdout wired to pipes.

        text=True gives us str I/O; bufsize=1 is line buffering, matching the
        newline-delimited framing both sides use. stderr stays inherited so the
        server's own logs still reach the terminal for debugging.
        """
        if self._proc is not None:
            return
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except (OSError, ValueError) as exc:
            # e.g. the command isn't installed — the classic "no server" case.
            raise MCPError(f"could not launch MCP server {self.command!r}: {exc}") from exc

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

    def __enter__(self) -> MCPClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- JSON-RPC transport -------------------------------------------------
    def _send(self, method: str, params: dict | None = None, *, notify: bool = False) -> Any:
        """Write one JSON-RPC frame and (unless a notification) read its reply.

        A *notification* has no id and expects no response — MCP uses one after
        the handshake ("notifications/initialized"). Everything else is a
        request: we match the reply by id and surface any `error` object.
        """
        if self._proc is None:
            self.start()
        assert self._proc and self._proc.stdin and self._proc.stdout

        frame: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        if not notify:
            self._id += 1
            frame["id"] = self._id

        try:
            self._proc.stdin.write(json.dumps(frame) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPError(f"MCP server closed the connection: {exc}") from exc

        if notify:
            return None

        # Read lines until we get the reply carrying our id. Servers may emit
        # unrelated notifications in between, which we skip over.
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise MCPError("MCP server ended without responding (is it a valid MCP server?)")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # not JSON (a stray log line) — ignore and keep reading
            if msg.get("id") != frame["id"]:
                continue
            if "error" in msg:
                err = msg["error"]
                raise MCPError(f"MCP error {err.get('code')}: {err.get('message')}")
            return msg.get("result", {})

    # -- the three verbs ----------------------------------------------------
    def initialize(self) -> dict:
        """Perform the handshake. Must be called before list_tools/call_tool.

        We advertise our protocol version and (empty) capabilities; the server
        replies with its own. Then we send the required `initialized`
        notification, signalling "handshake done, ready for real requests".
        """
        result = self._send("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "musaeus", "version": __version__},
        })
        self._send("notifications/initialized", notify=True)
        return result

    def list_tools(self) -> list[dict]:
        """Discover the server's tools. Each dict has name/description/inputSchema."""
        result = self._send("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, args: dict) -> str:
        """Invoke a tool and return its text output (content blocks flattened).

        MCP returns {"content": [<blocks>], "isError": bool}. We join the text
        blocks; non-text blocks are noted by type so nothing silently vanishes. A
        tool-level failure (isError) is raised, distinct from a transport error.
        """
        result = self._send("tools/call", {"name": name, "arguments": args})
        blocks = result.get("content", [])
        parts: list[str] = []
        for block in blocks:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(f"[{block.get('type', 'unknown')} content]")
        text = "\n".join(parts)
        if result.get("isError"):
            raise MCPError(f"tool {name!r} failed: {text}")
        return text
