"""Official MCP SDK stdio server and persistent local desktop connection."""

import argparse
import asyncio
import json
import logging
import os
import socket
import stat
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import anyio
from jsonschema import validate
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    AudioContent,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ListToolsResult,
    ResourceLink,
    TextContent,
    Tool,
)

from .commands import CONVENTIONS
from .protocol import MAX_MESSAGE, TOOLS, runtime_directory


def desktop_environment():
    """Restore desktop access when an MCP client filters its child environment."""
    environment = dict(os.environ)
    if environment.get("DISPLAY"):
        return environment
    manager_environment = dict(environment)
    # systemctl resolves the user manager through its standard runtime directory,
    # even when the private BIMAX sockets use a separate XDG_RUNTIME_DIR.
    manager_environment["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    manager_environment.setdefault(
        "DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus"
    )
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            env=manager_environment,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "BIMAX cannot access the graphical desktop: DISPLAY is missing and the "
            "user session environment is unavailable. Start BIMAX from your desktop "
            "before connecting, or pass DISPLAY and XAUTHORITY in the MCP configuration."
        ) from error
    allowed = {"DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in allowed and value and not environment.get(key):
            environment[key] = value
    if not environment.get("DISPLAY"):
        raise RuntimeError(
            "BIMAX requires an active graphical desktop with DISPLAY available. "
            "Start BIMAX from your desktop before connecting, or pass DISPLAY and "
            "XAUTHORITY in the MCP configuration."
        )
    return environment


class RemoteError(RuntimeError):
    def __init__(self, error):
        self.error = error
        super().__init__(error["message"])


class Connection:
    def __init__(self):
        self.socket: socket.socket | None = None
        self.stream = None
        self.instance_id = None

    def close(self):
        if self.stream:
            self.stream.close()
        if self.socket:
            self.socket.close()
        self.stream = self.socket = self.instance_id = None

    def attach(self, path):
        info = path.lstat()
        if (
            not stat.S_ISSOCK(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise PermissionError("BIMAX socket must be owner-only and owned by this user.")
        self.close()
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        peer.settimeout(30)
        try:
            peer.connect(str(path))
        except Exception:
            peer.close()
            raise
        self.socket, self.stream, self.instance_id = peer, peer.makefile("rb"), path.stem

    def request(self, tool, arguments):
        if self.socket is None:
            self.connect(None)
        data = json.dumps({"tool": tool, "arguments": arguments}, allow_nan=False).encode() + b"\n"
        if len(data) > MAX_MESSAGE:
            raise ValueError("Request exceeds the local message limit.")
        assert self.socket is not None and self.stream is not None
        try:
            self.socket.sendall(data)
            line = self.stream.readline(MAX_MESSAGE + 1)
            if not line.endswith(b"\n") or len(line) > MAX_MESSAGE:
                raise ConnectionError("BIMAX disconnected or exceeded the message limit.")
            response = json.loads(line)
        except (OSError, ValueError):
            self.close()
            raise
        if "error" in response:
            raise RemoteError(response["error"])
        return response["result"]

    @staticmethod
    def discover():
        instances = []
        for path in sorted(runtime_directory().glob("*.sock")):
            probe = Connection()
            try:
                probe.attach(path)
                assert probe.socket is not None
                probe.socket.settimeout(1)
                state = probe.request("inspect_state", {})
                instances.append({"instance_id": path.stem, "project_path": state["project_path"]})
            except (OSError, ValueError, RemoteError):
                continue
            finally:
                probe.close()
        return instances

    def connect(self, instance_id):
        instances = self.discover()
        if not instances and instance_id is None:
            executable = Path.home() / ".local/bin/bimax"
            if not executable.is_file():
                raise FileNotFoundError("Install BIMAX first: ~/.local/bin/bimax is missing.")
            app = subprocess.Popen(
                [str(executable)],
                env=desktop_environment(),
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr,
                stderr=sys.stderr,
                start_new_session=True,
            )
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                exit_code = app.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"Installed BIMAX exited during startup (exit code {exit_code}). "
                        "See the MCP server's stderr diagnostics."
                    )
                instances = self.discover()
                if instances:
                    break
                time.sleep(0.1)
            if not instances:
                raise TimeoutError(
                    "Installed BIMAX did not expose its local socket within 30 seconds."
                )
        if instance_id is None and len(instances) != 1:
            raise RemoteError(
                {
                    "code": "select_instance",
                    "message": "Select an instance_id from discover_instances.",
                    "instances": instances,
                }
            )
        selected = instance_id or instances[0]["instance_id"]
        if selected not in {item["instance_id"] for item in instances}:
            raise ValueError("Unknown running BIMAX instance ID.")
        if self.instance_id != selected:
            self.attach(runtime_directory() / f"{selected}.sock")
        return self.request("inspect_state", {})


def make_server(connection):
    lock = anyio.Lock()

    async def list_tools(ctx, params):
        return ListToolsResult(
            tools=[
                Tool(
                    name=name,
                    description=description
                    + " Public coordinates: "
                    + CONVENTIONS["coordinates"]
                    + ". "
                    + CONVENTIONS["angles"]
                    + "; "
                    + CONVENTIONS["inheritance"]
                    + ". Mutations require current session_id, document_id and revision from state.",
                    input_schema=model.model_json_schema(),
                )
                for name, (model, description) in TOOLS.items()
            ]
        )

    async def call_tool(ctx, params):
        try:
            if params.name not in TOOLS:
                raise ValueError("Unknown tool.")
            model = TOOLS[params.name][0]
            arguments = params.arguments or {}
            validate(arguments, model.model_json_schema())
            model.model_validate(arguments)
            async with lock:
                if params.name == "discover_instances":
                    result = {"instances": await anyio.to_thread.run_sync(connection.discover)}
                elif params.name == "connect_instance":
                    result = await anyio.to_thread.run_sync(
                        partial(connection.connect, arguments.get("instance_id"))
                    )
                else:
                    result = await anyio.to_thread.run_sync(
                        partial(connection.request, params.name, arguments)
                    )
            result = dict(result)
            image = result.pop("image", None)
            content: list[
                TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource
            ] = [TextContent(type="text", text=json.dumps(result, allow_nan=False))]
            if image:
                content.append(ImageContent(type="image", data=image, mime_type="image/png"))
            return CallToolResult(content=content, structured_content=result)
        except Exception as error:
            details = (
                error.error
                if isinstance(error, RemoteError)
                else {"code": "invalid_request", "message": str(error)}
            )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(details))],
                structured_content={"error": details},
                is_error=True,
            )

    return Server(
        "bimax-simulation",
        version="0.1.0",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        instructions="Connect, inspect state, begin an editing session, target scenarios by name, preview and poll for PNG, revise, explicitly save, then end the session. Never accept cuts merely by previewing.",
    )


async def serve():
    connection = Connection()
    server = make_server(connection)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        connection.close()


def main():
    argparse.ArgumentParser(description="BIMAX local MCP control over stdio").parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
