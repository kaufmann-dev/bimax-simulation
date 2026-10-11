"""Desktop startup with the environment filtering used by MCP clients."""

import os
import subprocess
from unittest.mock import Mock

import pytest

from bimax.mcp_server import Connection, desktop_environment


def test_inherited_display_needs_no_session_lookup(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":7")
    lookup = Mock(side_effect=AssertionError("Unexpected session lookup"))
    monkeypatch.setattr(subprocess, "run", lookup)
    assert desktop_environment()["DISPLAY"] == ":7"
    lookup.assert_not_called()


def test_filtered_environment_restores_only_desktop_variables(monkeypatch):
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/private/test-runtime")
    before = dict(os.environ)
    lookup = Mock(return_value=subprocess.CompletedProcess(
        [], 0, "DISPLAY=:0\nWAYLAND_DISPLAY=wayland-0\nXAUTHORITY=/run/user/auth\n"
        "XDG_RUNTIME_DIR=/run/user/session\nDBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/bus\n"
        "SECRET_TOKEN=must-not-import\nQT_QPA_PLATFORM=offscreen\n", ""
    ))
    monkeypatch.setattr(subprocess, "run", lookup)
    environment = desktop_environment()
    assert environment["DISPLAY"] == ":0"
    assert environment["WAYLAND_DISPLAY"] == "wayland-0"
    assert environment["XAUTHORITY"] == "/run/user/auth"
    assert environment["XDG_RUNTIME_DIR"] == "/private/test-runtime"
    assert environment["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/bus"
    assert "SECRET_TOKEN" not in environment
    assert environment.get("QT_QPA_PLATFORM") == before.get("QT_QPA_PLATFORM")
    assert dict(os.environ) == before
    manager = lookup.call_args.kwargs["env"]
    assert manager["XDG_RUNTIME_DIR"] == f"/run/user/{os.getuid()}"
    assert manager["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path=/run/user/{os.getuid()}/bus"


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("systemctl", 5)])
def test_missing_desktop_session_gives_actionable_error(monkeypatch, failure):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=failure))
    with pytest.raises(RuntimeError, match="Start BIMAX from your desktop"):
        desktop_environment()


def test_session_without_display_is_rejected(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, "", "")))
    with pytest.raises(RuntimeError, match="active graphical desktop"):
        desktop_environment()


def test_app_crash_is_reported_without_waiting_for_socket(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    discover = Mock(return_value=[])
    monkeypatch.setattr(Connection, "discover", discover)
    monkeypatch.setattr("bimax.mcp_server.Path.is_file", lambda path: True)
    child = Mock()
    child.poll.return_value = -6
    launch = Mock(return_value=child)
    monkeypatch.setattr(subprocess, "Popen", launch)
    sleep = Mock(side_effect=AssertionError("Must not wait after a startup crash"))
    monkeypatch.setattr("bimax.mcp_server.time.sleep", sleep)
    with pytest.raises(RuntimeError, match="exited during startup.*-6"):
        Connection().connect(None)
    discover.assert_called_once()
    assert launch.call_args.kwargs["env"]["DISPLAY"] == ":0"
    sleep.assert_not_called()
