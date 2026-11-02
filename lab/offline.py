"""Tools to guarantee and verify that code runs without network access."""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Iterator

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", None, ""}


class NetworkAccessError(RuntimeError):
    """Raised when code tries to reach a non-local host while blocked."""


def _host_of(address) -> str | None:
    if isinstance(address, tuple) and address:
        return str(address[0])
    if isinstance(address, (str, bytes)):  # Unix sockets / named pipes
        return "localhost"
    return None


@contextlib.contextmanager
def block_network(allow_localhost: bool = True) -> Iterator[list[str]]:
    """Block every outgoing connection to a non-local host.

    Yields a list that records the hosts that were attempted, which makes it
    possible to report precisely which code tried to go online.
    """
    attempts: list[str] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def _check(host: str | None) -> None:
        if allow_localhost and host in _LOCAL_HOSTS:
            return
        attempts.append(str(host))
        raise NetworkAccessError(f"Network access blocked (offline mode): {host}")

    def guarded_connect(self, address):
        _check(_host_of(address))
        return original_connect(self, address)

    def guarded_connect_ex(self, address):
        _check(_host_of(address))
        return original_connect_ex(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        _check(_host_of(address))
        return original_create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):
        _check(host.decode() if isinstance(host, bytes) else host)
        return original_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.create_connection = guarded_create_connection
    socket.getaddrinfo = guarded_getaddrinfo
    try:
        yield attempts
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
