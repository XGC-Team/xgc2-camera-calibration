"""Lazy, single-flight discovery for the helper's granted Unix peers."""

import ctypes
import json
import logging
import os
import random
import select
import struct
import threading
import time
from pathlib import Path

from xgc2_xrpc import Client, Endpoint, Limits, ServiceRef


class PeerUnavailable(RuntimeError):
    pass


class UnixPeer:
    """One directory watch and one held describe per peer, shared by its users.

    A ready describe returns immediately in XRPC. After that reply, socket
    events and actual domain-call failures invalidate the binding; repeatedly
    re-arming a ready describe would be a busy loop. No domain call is replayed.
    """

    def __init__(self, path, service, api_version, *, runtime, local_target,
                 limits=None, connected=None):
        self.reference = ServiceRef(local_target, service, api_version, "", "http.v1",
                                    Endpoint("unix", str(path))).validate(discovery=True)
        self.runtime = runtime
        self.local_target = local_target
        self.limits = limits or Limits(connections=2, in_flight=4, call_timeout=35.0)
        self.connected = connected
        self.client = None
        self.description = {}
        self._condition = threading.Condition()
        self._thread = None
        self._stop = False
        self._invalid = False
        self._wake_read = self._wake_write = self._watch_fd = None
        self._state = {"reachable": False, "ready": False, "reason": "not_requested",
                       "instance_id": None, "connect_attempts": 0}
        self._last_log = None
        self._suppressed = 0

    def status(self):
        with self._condition:
            return dict(self._state)

    def activate(self):
        with self._condition:
            if self._stop or self._thread is not None:
                return
            libc = ctypes.CDLL(None, use_errno=True)
            descriptor = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if descriptor < 0:
                raise OSError(ctypes.get_errno(), "inotify_init1")
            # Core creates the runtime directory before starting its processes.
            mask = 0x100 | 0x200 | 0x80 | 0x40 | 0x400 | 0x800
            if libc.inotify_add_watch(descriptor, os.fsencode(Path(self.reference.endpoint.address).parent), mask) < 0:
                os.close(descriptor)
                raise OSError(ctypes.get_errno(), "inotify_add_watch")
            self._watch_fd = descriptor
            self._wake_read, self._wake_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
            self._thread = threading.Thread(target=self._run, name=self.reference.service + "-peer")
            self._thread.start()

    def _wake(self, *_args):
        try:
            os.write(self._wake_write, b"x")
        except (OSError, TypeError):
            pass

    def require_client(self, timeout_seconds=0):
        self.activate()
        with self._condition:
            if timeout_seconds > 0:
                # A capture worker may be the first consumer after startup.
                # Wait on the existing discovery, never replay its command.
                self._condition.wait_for(
                    lambda: self._stop or self._state["ready"], timeout_seconds)
            if not self._state["ready"] or self.client is None:
                raise PeerUnavailable(self.reference.service + " is " + self._state["reason"])
            return self.client

    def invalidate(self, error):
        with self._condition:
            self._invalid = True
            self._state.update(reachable=False, ready=False, reason=str(error))
            self._condition.notify_all()
        self._wake()

    def wait_available(self, stop):
        """Wait only in the capture worker; never in init, status or a handler."""
        self.activate()
        with self._condition:
            while not self._stop and not stop.is_set() and not self._state["ready"]:
                self._condition.wait(.5)
            return not self._stop and not stop.is_set()

    def _failure(self, error):
        now = time.monotonic()
        if self._last_log is None or now - self._last_log >= 10:
            logging.getLogger(__name__).warning("%s unavailable: %s (%d suppressed attempts)",
                self.reference.service, error, self._suppressed)
            self._last_log, self._suppressed = now, 0
        else:
            self._suppressed += 1
        with self._condition:
            self._state.update(reachable=False, ready=False, reason=str(error))
            self._condition.notify_all()

    def _socket_events(self):
        changed = False
        data = os.read(self._watch_fd, 64 << 10)
        offset = 0
        while offset + 16 <= len(data):
            _watch, mask, _cookie, length = struct.unpack_from("iIII", data, offset)
            name = data[offset + 16:offset + 16 + length].split(b"\0", 1)[0]
            offset += 16 + length
            if name == os.fsencode(Path(self.reference.endpoint.address).name) or mask & (0x400 | 0x800 | 0x4000):
                changed = True
        return changed

    def _run(self):
        discovery = Client.from_service(self.reference, runtime=self.runtime,
            local_target=self.local_target, discovery=True, limits=self.limits)
        pending = None
        retry_at, ceiling = 0.0, .25
        socket_changed = False
        try:
            while not self._stop:
                if socket_changed or self._invalid:
                    if pending is not None:
                        pending.cancel()
                        pending = None
                    old, self.client = self.client, None
                    if old is not None:
                        old.close()
                    with self._condition:
                        self._state.update(reachable=False, ready=False, reason="disconnected")
                        self._invalid = False
                        self._condition.notify_all()
                    # A socket appearance is an event, not another retry timer.
                    retry_at = time.monotonic() if socket_changed else time.monotonic() + random.uniform(0, ceiling)
                    ceiling = min(30.0, ceiling * 2)
                    socket_changed = False
                if pending is not None and pending.done():
                    finished, pending = pending, None
                    try:
                        document = json.loads(finished.result().body)
                        observed = ServiceRef.from_dict(document["service_ref"]).validate()
                        if (observed.service != self.reference.service or observed.api_version != self.reference.api_version
                                or observed.target_id != self.local_target or observed.endpoint != self.reference.endpoint):
                            raise ValueError("peer discovery differs from its granted endpoint")
                        self.description = document
                        with self._condition:
                            self._state.update(reachable=True, instance_id=observed.instance_id, reason="waiting_ready")
                        if document["ready"] is True:
                            client = Client.from_service(observed, runtime=self.runtime,
                                local_target=self.local_target, limits=self.limits)
                            try:
                                if self.connected is not None:
                                    self.connected(client, observed)
                            except BaseException:
                                client.close()
                                raise
                            with self._condition:
                                self.client = client
                                self._state.update(ready=True, reason="available")
                                self._condition.notify_all()
                            ceiling = .25
                            self._last_log, self._suppressed = None, 0
                            retry_at = 0.0
                        else:
                            # Protect against a peer that returns not-ready
                            # early instead of holding its readiness request.
                            retry_at = time.monotonic() + random.uniform(0, ceiling)
                            ceiling = min(30.0, ceiling * 2)
                    except Exception as error:
                        self._failure(error)
                        retry_at = time.monotonic() + random.uniform(0, ceiling)
                        ceiling = min(30.0, ceiling * 2)
                missing = not Path(self.reference.endpoint.address).exists()
                now = time.monotonic()
                if self.client is None and pending is None and not missing and now >= retry_at:
                    with self._condition:
                        self._state["connect_attempts"] += 1
                    pending = self.runtime.submit(discovery.call_async(
                        "/v1/describe?wait_ready_ms=30000", method="GET", timeout=32.0))
                    pending.add_done_callback(self._wake)
                if missing and self._state["reason"] != "waiting_socket":
                    self._failure("waiting_socket")
                delay = max(0, retry_at - time.monotonic()) if not missing and pending is None and self.client is None else None
                readable, _, _ = select.select([self._watch_fd, self._wake_read], [], [], delay)
                if self._wake_read in readable:
                    os.read(self._wake_read, 64 << 10)
                if self._watch_fd in readable:
                    socket_changed = self._socket_events()
        finally:
            if pending is not None:
                pending.cancel()
            if self.client is not None:
                self.client.close()
                self.client = None
            discovery.close()

    def close(self):
        with self._condition:
            self._stop = True
            self._state.update(ready=False, reason="stopped")
            self._condition.notify_all()
        self._wake()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        for descriptor in (self._watch_fd, self._wake_read, self._wake_write):
            if descriptor is not None:
                os.close(descriptor)
        self._watch_fd = self._wake_read = self._wake_write = None
