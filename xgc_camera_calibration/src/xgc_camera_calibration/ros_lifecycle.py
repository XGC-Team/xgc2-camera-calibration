"""Serve local state while waiting for the Run's ROS master, then fence it."""

import logging
import os
import random
import signal
import sys
import threading
import time
from xmlrpc.client import ServerProxy, Transport

import rosgraph
import rospy
from rospy.client import load_command_line_node_params


class _DeadlineTransport(Transport):
    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = 1.0
        return connection


class NodeParameters:
    """Read Core's private command arguments without contacting a ROS peer.

    roslaunch installs private parameters on the already running master instead
    of passing those arguments. Read that namespace once for that launch path.
    """

    def __init__(self, node_name):
        mappings = rosgraph.names.load_mappings(sys.argv)
        namespace = mappings.get("__ns", os.environ.get("ROS_NAMESPACE", "/"))
        self.name = rosgraph.names.ns_join(namespace, mappings.get("__name", node_name))
        self.values = load_command_line_node_params(sys.argv)
        if not self.values:
            try:
                with ServerProxy(rosgraph.get_master_uri(), transport=_DeadlineTransport()) as master:
                    code, _message, values = master.getParam(self.name, self.name)
                if code == 1 and isinstance(values, dict):
                    self.values = values
            except Exception as error:
                logging.getLogger(__name__).debug("ROS launch parameters unavailable: %s", error)

    def get(self, name, *default):
        key = name.lstrip("~")
        if key in self.values:
            return self.values[key]
        if default:
            return default[0]
        raise KeyError("required private parameter: " + name)


class RosMasterBinding:
    """The master is the sole hard dependency; data peers never set ready.

    XMLRPC has no master lifecycle event. Its bounded probes use full jitter
    from 250 ms to 30 s while absent, and one check every 10 s when bound.
    """

    def __init__(self, node_name):
        self.node_name = node_name
        self.uri = rosgraph.get_master_uri()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._identity = None
        self._own_ready = False
        self._status = {"available": False, "reason": "waiting_master", "uri": self.uri,
                        "run_id": None, "pid": None, "connect_attempts": 0}
        self._state_listeners = set()
        self._last_log = None
        self._suppressed = 0
        self._previous_signals = {}
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous_signals[signum] = signal.signal(signum, self._signal)

    def _signal(self, *_args):
        self.stop_readiness()
        # rospy's shutdown flag also interrupts init_node before initialization
        # completes; waiting for is_initialized can lose this stop request.
        rospy.signal_shutdown("process stopped")

    def status(self):
        with self._condition:
            return dict(self._status)

    def subscribe_state(self, callback):
        with self._condition:
            self._state_listeners.add(callback)
        def release():
            with self._condition:
                self._state_listeners.discard(callback)
        return release

    def _state_changed_locked(self):
        for callback in tuple(self._state_listeners):
            callback()

    def _update_status_locked(self, **values):
        if any(self._status.get(key) != value for key, value in values.items()):
            self._status.update(values)
            self._state_changed_locked()

    def _probe(self):
        with self._condition:
            self._update_status_locked(connect_attempts=self._status["connect_attempts"] + 1)
        with ServerProxy(self.uri, transport=_DeadlineTransport()) as master:
            code, _message, pid = master.getPid("/" + self.node_name)
            run_code, _message, run_id = master.getParam("/" + self.node_name, "/run_id")
        if code != 1 or run_code != 1 or not run_id:
            raise RuntimeError("ROS master identity is not available")
        return pid, run_id

    def _unavailable(self, error):
        with self._condition:
            self._update_status_locked(available=False, reason="master_unavailable")
            self._condition.notify_all()
        now = time.monotonic()
        if self._last_log is None or now - self._last_log >= 10:
            logging.getLogger(__name__).warning("ROS master unavailable: %s (%d suppressed attempts)", error, self._suppressed)
            self._last_log, self._suppressed = now, 0
        else:
            self._suppressed += 1

    def initialize(self):
        ceiling = .25
        while not self._stop.is_set():
            try:
                identity = self._probe()
            except Exception as error:
                self._unavailable(error)
                self._stop.wait(random.uniform(0, ceiling))
                ceiling = min(30.0, ceiling * 2)
                continue
            self._identity = identity
            try:
                rospy.init_node(self.node_name, disable_signals=True)
            except rospy.exceptions.ROSInitException:
                if self._stop.is_set():
                    return False
                raise
            if self._stop.is_set():
                return False
            with self._condition:
                self._update_status_locked(available=True, reason="available", pid=identity[0], run_id=identity[1])
                self._condition.notify_all()
            self._thread = threading.Thread(target=self._monitor, name="ros-master-binding")
            self._thread.start()
            return True
        return False

    def mark_ready(self):
        with self._condition:
            if not self._own_ready:
                self._own_ready = True
                self._state_changed_locked()
            self._condition.notify_all()

    def wait_ready(self, timeout):
        with self._condition:
            self._condition.wait_for(lambda: self._stop.is_set() or
                (self._own_ready and self._status["available"]), timeout)
            return not self._stop.is_set() and self._own_ready and self._status["available"]

    def stop_readiness(self):
        with self._condition:
            if not self._stop.is_set():
                self._stop.set()
                self._state_changed_locked()
            self._condition.notify_all()

    def _monitor(self):
        delay, ceiling = 10.0, .25
        while not self._stop.wait(delay):
            try:
                identity = self._probe()
            except Exception as error:
                self._unavailable(error)
                delay = random.uniform(0, ceiling)
                ceiling = min(30.0, ceiling * 2)
                continue
            if identity != self._identity:
                with self._condition:
                    self._update_status_locked(available=False, reason="master_replaced")
                self.stop_readiness()
                rospy.signal_shutdown("ROS master was replaced")
                return
            with self._condition:
                self._update_status_locked(available=True, reason="available")
                self._condition.notify_all()
            delay, ceiling = 10.0, .25

    def close(self):
        self.stop_readiness()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        for signum, handler in self._previous_signals.items():
            signal.signal(signum, handler)
