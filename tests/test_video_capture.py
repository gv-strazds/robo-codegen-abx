"""Unit tests for SnapshotCapture.request_event metadata handling.

Constructs SnapshotCapture instances via ``__new__`` + manual attribute
assignment to bypass the Isaac Sim ``Camera`` creation in ``__init__`` —
the request-event logic and metadata merging is pure Python and can be
exercised in isolation.

Covers:
  * Basic ``request_event`` queues a pending capture with correct
    metadata.
  * ``extra_metadata`` keys are merged into the sidecar metadata.
  * ``extra_metadata`` keys override matching keys returned by the
    ``metadata_provider`` callback (caller wins).
  * ``metadata_provider`` errors don't break ``request_event``.
"""
from video_capture import SnapshotCapture


def _make_capture(metadata_provider=None, time_based_enabled=True, time_period_s=0.2):
    """Construct a SnapshotCapture without invoking __init__ — bypasses the
    Isaac Sim Camera instantiation while exposing the same surface."""
    cap = SnapshotCapture.__new__(SnapshotCapture)
    cap._closed = False
    cap._settling_frames = 3
    cap._pending_event_captures = []
    cap._metadata_provider = metadata_provider or (lambda: {})
    cap._time_based_enabled = time_based_enabled
    cap._time_period_s = time_period_s
    cap._time_based_count = 0
    cap._last_time_based_sim_t = None
    cap._capture_calls = []  # used by tick() tests via patched _capture_to
    return cap


# ---------------------------------------------------------------------------
# Basic request_event behaviour
# ---------------------------------------------------------------------------

def test_request_event_queues_pending_capture():
    cap = _make_capture()
    cap.request_event("pick_started", sim_time=1.5, pick_index=2)

    assert len(cap._pending_event_captures) == 1
    remaining, basename, meta = cap._pending_event_captures[0]
    assert remaining == 3  # initial settling frames
    assert basename == "pick_started_pick02_t1.500"
    assert meta["kind"] == "event"
    assert meta["event"] == "pick_started"
    assert meta["sim_time"] == 1.5
    assert meta["pick_index"] == 2


def test_request_event_skips_when_closed():
    cap = _make_capture()
    cap._closed = True
    cap.request_event("pick_started", sim_time=1.0, pick_index=0)
    assert cap._pending_event_captures == []


# ---------------------------------------------------------------------------
# metadata_provider integration
# ---------------------------------------------------------------------------

def test_metadata_provider_keys_merged_into_sidecar():
    cap = _make_capture(metadata_provider=lambda: {"pick_name": "cube_0", "extra": "from_provider"})
    cap.request_event("item_grasped", sim_time=2.0, pick_index=1)

    _, _, meta = cap._pending_event_captures[0]
    assert meta["pick_name"] == "cube_0"
    assert meta["extra"] == "from_provider"


def test_metadata_provider_exception_does_not_break_request_event():
    def _raises():
        raise RuntimeError("oops")
    cap = _make_capture(metadata_provider=_raises)
    cap.request_event("evt", sim_time=1.0, pick_index=0)

    # Capture still queued; metadata only has the base keys.
    assert len(cap._pending_event_captures) == 1
    _, _, meta = cap._pending_event_captures[0]
    assert meta["event"] == "evt"
    assert "pick_name" not in meta  # provider failed to contribute


# ---------------------------------------------------------------------------
# extra_metadata
# ---------------------------------------------------------------------------

def test_extra_metadata_merged_into_sidecar():
    cap = _make_capture()
    cap.request_event(
        "verify_fail", sim_time=10.0, pick_index=4,
        extra_metadata={
            "failed_pick_name": "mustard_bottle_4",
            "failure_detail": "not placed inside target container",
        },
    )

    _, _, meta = cap._pending_event_captures[0]
    assert meta["failed_pick_name"] == "mustard_bottle_4"
    assert meta["failure_detail"] == "not placed inside target container"
    assert meta["event"] == "verify_fail"


def test_extra_metadata_overrides_metadata_provider_keys():
    """Caller-supplied extra_metadata keys must win over metadata_provider's
    defaults, since the caller has more authoritative information about the
    specific event being captured."""
    cap = _make_capture(
        metadata_provider=lambda: {"pick_name": "current_pick", "target_name": "current_target"}
    )
    cap.request_event(
        "verify_fail", sim_time=5.0, pick_index=2,
        extra_metadata={
            "pick_name": "failed_pick",      # overrides provider
            "target_name": "failed_target",  # overrides provider
        },
    )

    _, _, meta = cap._pending_event_captures[0]
    assert meta["pick_name"] == "failed_pick"
    assert meta["target_name"] == "failed_target"


def test_extra_metadata_none_is_no_op():
    cap = _make_capture(metadata_provider=lambda: {"k": "v"})
    cap.request_event("evt", sim_time=1.0, pick_index=0, extra_metadata=None)
    _, _, meta = cap._pending_event_captures[0]
    assert meta["k"] == "v"  # provider keys still merged


def test_extra_metadata_empty_dict_is_no_op():
    cap = _make_capture(metadata_provider=lambda: {"k": "v"})
    cap.request_event("evt", sim_time=1.0, pick_index=0, extra_metadata={})
    _, _, meta = cap._pending_event_captures[0]
    assert meta["k"] == "v"


def test_extra_metadata_preserves_base_event_keys():
    cap = _make_capture()
    cap.request_event(
        "verify_fail", sim_time=7.5, pick_index=3,
        extra_metadata={"pick_name": "x"},
    )
    _, _, meta = cap._pending_event_captures[0]
    # extra_metadata should not clobber the structural keys (kind/event/sim_time/pick_index)
    # unless the caller explicitly does so.  Test the default case.
    assert meta["kind"] == "event"
    assert meta["event"] == "verify_fail"
    assert meta["sim_time"] == 7.5
    assert meta["pick_index"] == 3


# ---------------------------------------------------------------------------
# time_based_enabled (failure-only mode in --snapshot-errors)
# ---------------------------------------------------------------------------

def _patched_capture(cap):
    """Replace ``_capture_to`` with a recorder so ``tick()`` doesn't try to
    actually grab a frame (no Isaac Sim Camera in tests)."""
    captures = []
    def _record(basename, meta):
        captures.append((basename, meta))
    cap._capture_to = _record
    return captures


def test_tick_time_based_enabled_fires_on_first_tick():
    cap = _make_capture(time_based_enabled=True, time_period_s=0.2)
    captures = _patched_capture(cap)
    cap.tick(sim_time=0.0, pick_index=-1)
    assert len(captures) == 1
    basename, meta = captures[0]
    assert meta["kind"] == "time_based"


def test_tick_time_based_disabled_no_capture():
    cap = _make_capture(time_based_enabled=False, time_period_s=0.2)
    captures = _patched_capture(cap)
    cap.tick(sim_time=0.0, pick_index=-1)
    cap.tick(sim_time=0.5, pick_index=-1)
    cap.tick(sim_time=1.0, pick_index=-1)
    # Time-based portion completely suppressed.
    assert captures == []


def test_tick_time_based_disabled_still_drains_pending_event_captures():
    """Failure-only mode must still process queued failure-event captures."""
    cap = _make_capture(time_based_enabled=False)
    captures = _patched_capture(cap)
    # Queue an event-driven capture with settling=0 so it fires on first tick.
    cap._pending_event_captures.append((0, "verify_fail_pick4_t1.000",
                                         {"kind": "event", "event": "verify_fail"}))
    cap.tick(sim_time=1.0, pick_index=4)
    assert len(captures) == 1
    assert captures[0][1]["event"] == "verify_fail"


# ---------------------------------------------------------------------------
# FAILURE_EVENT_NAMES registry
# ---------------------------------------------------------------------------

def test_failure_event_names_includes_all_failure_kinds():
    """Sanity check: the failure-event registry covers every failure source —
    BT FAILURE-edge events, recovery-branch SUCCESS events, watchdog timeouts,
    and the non-BT verify_fail event.  Catches typos / accidental removals."""
    from bt_event_visitor import (
        FAILURE_EVENT_NAMES,
        EV_PICK_UNREACHABLE, EV_GRASP_PREP_FAILED, EV_GRASP_SLIPPED,
        EV_PICK_DEFERRED, EV_PLACE_RECOVERED,
        EV_TIMEOUT_PRE_GRASP, EV_TIMEOUT_APPROACH, EV_TIMEOUT_MOVE_TO_PLACE,
        EV_TIMEOUT_DESCENT, EV_TIMEOUT_LIFT,
        EV_VERIFY_FAIL,
    )
    expected = {
        EV_PICK_UNREACHABLE, EV_GRASP_PREP_FAILED, EV_GRASP_SLIPPED,
        EV_PICK_DEFERRED, EV_PLACE_RECOVERED,
        EV_TIMEOUT_PRE_GRASP, EV_TIMEOUT_APPROACH, EV_TIMEOUT_MOVE_TO_PLACE,
        EV_TIMEOUT_DESCENT, EV_TIMEOUT_LIFT,
        EV_VERIFY_FAIL,
    }
    assert FAILURE_EVENT_NAMES == expected


def test_failure_event_names_excludes_happy_path_events():
    """Happy-path phase events must NOT be in the failure registry, or
    --snapshot-errors mode would over-trigger and defeat its purpose."""
    from bt_event_visitor import (
        FAILURE_EVENT_NAMES,
        EV_TASK_STARTED, EV_PICK_STARTED, EV_ITEM_GRASPED, EV_ITEM_LIFTED,
        EV_ITEM_AT_PLACE, EV_ITEM_RELEASED, EV_PICK_COMPLETE, EV_TASK_FINISHED,
    )
    happy_path = {
        EV_TASK_STARTED, EV_PICK_STARTED, EV_ITEM_GRASPED, EV_ITEM_LIFTED,
        EV_ITEM_AT_PLACE, EV_ITEM_RELEASED, EV_PICK_COMPLETE, EV_TASK_FINISHED,
    }
    assert FAILURE_EVENT_NAMES.isdisjoint(happy_path)


def test_failure_event_names_is_frozenset():
    """Immutable set — no accidental in-place additions at module use sites."""
    from bt_event_visitor import FAILURE_EVENT_NAMES
    assert isinstance(FAILURE_EVENT_NAMES, frozenset)
