import importlib
import logging


def _setup_log_capture():
    captured = {"errors": [], "warns": []}

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                captured["errors"].append(record.getMessage())
            elif record.levelno >= logging.WARNING:
                captured["warns"].append(record.getMessage())

    logger = logging.getLogger("asset_utils")
    logger.handlers.clear()
    logger.addHandler(CaptureHandler())
    logger.setLevel(logging.WARNING)
    return captured


def test_asset_metadata_primitive_and_usd_detection():
    _setup_log_capture()
    au = importlib.import_module("asset_utils")

    # Primitive asset_type should set is_primitive=True
    meta_prim = au.AssetMetaData(asset_type="cube")
    assert meta_prim.is_primitive is True

    # Non-primitive requires usd_path; should set is_primitive=False
    meta_usd = au.AssetMetaData(asset_type="custom_thing", usd_path="/Some/Path.usd")
    assert meta_usd.is_primitive is False


def test_asset_metadata_invalid_raises():
    _setup_log_capture()
    au = importlib.import_module("asset_utils")
    try:
        au.AssetMetaData(asset_type="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing usd_path when not a primitive")


def test_items_map_contains_expected_entries_and_colors():
    _setup_log_capture()
    au = importlib.import_module("asset_utils")
    assert "soup_can" in au.ITEMS_MAP
    assert "cracker_box" in au.ITEMS_MAP

    soup = au.ITEMS_MAP["soup_can"]
    box = au.ITEMS_MAP["cracker_box"]
    assert isinstance(soup, au.AssetMetaData)
    assert isinstance(box, au.AssetMetaData)
    assert soup.is_primitive is False
    assert box.is_primitive is False
    assert soup.color == "red"
    assert box.color == "red"
    assert soup.usd_path.endswith("005_tomato_soup_can.usd")
    assert box.usd_path.endswith("003_cracker_box.usd")


def test_add_asset_unknown_type_logs_error_and_returns_none():
    captured = _setup_log_capture()
    au = importlib.import_module("asset_utils")
    res = au.add_asset(scene=None, asset_type="does_not_exist")
    assert res is None
    # Ensure we logged an error
    assert any("Unknown asset_type" in e for e in captured["errors"])
