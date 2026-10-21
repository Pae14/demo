import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt6.QtCore')

from sensor_msgs.msg import PointField
import config
import workers
from workers import ROS2Thread


def _pointcloud_message(stamp_ns, points, frame_id='gelsightmini_camera_frame'):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            ),
            frame_id=frame_id,
        ),
        fields=[
            SimpleNamespace(name='x', offset=0, datatype=PointField.FLOAT32),
            SimpleNamespace(name='y', offset=4, datatype=PointField.FLOAT32),
            SimpleNamespace(name='z', offset=8, datatype=PointField.FLOAT32),
        ],
        point_step=12,
        height=2,
        width=2,
        data=points.tobytes(),
    )


def test_pointcloud_is_matched_only_to_identical_image_timestamp():
    thread = ROS2Thread((None, None, None, None), 0.25)
    stamp_ns = 1_000_000_100
    points = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    thread._pointcloud_callback(_pointcloud_message(stamp_ns, points))

    near_match = thread._cache_image(
        stamp_ns + 50_000_000,
        'gelsightmini_camera_frame',
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    matched = thread._cache_image(
        stamp_ns,
        'gelsightmini_camera_frame',
        np.zeros((2, 2, 3), dtype=np.uint8),
    )

    assert near_match is None
    assert matched is not None
    np.testing.assert_array_equal(matched[1], points)


def test_pointcloud_matches_image_when_it_arrives_later(monkeypatch):
    thread = ROS2Thread((None, None, None, None), 0.25)
    stamp_ns = 1_000_000_100
    points = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    scheduled = []
    monkeypatch.setattr(
        thread,
        '_schedule_inference',
        lambda *args, **kwargs: scheduled.append((args, kwargs)),
    )
    thread._cache_image(stamp_ns, 'gelsightmini_camera_frame', image)

    thread._pointcloud_callback(_pointcloud_message(stamp_ns, points))

    assert len(scheduled) == 1
    assert scheduled[0][1] == {'synchronized': True}
    np.testing.assert_array_equal(scheduled[0][0][3], points)


def test_ros_inference_respects_configured_interval_and_keeps_latest_pair(monkeypatch):
    thread = ROS2Thread((None, None, None, None), 0.25)
    monkeypatch.setattr(config, 'INFER_INTERVAL', 0.08)
    started = []
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def record_inference(image, *_args):
        started.append((image, time.monotonic()))
        if len(started) == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        elif len(started) == 2:
            second_started.set()

    monkeypatch.setattr(thread, '_do_inference', record_inference)
    thread._schedule_inference('first', 0, 0.25)
    assert first_started.wait(timeout=2)
    thread._schedule_inference('stale-pair', 0, 0.25, synchronized=True)
    thread._schedule_inference('latest-pair', 0, 0.25, synchronized=True)
    release_first.set()

    assert second_started.wait(timeout=2)
    assert [image for image, _ in started] == ['first', 'latest-pair']
    assert started[1][1] - started[0][1] >= 0.075


def test_unsynchronized_segmentation_does_not_clear_last_depth_fusion(monkeypatch):
    model = SimpleNamespace(
        names={0: 'defect'},
        predict=lambda *_args, **_kwargs: [SimpleNamespace(boxes=None)],
    )
    thread = ROS2Thread((model, None, None, None), 0.25)
    fusion_updates = []
    thread.fusion_ready.connect(fusion_updates.append)
    monkeypatch.setattr(workers, '_draw_yolo', lambda image, *_args: (
        image, '', (), None
    ))
    monkeypatch.setattr(
        workers, 'extract_segmentation_regions', lambda *_args: []
    )

    thread._do_inference(np.zeros((8, 8, 3), dtype=np.uint8), 0, 0.25)

    assert fusion_updates == []
