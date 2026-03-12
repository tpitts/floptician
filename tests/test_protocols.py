from __future__ import annotations

from floptician.protocols import CameraProtocol, DetectorProtocol, MessageBroadcaster, OBSProtocol
from tests.conftest import FakeCamera, FakeDetector, FakeOBSClient, FakeWebSocketServer


class TestProtocolConformance:
    def test_fake_camera_implements_protocol(self):
        assert isinstance(FakeCamera(), CameraProtocol)

    def test_fake_obs_implements_protocol(self):
        assert isinstance(FakeOBSClient(), OBSProtocol)

    def test_fake_detector_implements_protocol(self):
        assert isinstance(FakeDetector(), DetectorProtocol)

    def test_fake_ws_implements_protocol(self):
        assert isinstance(FakeWebSocketServer(), MessageBroadcaster)
