from pyvrcft.client import VRCFTClient
from pyvrcft.expressions import UnifiedTrackingData


def test_tracking_override_replaces_adaptive_pupil_value_before_publish():
    client = VRCFTClient(use_oscquery=False)
    data = UnifiedTrackingData()
    data.eye.left.pupil_diameter_mm = 0.72
    data.eye.right.pupil_diameter_mm = 0.68

    client.update_tracking(data, overrides={"v2/PupilDilation": 0.7})

    assert client._values["v2/PupilDilation"] == 0.7
