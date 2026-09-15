from unittest import mock

from pyvrcft.avatar import AvatarInfo, AvatarParameter
from pyvrcft.client import VRCFTClient
from osc.OSCMessage import OSCMessage, OSCMessageType
from osc.PyVRCFTSender import PyVRCFTSender
from tests import EyeInfoMock


def _avatar(avatar_id: str) -> AvatarInfo:
    return AvatarInfo(
        id=avatar_id,
        name=avatar_id,
        parameters=[
            AvatarParameter(
                name="EyeLeftX",
                address="/avatar/parameters/FT/v2/EyeLeftX",
                type=float,
            )
        ],
    )


def test_oscquery_retries_startup_and_polls_changes_without_receiver():
    client = VRCFTClient()
    client._running = True
    client._recv_sock = None
    client.set("EyeLeftX", 0.25)
    first = _avatar("first")
    second = _avatar("second")
    changes = []
    client.on_avatar_change = lambda info: changes.append(info.id)

    waits = 0

    def finish_after_four_polls(_seconds):
        nonlocal waits
        waits += 1
        if waits == 4:
            client._running = False

    client._wait_while_running = finish_after_four_polls

    with (
        mock.patch(
            "pyvrcft.client.mdns.discover_vrchat_oscquery",
            return_value=("127.0.0.1", 12345),
        ),
        mock.patch(
            "pyvrcft.client.fetch_avatar_oscquery",
            side_effect=[None, first, first, second],
        ) as fetch,
    ):
        client._discover_loop()

    assert fetch.call_count == 4
    assert changes == ["first", "second"]
    assert client.avatar == second
    assert client._slots["v2/EyeLeftX"][0].address.endswith("/FT/v2/EyeLeftX")


def test_oscquery_stops_polling_when_avatar_change_receiver_is_available():
    client = VRCFTClient()
    client._running = True
    client._recv_sock = object()
    client._wait_while_running = mock.Mock()

    with (
        mock.patch(
            "pyvrcft.client.mdns.discover_vrchat_oscquery",
            return_value=("127.0.0.1", 12345),
        ),
        mock.patch("pyvrcft.client.fetch_avatar_oscquery", return_value=_avatar("ready")),
    ):
        client._discover_loop()

    assert client.avatar.id == "ready"
    client._wait_while_running.assert_not_called()


def test_pyvrcft_forwards_normalized_dilation_without_second_normalization():
    sender = PyVRCFTSender()
    sender.client = mock.Mock()
    config = mock.Mock(osc_invert_eye_close=False)
    main_config = mock.Mock(eye_display_id=2)
    eye_info = EyeInfoMock(
        x=0.1,
        y=-0.2,
        blink=0.8,
        pupil_dilation=0.7,
        avg_velocity=0.0,
    )
    eye_info.auxiliary_expressions = {"EyeWideLeft": 0.4, "EyeSquintLeft": 0.9}

    sender.output_osc_info(
        OSCMessage(OSCMessageType.EYE_INFO, (1, eye_info)),
        main_config,
        config,
    )
    sender.output_osc_info(
        OSCMessage(OSCMessageType.EYE_INFO, (0, eye_info)),
        main_config,
        config,
    )

    sender.client.set.assert_called_with("PupilDilation", 0.7)
    assert sender.data.shapes["EyeWideLeft"] == 0.4
    # NEXT's squeeze-derived value owns this channel.
    assert sender.data.shapes["EyeSquintLeft"] == 0.0
