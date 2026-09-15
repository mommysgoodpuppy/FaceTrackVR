from unittest import mock

from pyvrcft.avatar import AvatarInfo, AvatarParameter
from pyvrcft.client import VRCFTClient


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
