from settings.BaseSettings import BaseSettingsWidget


class _FakeRoot:
    def __init__(self):
        self.scheduled = []
        self.cancelled = []

    def after(self, delay, callback):
        token = f"after-{len(self.scheduled)}"
        self.scheduled.append((token, delay, callback))
        return token

    def after_cancel(self, token):
        self.cancelled.append(token)


class _FakeFrame:
    def __init__(self, root):
        self.root = root

    def winfo_toplevel(self):
        return self.root


def _widget():
    widget = BaseSettingsWidget.__new__(BaseSettingsWidget)
    root = _FakeRoot()
    widget.frame = _FakeFrame(root)
    widget._pending_validated = None
    widget._config_save_after_id = None
    return widget, root


def test_identical_polled_changes_do_not_starve_debounce_timer():
    widget, root = _widget()
    widget._schedule_debounced_settings_save({"gui_lip_bounce_enable": True})
    first_token = widget._config_save_after_id

    # The 200 ms settings poll sees the same unapplied edit while the 450 ms
    # timer is pending. It must leave that timer alone so it can fire.
    widget._schedule_debounced_settings_save({"gui_lip_bounce_enable": True})
    assert widget._config_save_after_id == first_token
    assert len(root.scheduled) == 1
    assert root.cancelled == []


def test_new_edit_restarts_debounce_timer():
    widget, root = _widget()
    widget._schedule_debounced_settings_save({"gui_lip_bounce_mix": 35.0})
    first_token = widget._config_save_after_id
    widget._schedule_debounced_settings_save({"gui_lip_bounce_mix": 80.0})

    assert len(root.scheduled) == 2
    assert root.cancelled == [first_token]
    assert widget._pending_validated == {"gui_lip_bounce_mix": 80.0}
