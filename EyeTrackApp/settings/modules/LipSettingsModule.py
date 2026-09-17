import threading
import tkinter as tk
from tkinter import filedialog, ttk

from pydantic import model_validator

from settings.modules.BaseModule import BaseSettingsModule, BaseValidationModel

from camera_enum import format_uvc_named_source, label_uvc_cameras, list_uvc_cameras
from localization import tr
from utils.tooltips import attach_tooltip


# These are safe to restore while tracking is running. Device/model selection
# and the master enable switch are deliberately excluded: resetting tuning
# must not start unsupported hardware or disconnect a working camera.
LIP_PROCESSING_FIELDS = (
    "gui_lip_preview",
    "gui_lip_debug_visuals",
    "gui_lip_output_mode",
    "gui_lip_smooth_tongue_stability",
    "gui_lip_smooth_boost_jaw_open",
    "gui_lip_smooth_tongue_retracted_range",
    "gui_lip_response_enable",
    "gui_lip_response_min",
    "gui_lip_response_max",
    "gui_lip_response_curve",
    "gui_lip_bounce_enable",
    "gui_lip_bounce_response_hz",
    "gui_lip_bounce_damping",
    "gui_lip_bounce_mix",
    "gui_lip_use_etvr_smoothing",
    "gui_lip_vrcft_adjust",
    "gui_lip_adjust_jaw_open_min",
    "gui_lip_adjust_jaw_open_max",
    "gui_lip_adjust_mouth_closed_min",
    "gui_lip_adjust_mouth_closed_max",
    "gui_lip_adjust_mouth_open_min",
    "gui_lip_adjust_mouth_open_max",
    "gui_lip_adjust_smile_min",
    "gui_lip_adjust_smile_max",
    "gui_lip_adjust_frown_min",
    "gui_lip_adjust_frown_max",
    "gui_lip_adjust_pucker_min",
    "gui_lip_adjust_pucker_max",
    "gui_lip_adjust_funnel_min",
    "gui_lip_adjust_funnel_max",
    "gui_lip_adjust_tongue_out_min",
    "gui_lip_adjust_tongue_out_max",
)


def lip_processing_defaults() -> dict:
    """Return one authoritative set of user-restorable mouth defaults."""
    from config import EyeTrackSettingsConfig

    defaults = EyeTrackSettingsConfig()
    return {field: getattr(defaults, field) for field in LIP_PROCESSING_FIELDS}


class LipValidationModel(BaseValidationModel):
    gui_lip_enable: bool
    gui_lip_use_etvr_smoothing: bool
    gui_lip_device: str
    gui_lip_onnx_path: str
    gui_lip_preview: bool
    gui_lip_debug_visuals: bool
    gui_lip_output_mode: str
    gui_lip_smooth_tongue_stability: str
    gui_lip_smooth_boost_jaw_open: bool
    gui_lip_smooth_tongue_retracted_range: float
    gui_lip_response_enable: bool
    gui_lip_response_min: float
    gui_lip_response_max: float
    gui_lip_response_curve: float
    gui_lip_bounce_enable: bool
    gui_lip_bounce_response_hz: float
    gui_lip_bounce_damping: float
    gui_lip_bounce_mix: float
    gui_lip_vrcft_adjust: bool
    gui_lip_adjust_jaw_open_min: float
    gui_lip_adjust_jaw_open_max: float
    gui_lip_adjust_mouth_closed_min: float
    gui_lip_adjust_mouth_closed_max: float
    gui_lip_adjust_mouth_open_min: float
    gui_lip_adjust_mouth_open_max: float
    gui_lip_adjust_smile_min: float
    gui_lip_adjust_smile_max: float
    gui_lip_adjust_frown_min: float
    gui_lip_adjust_frown_max: float
    gui_lip_adjust_pucker_min: float
    gui_lip_adjust_pucker_max: float
    gui_lip_adjust_funnel_min: float
    gui_lip_adjust_funnel_max: float
    gui_lip_adjust_tongue_out_min: float
    gui_lip_adjust_tongue_out_max: float

    @model_validator(mode="after")
    def check_paths(self):
        if self.gui_lip_device is None:
            self.gui_lip_device = ""
        if self.gui_lip_onnx_path is None:
            self.gui_lip_onnx_path = ""
        if self.gui_lip_output_mode not in ("vrcft", "direct", "new_smooth"):
            raise ValueError("unknown mouth output mode")
        if self.gui_lip_smooth_tongue_stability not in (
            "responsive", "balanced", "strong"
        ):
            raise ValueError("unknown tongue stability preset")
        if not 0.0 <= self.gui_lip_smooth_tongue_retracted_range <= 1.0:
            raise ValueError("retracted tongue range must be 0-1")
        if self.gui_lip_response_min < 0.0:
            raise ValueError("global response minimum cannot be negative")
        if self.gui_lip_response_max <= self.gui_lip_response_min:
            raise ValueError("global response maximum must exceed minimum")
        if not 0.05 <= self.gui_lip_response_curve <= 5.0:
            raise ValueError("global response curve must be 0.05-5")
        if not 0.01 <= self.gui_lip_bounce_response_hz <= 20.0:
            raise ValueError("bounce response must be 0.01-20 Hz")
        if not 0.0 <= self.gui_lip_bounce_damping <= 1.0:
            raise ValueError("bounce damping must be 0-1")
        if not 0.0 <= self.gui_lip_bounce_mix <= 100.0:
            raise ValueError("bounce mix must be 0-100 percent")
        for group in ("jaw_open", "mouth_closed", "mouth_open", "smile",
                      "frown", "pucker", "funnel", "tongue_out"):
            if getattr(self, f"gui_lip_adjust_{group}_min") == getattr(
                self, f"gui_lip_adjust_{group}_max"
            ):
                raise ValueError(f"{group} adjustment minimum and maximum must differ")
        return self


class LipSettingsModule(BaseSettingsModule):
    """Mouth tracking controls, device/model paths, and live status."""

    def __init__(self, config, widget_id, **kwargs):
        super().__init__(config=config, widget_id=widget_id, **kwargs)
        self.validation_model = LipValidationModel
        self.gui_lip_enable = f"-LIPENABLE{widget_id}-"
        self.gui_lip_use_etvr_smoothing = f"-LIPSMOOTH{widget_id}-"
        self.gui_lip_device = f"-LIPDEVICE{widget_id}-"
        self.gui_lip_onnx_path = f"-LIPONNX{widget_id}-"
        self.gui_lip_preview = f"-LIPPREVIEW{widget_id}-"
        self.gui_lip_debug_visuals = f"-LIPDEBUG{widget_id}-"
        self.gui_lip_output_mode = f"-LIPMODE{widget_id}-"
        self.gui_lip_smooth_tongue_stability = f"-LIPTONGUESTABILITY{widget_id}-"
        self.gui_lip_smooth_boost_jaw_open = f"-LIPBOOSTJAWOPEN{widget_id}-"
        self.gui_lip_smooth_tongue_retracted_range = f"-LIPTONGUERETRACTEDRANGE{widget_id}-"
        self.gui_lip_response_enable = f"-LIPRESPONSE{widget_id}-"
        self.gui_lip_response_min = f"-LIPRESPONSEMIN{widget_id}-"
        self.gui_lip_response_max = f"-LIPRESPONSEMAX{widget_id}-"
        self.gui_lip_response_curve = f"-LIPRESPONSECURVE{widget_id}-"
        self.gui_lip_bounce_enable = f"-LIPBOUNCE{widget_id}-"
        self.gui_lip_bounce_response_hz = f"-LIPBOUNCERESPONSE{widget_id}-"
        self.gui_lip_bounce_damping = f"-LIPBOUNCEDAMPING{widget_id}-"
        self.gui_lip_bounce_mix = f"-LIPBOUNCEMIX{widget_id}-"
        self.gui_lip_vrcft_adjust = f"-LIPVRCFTADJUST{widget_id}-"
        self._adjust_groups = (
            "jaw_open", "mouth_closed", "mouth_open", "smile",
            "frown", "pucker", "funnel", "tongue_out",
        )
        for group in self._adjust_groups:
            setattr(self, f"gui_lip_adjust_{group}_min", f"-LIPADJ-{group}-MIN-{widget_id}-")
            setattr(self, f"gui_lip_adjust_{group}_max", f"-LIPADJ-{group}-MAX-{widget_id}-")
        self._status_after_id = None
        self._device_display_map: dict[str, str] = {}

    def get_values_map(self) -> dict:
        values = {}
        for key, var in self.tk_vars.items():
            if key == getattr(self, "_status_key", None):
                continue
            value = var.get()
            if key == self.gui_lip_device:
                value = self._device_display_map.get(value, value)
            values[key] = value
        return values

    def build(self, parent):
        row = 0

        enable_var = tk.BooleanVar(value=bool(self.config.gui_lip_enable))
        self.tk_vars[self.gui_lip_enable] = enable_var
        enable_cb = ttk.Checkbutton(
            parent, text=tr("lip.enable"), variable=enable_var
        )
        enable_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        attach_tooltip(enable_cb, tr("lip.enable_tip"))
        row += 1

        preview_var = tk.BooleanVar(value=bool(self.config.gui_lip_preview))
        self.tk_vars[self.gui_lip_preview] = preview_var
        ttk.Checkbutton(parent, text=tr("lip.show_preview"), variable=preview_var).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2
        )
        row += 1

        debug_var = tk.BooleanVar(value=bool(self.config.gui_lip_debug_visuals))
        self.tk_vars[self.gui_lip_debug_visuals] = debug_var
        debug_cb = ttk.Checkbutton(
            parent, text=tr("lip.debug_visuals"), variable=debug_var
        )
        debug_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        attach_tooltip(debug_cb, tr("lip.debug_visuals_tip"))
        row += 1

        mode_row = ttk.Frame(parent)
        mode_row.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        mode_label = ttk.Label(mode_row, text=tr("lip.output_mode"))
        mode_label.pack(side="left", padx=(0, 6))
        mode = str(getattr(self.config, "gui_lip_output_mode", "vrcft"))
        if mode not in ("vrcft", "direct", "new_smooth"):
            mode = "vrcft"
        mode_var = tk.StringVar(value=mode)
        self.tk_vars[self.gui_lip_output_mode] = mode_var
        mode_labels = {
            "vrcft": tr("lip.mode_vrcft"),
            "direct": tr("lip.mode_direct"),
            "new_smooth": tr("lip.mode_new_smooth"),
        }
        display_to_mode = {label: key for key, label in mode_labels.items()}
        display_var = tk.StringVar(value=mode_labels[mode])
        self._mode_display_var = display_var
        self._mode_labels = mode_labels
        mode_combo = ttk.Combobox(
            mode_row,
            textvariable=display_var,
            values=tuple(mode_labels.values()),
            state="readonly",
            width=14,
        )
        mode_combo.pack(side="left")
        attach_tooltip(mode_combo, tr("lip.output_mode_tip"))
        row += 1

        self._smooth_frame = ttk.Frame(parent)
        self._smooth_frame.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 4)
        )
        ttk.Label(self._smooth_frame, text=tr("lip.tongue_stability")).grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=1
        )
        stability = str(
            getattr(self.config, "gui_lip_smooth_tongue_stability", "balanced")
        )
        if stability not in ("responsive", "balanced", "strong"):
            stability = "balanced"
        stability_labels = {
            "responsive": tr("lip.tongue_stability_responsive"),
            "balanced": tr("lip.tongue_stability_balanced"),
            "strong": tr("lip.tongue_stability_strong"),
        }
        stability_var = tk.StringVar(value=stability)
        self.tk_vars[self.gui_lip_smooth_tongue_stability] = stability_var
        stability_display = tk.StringVar(value=stability_labels[stability])
        self._stability_display_var = stability_display
        self._stability_labels = stability_labels
        stability_combo = ttk.Combobox(
            self._smooth_frame,
            textvariable=stability_display,
            values=tuple(stability_labels.values()),
            state="readonly",
            width=12,
        )
        stability_combo.grid(row=0, column=1, sticky="w", pady=1)

        def _on_stability_selected(_event=None):
            display_to_stability = {
                label: key for key, label in stability_labels.items()
            }
            stability_var.set(
                display_to_stability.get(stability_display.get(), "balanced")
            )

        stability_combo.bind("<<ComboboxSelected>>", _on_stability_selected)

        ttk.Label(
            self._smooth_frame, text=tr("lip.tongue_retracted_range")
        ).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=1)
        retracted_var = tk.StringVar(
            value=str(self.config.gui_lip_smooth_tongue_retracted_range)
        )
        self.tk_vars[self.gui_lip_smooth_tongue_retracted_range] = retracted_var
        ttk.Entry(
            self._smooth_frame, textvariable=retracted_var, width=8
        ).grid(row=1, column=1, sticky="w", pady=1)

        boost_jaw_var = tk.BooleanVar(
            value=bool(self.config.gui_lip_smooth_boost_jaw_open)
        )
        self.tk_vars[self.gui_lip_smooth_boost_jaw_open] = boost_jaw_var
        boost_jaw = ttk.Checkbutton(
            self._smooth_frame,
            text=tr("lip.boost_jaw_open"),
            variable=boost_jaw_var,
        )
        boost_jaw.grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 1)
        )
        attach_tooltip(boost_jaw, tr("lip.boost_jaw_open_tip"))
        if mode != "new_smooth":
            self._smooth_frame.grid_remove()

        def _on_mode_selected(_event=None):
            selected = display_to_mode.get(display_var.get(), "vrcft")
            mode_var.set(selected)
            if selected == "new_smooth":
                self._smooth_frame.grid()
            else:
                self._smooth_frame.grid_remove()

        mode_combo.bind("<<ComboboxSelected>>", _on_mode_selected)
        row += 1

        response_var = tk.BooleanVar(value=bool(self.config.gui_lip_response_enable))
        self.tk_vars[self.gui_lip_response_enable] = response_var
        response_frame = ttk.Frame(parent)
        self._response_frame = response_frame

        def _sync_response_controls():
            if response_var.get():
                response_frame.grid()
            else:
                response_frame.grid_remove()

        response_cb = ttk.Checkbutton(
            parent, text=tr("lip.response_enable"), variable=response_var,
            command=_sync_response_controls,
        )
        response_cb.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2
        )
        attach_tooltip(response_cb, tr("lip.response_enable_tip"))
        row += 1

        response_tuning = (
            ("gui_lip_response_min", "lip.response_min"),
            ("gui_lip_response_max", "lip.response_max"),
            ("gui_lip_response_curve", "lip.response_curve"),
        )
        response_frame.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 4)
        )
        for tune_col, (field, label_key) in enumerate(response_tuning):
            label = ttk.Label(response_frame, text=tr(label_key))
            label.grid(row=0, column=tune_col, sticky="w", padx=(0, 12), pady=1)
            if field == "gui_lip_response_curve":
                attach_tooltip(label, tr("lip.response_curve_tip"))
            var = tk.StringVar(value=str(getattr(self.config, field)))
            self.tk_vars[getattr(self, field)] = var
            ttk.Entry(response_frame, textvariable=var, width=8).grid(
                row=1, column=tune_col, sticky="w", pady=1
            )
        if not response_var.get():
            response_frame.grid_remove()
        row += 1

        bounce_var = tk.BooleanVar(value=bool(self.config.gui_lip_bounce_enable))
        self.tk_vars[self.gui_lip_bounce_enable] = bounce_var

        bounce_frame = ttk.Frame(parent)
        self._bounce_frame = bounce_frame

        def _sync_bounce_controls():
            if bounce_var.get():
                bounce_frame.grid()
            else:
                bounce_frame.grid_remove()

        bounce_cb = ttk.Checkbutton(
            parent, text=tr("lip.bounce_enable"), variable=bounce_var,
            command=_sync_bounce_controls,
        )
        bounce_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        attach_tooltip(bounce_cb, tr("lip.bounce_enable_tip"))
        row += 1

        bounce_tuning = (
            ("gui_lip_bounce_response_hz", "lip.bounce_response"),
            ("gui_lip_bounce_damping", "lip.bounce_damping"),
            ("gui_lip_bounce_mix", "lip.bounce_mix"),
        )
        bounce_frame.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 4)
        )
        for tune_col, (field, label_key) in enumerate(bounce_tuning):
            ttk.Label(bounce_frame, text=tr(label_key)).grid(
                row=0, column=tune_col, sticky="w", padx=(0, 12), pady=1
            )
            var = tk.StringVar(value=str(getattr(self.config, field)))
            self.tk_vars[getattr(self, field)] = var
            ttk.Entry(bounce_frame, textvariable=var, width=8).grid(
                row=1, column=tune_col, sticky="w", pady=1
            )
        if not bounce_var.get():
            bounce_frame.grid_remove()
        row += 1

        smoothing_var = tk.BooleanVar(
            value=bool(self.config.gui_lip_use_etvr_smoothing)
        )
        self.tk_vars[self.gui_lip_use_etvr_smoothing] = smoothing_var
        smoothing_cb = ttk.Checkbutton(
            parent, text=tr("lip.use_etvr_smoothing"), variable=smoothing_var
        )
        smoothing_cb.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2
        )
        attach_tooltip(smoothing_cb, tr("lip.use_etvr_smoothing_tip"))
        row += 1

        dev_label = ttk.Label(parent, text=tr("lip.device"))
        dev_label.grid(row=row, column=0, sticky="w", padx=8, pady=2)
        no_source_label = tr("tracking.no_source")
        initial_device = str(self.config.gui_lip_device or "")
        dev_var = tk.StringVar(
            value=initial_device if initial_device else no_source_label
        )
        self.tk_vars[self.gui_lip_device] = dev_var
        self._device_display_map = {no_source_label: ""}
        dev_row = ttk.Frame(parent)
        dev_row.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        dev_entry = ttk.Combobox(
            dev_row, textvariable=dev_var, values=(), width=36
        )
        dev_entry.pack(side="left")

        def _scan_devices():
            scan_button.state(["disabled"])
            scan_button.configure(text=tr("lip.scanning"))
            result = {"done": False, "cameras": []}

            def _worker():
                try:
                    result["cameras"] = list_uvc_cameras()
                except Exception:
                    result["cameras"] = []
                result["done"] = True

            def _apply_when_ready():
                if not result["done"]:
                    parent.after(50, _apply_when_ready)
                    return
                choices = label_uvc_cameras(result["cameras"])
                self._device_display_map = {no_source_label: ""}
                for label, camera in choices:
                    self._device_display_map[label] = format_uvc_named_source(
                        camera["name"], camera["address"]
                    )
                current = self._device_display_map.get(
                    dev_var.get(), dev_var.get()
                )
                address_to_label = {
                    address: label
                    for label, address in self._device_display_map.items()
                }
                # Recognize the address-only format used by older mouth
                # settings, then migrate it to the canonical uvc:name@address
                # source the next time settings are saved.
                address_to_label.update({
                    camera["address"]: label for label, camera in choices
                })
                dev_entry.configure(
                    values=[no_source_label]
                    + [label for label, _camera in choices]
                )
                if current in address_to_label:
                    dev_var.set(address_to_label[current])
                scan_button.configure(text=tr("tracking.scan_btn"))
                scan_button.state(["!disabled"])

            threading.Thread(target=_worker, daemon=True).start()
            parent.after(50, _apply_when_ready)

        scan_button = ttk.Button(
            dev_row, text=tr("tracking.scan_btn"), command=_scan_devices
        )
        scan_button.pack(side="left", padx=(6, 0))
        attach_tooltip(scan_button, tr("lip.scan_cameras_tip"))
        attach_tooltip(dev_entry, tr("lip.device_tip"))
        _scan_devices()
        row += 1

        onnx_label = ttk.Label(parent, text=tr("lip.onnx_path"))
        onnx_label.grid(row=row, column=0, sticky="w", padx=8, pady=2)
        onnx_var = tk.StringVar(value=str(self.config.gui_lip_onnx_path or ""))
        self.tk_vars[self.gui_lip_onnx_path] = onnx_var
        onnx_row = ttk.Frame(parent)
        onnx_row.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        onnx_entry = ttk.Entry(onnx_row, textvariable=onnx_var, width=42)
        onnx_entry.pack(side="left")

        def _choose_model():
            selected = filedialog.askopenfilename(
                title=tr("lip.choose_model"),
                filetypes=((tr("lip.onnx_files"), "*.onnx"), ("All files", "*")),
            )
            if selected:
                onnx_var.set(selected)

        ttk.Button(
            onnx_row, text=tr("lip.browse"), command=_choose_model
        ).pack(side="left", padx=(6, 0))
        attach_tooltip(onnx_entry, tr("lip.onnx_path_tip"))
        row += 1

        # Reset the stateful HTC postprocessor and re-enter its warm-up period.
        recal_btn = ttk.Button(parent, text=tr("lip.recalibrate"), command=self._recalibrate)
        recal_btn.grid(row=row, column=0, sticky="w", padx=8, pady=(8, 2))
        attach_tooltip(recal_btn, tr("lip.recalibrate_tip"))

        self._status_key = f"-LIPSTATUS{self.widget_id}-"
        status_var = tk.StringVar(value="")
        self.tk_vars[self._status_key] = status_var
        status_lbl = ttk.Label(parent, textvariable=status_var, foreground="#9fd49f")
        status_lbl.grid(row=row, column=1, sticky="w", padx=8, pady=(8, 2))
        row += 1

        self._poll_status(parent)
        return None

    def reset_processing_defaults(self) -> dict:
        """Restore mouth tuning without touching enable, device, or model path."""
        defaults = lip_processing_defaults()
        for field, value in defaults.items():
            key = getattr(self, field)
            var = self.tk_vars.get(key)
            if var is not None:
                var.set(value)

        mode = defaults["gui_lip_output_mode"]
        if hasattr(self, "_mode_display_var"):
            self._mode_display_var.set(self._mode_labels[mode])
        if hasattr(self, "_smooth_frame"):
            if mode == "new_smooth":
                self._smooth_frame.grid()
            else:
                self._smooth_frame.grid_remove()

        stability = defaults["gui_lip_smooth_tongue_stability"]
        if hasattr(self, "_stability_display_var"):
            self._stability_display_var.set(self._stability_labels[stability])

        if hasattr(self, "_response_frame"):
            if defaults["gui_lip_response_enable"]:
                self._response_frame.grid()
            else:
                self._response_frame.grid_remove()
        if hasattr(self, "_bounce_frame"):
            if defaults["gui_lip_bounce_enable"]:
                self._bounce_frame.grid()
            else:
                self._bounce_frame.grid_remove()
        return defaults

    def build_advanced(self, parent):
        adjust_var = tk.BooleanVar(value=bool(self.config.gui_lip_vrcft_adjust))
        self.tk_vars[self.gui_lip_vrcft_adjust] = adjust_var
        cb = ttk.Checkbutton(parent, text=tr("lip.vrcft_adjust"), variable=adjust_var)
        cb.grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=2)
        attach_tooltip(cb, tr("lip.vrcft_adjust_tip"))
        ttk.Label(parent, text=tr("lip.adjust_group")).grid(row=1, column=0, padx=8, sticky="w")
        ttk.Label(parent, text=tr("lip.adjust_min")).grid(row=1, column=1, padx=4)
        ttk.Label(parent, text=tr("lip.adjust_max")).grid(row=1, column=2, padx=4)
        for row, group in enumerate(self._adjust_groups, start=2):
            ttk.Label(parent, text=tr(f"lip.adjust_{group}")).grid(
                row=row, column=0, padx=8, pady=1, sticky="w"
            )
            for column, suffix in ((1, "min"), (2, "max")):
                field = f"gui_lip_adjust_{group}_{suffix}"
                var = tk.StringVar(value=str(getattr(self.config, field)))
                self.tk_vars[getattr(self, field)] = var
                ttk.Entry(parent, textvariable=var, width=7).grid(
                    row=row, column=column, padx=4, pady=1
                )

    def _recalibrate(self):
        from lip import get_active_tracker

        tracker = get_active_tracker()
        if tracker is None or tracker._thread is None:
            self._set_status(tr("lip.status_not_running"))
            return
        tracker.recalibrate()
        self._set_status(tr("lip.status_recalibrated"))

    def _set_status(self, text):
        var = self.tk_vars.get(getattr(self, "_status_key", None))
        if var is not None:
            var.set(text)

    def _poll_status(self, parent):
        """Live status: presence + strongest shapes, refreshed ~2 Hz."""
        try:
            from lip import get_active_tracker

            tracker = get_active_tracker()
            if tracker is None or tracker._thread is None:
                self._set_status(
                    tr("lip.status_on") if self.config.gui_lip_enable else ""
                )
            else:
                named = getattr(tracker, "last_shapes", {}) or {}
                if isinstance(named, dict) and named:
                    top = sorted(named.items(), key=lambda kv: -kv[1])[:3]
                    tops = ", ".join(f"{k} {v:.2f}" for k, v in top if v > 0.02)
                else:
                    tops = ""
                self._set_status(
                    f"{tr('lip.status_presence')}: {tracker.presence:.2f}"
                    + (f"  |  {tops}" if tops else "")
                )
        except Exception:
            pass
        try:
            self._status_after_id = parent.winfo_toplevel().after(500, lambda: self._poll_status(parent))
        except Exception:
            pass
