import cv2
import tkinter as tk
from tkinter import ttk

from config import EyeTrackConfig
from eye import EyeId
from localization import tr
from settings.BaseSettings import BaseSettingsWidget
from settings.modules.LipSettingsModule import LipSettingsModule
from utils.img_utils import tk_photo_from_rgb


class LipSettingsWidget(BaseSettingsWidget):
    """Mouth tracking settings and preview."""

    def __init__(self, widget_id: EyeId, main_config: EyeTrackConfig):
        settings_modules = [
            LipSettingsModule,
        ]
        super().__init__(widget_id, main_config, settings_modules)
        self._preview_label = None
        self._preview_photo = None
        self._adjustment_visible = False
        self._adjustment_frame = None
        self._adjustment_button = None
        self._preview_section = None
        self._scroll_canvas = None
        self._scroll_window = None

    def build(self, parent):
        outer = ttk.Frame(parent)
        canvas_bg = ttk.Style(outer).lookup("TFrame", "background") or "#1e1f23"
        self._scroll_canvas = tk.Canvas(
            outer,
            background=canvas_bg,
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            outer, orient="vertical", command=self._scroll_canvas.yview
        )
        self._scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # BaseSettingsWidget expects self.frame to contain the settings
        # sections. Keep that contract while returning the outer viewport to
        # the page host.
        self.frame = ttk.Frame(self._scroll_canvas)
        self._scroll_window = self._scroll_canvas.create_window(
            (0, 0), window=self.frame, anchor="nw"
        )
        self.frame.bind("<Configure>", self._update_scroll_region, add="+")
        self._scroll_canvas.bind("<Configure>", self._resize_scroll_content, add="+")
        self._build_module_sections()

        module = self.initialized_modules[0]
        toggle_row = ttk.Frame(self.frame)
        toggle_row.pack(fill="x", padx=8, pady=(2, 0), anchor="n")
        self._adjustment_button = ttk.Button(
            toggle_row,
            text=tr("lip.show_vrcft_adjust"),
            command=self._toggle_adjustment,
        )
        self._adjustment_button.pack(side="left")
        reset_button = ttk.Button(
            toggle_row,
            text=tr("lip.reset_defaults"),
            command=self._reset_mouth_defaults,
        )
        reset_button.pack(side="left", padx=(8, 0))

        self._adjustment_frame = ttk.LabelFrame(
            self.frame, text=tr("lip.vrcft_section")
        )
        # Build eagerly so hidden fields still participate in validation and
        # saving, matching the other settings submenus.
        module.build_advanced(self._adjustment_frame)

        self._preview_section = ttk.LabelFrame(
            self.frame, text=tr("lip.preview_title")
        )
        self._preview_section.pack(fill="x", padx=8, pady=6, anchor="n")
        holder = tk.Frame(
            self._preview_section, width=480, height=240, bg="#1e1f23", bd=0,
            highlightthickness=0,
        )
        holder.pack(padx=8, pady=8, anchor="w")
        holder.pack_propagate(False)
        self._preview_label = tk.Label(
            holder, bg="#1e1f23", fg="#aaaaaa", bd=0,
            highlightthickness=0, text=tr("lip.preview_waiting"),
        )
        self._preview_label.pack(fill="both", expand=True)
        self._install_scroll_bindings(self.frame)
        return outer

    def _update_scroll_region(self, _event=None):
        if self._scroll_canvas is not None:
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

    def _resize_scroll_content(self, event):
        if self._scroll_canvas is not None and self._scroll_window is not None:
            self._scroll_canvas.itemconfigure(self._scroll_window, width=event.width)

    def _install_scroll_bindings(self, root_widget):
        """Route wheel events from every Mouth-page child to its viewport."""
        bind_tag = f"LipSettingsScroll{id(self)}"
        root_widget.bind_class(bind_tag, "<MouseWheel>", self._on_mousewheel)
        root_widget.bind_class(bind_tag, "<Button-4>", self._on_mousewheel)
        root_widget.bind_class(bind_tag, "<Button-5>", self._on_mousewheel)

        def add_tag(widget):
            tags = widget.bindtags()
            if bind_tag not in tags:
                widget.bindtags((bind_tag, *tags))
            for child in widget.winfo_children():
                add_tag(child)

        add_tag(root_widget)
        if self._scroll_canvas is not None:
            add_tag(self._scroll_canvas)

    def _on_mousewheel(self, event):
        if self._scroll_canvas is None:
            return None
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            delta = int(getattr(event, "delta", 0))
            units = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        self._scroll_canvas.yview_scroll(units, "units")
        return "break"

    def _reset_mouth_defaults(self):
        module = self.initialized_modules[0]
        defaults = module.reset_processing_defaults()
        self._cancel_debounced_settings_save()
        self.is_saving = True
        self._update_and_save_config(defaults)

    def _toggle_adjustment(self):
        if self._adjustment_frame is None or self._adjustment_button is None:
            return
        if self._adjustment_visible:
            self._adjustment_frame.pack_forget()
            self._adjustment_button.configure(text=tr("lip.show_vrcft_adjust"))
            self._adjustment_visible = False
        else:
            self._adjustment_frame.pack(
                fill="x", padx=8, pady=6, anchor="n",
                before=self._preview_section,
            )
            self._adjustment_button.configure(text=tr("lip.hide_vrcft_adjust"))
            self._adjustment_visible = True

    def render_tick(self):
        super().render_tick()
        self._update_preview()

    def _update_preview(self):
        if self._preview_label is None:
            return
        if not bool(self.config.gui_lip_preview):
            self._preview_photo = None
            self._preview_label.configure(image="", text=tr("lip.preview_disabled"))
            return
        try:
            from lip import get_active_tracker

            tracker = get_active_tracker()
            if tracker is None:
                self._preview_label.configure(image="", text=tr("lip.preview_waiting"))
                return
            debug = bool(self.config.gui_lip_debug_visuals)
            frame = tracker.get_preview_frame(debug=debug)
            if frame is None:
                return
            size = (480, 240)
            interpolation = cv2.INTER_AREA if frame.shape[0] > size[1] else cv2.INTER_LINEAR
            preview = cv2.resize(frame, size, interpolation=interpolation)
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            photo = tk_photo_from_rgb(rgb, self._preview_label)
            self._preview_photo = photo
            self._preview_label.configure(image=photo, text="")
        except Exception:
            pass
