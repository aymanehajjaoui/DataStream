import sys
import socket
import struct
import threading
import time
from collections import deque

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QDoubleSpinBox,
    QColorDialog,
    QGridLayout,
    QSplitter,
    QSlider,
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("Qt5Agg")

MAX_CHANNELS = 2
DEFAULT_XMAX = 200
XMIN, XMAX = 1, 100000
YMIN, YMAX = 1, 100000


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 6), dpi=100, facecolor="#fafafa")
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.fig.subplots_adjust(hspace=0.5)
        self.ax1.set_facecolor("#000000")
        self.ax2.set_facecolor("#000000")
        super().__init__(self.fig)

    def set_static_labels(self, font_size):
        scale = 0.7
        self.ax1.set_title("Signal (Voltage vs Time)", fontsize=font_size)
        self.ax2.set_title("Inference Output (Speed vs Time)", fontsize=font_size)
        self.ax1.set_xlabel("Time (samples)", fontsize=font_size * scale)
        self.ax1.set_ylabel("Voltage", fontsize=font_size * scale)
        self.ax2.set_xlabel("Time (samples)", fontsize=font_size * scale)
        self.ax2.set_ylabel("Speed", fontsize=font_size * scale)
        for ax in (self.ax1, self.ax2):
            ax.tick_params(axis='both', labelsize=font_size * 0.8)
            ax.grid(True, linestyle=":", linewidth=0.5, color="gray")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.buffers = {0: {}, 1: {}}
        self.full_buffers = {0: {}, 1: {}}
        self.threads = {0: {}, 1: {}}
        self.inputs = {0: {}, 1: {}}
        self.checkboxes = {0: {}, 1: {}}
        self.color_buttons = {0: {}, 1: {}}
        self.apply_buttons = {0: {}, 1: {}}
        self.delete_buttons = {0: {}, 1: {}}
        self.lines = {0: {}, 1: {}}
        self.colors = {0: {}, 1: {}}
        self.stats_labels = {0: {}, 1: {}}
        self.channel_rows = {0: {}, 1: {}}
        self.channel_counts = {0: 0, 1: 0}
        self.default_color = "yellow"
        self.bg_colors = ["#000000", "#000000"]
        self.y_inputs = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.x_inputs = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.canvas = MplCanvas()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.redraw_plot)
        self.viewing_history = False
        self.history_index = 0
        self._setup_ui()
        self.update_static_labels()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _setup_ui(self):
        self.setWindowTitle("TCP Data Stream")
        self.resize(1200, 800)
        self.setMinimumSize(800, 600)
        self.input_layout = QVBoxLayout()
        self.input_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.add_signal_button = QPushButton("Add Signal Channel")
        self.add_signal_button.clicked.connect(lambda: self.add_channel(0))
        self.add_inference_button = QPushButton("Add Inference Channel")
        self.add_inference_button.clicked.connect(lambda: self.add_channel(1))
        self.bg_color_button1 = QPushButton("Set BG Color (subplot 0)")
        self.bg_color_button1.clicked.connect(lambda: self.select_background_color(0))
        self.bg_color_button2 = QPushButton("Set BG Color (subplot 1)")
        self.bg_color_button2.clicked.connect(lambda: self.select_background_color(1))
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_timer)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_timer)
        button_grid = QGridLayout()
        buttons = [
            self.add_signal_button,
            self.add_inference_button,
            self.bg_color_button1,
            self.bg_color_button2,
            self.start_button,
            self.stop_button,
        ]
        for i, btn in enumerate(buttons):
            row = i // 2
            col = i % 2
            button_grid.addWidget(btn, row, col)
        left = QVBoxLayout()
        left.addLayout(self.input_layout)
        left.addStretch(1)
        left.addLayout(button_grid)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(600)
        horizontal_splitter = QSplitter(Qt.Horizontal)
        horizontal_splitter.addWidget(left_widget)
        horizontal_splitter.addWidget(self.canvas)
        horizontal_splitter.setStretchFactor(1, 1)
        bottom_layout = QVBoxLayout()
        for i in [0, 1]:
            y_label = QLabel(f"Y max (subplot {i}):")
            self.y_inputs[i].setRange(YMIN, YMAX)
            self.y_inputs[i].setValue(100)
            self.y_inputs[i].valueChanged.connect(self.redraw_plot)
            x_label = QLabel(f"X max (subplot {i}):")
            self.x_inputs[i].setRange(XMIN, XMAX)
            self.x_inputs[i].setValue(DEFAULT_XMAX)
            self.x_inputs[i].valueChanged.connect(self.redraw_plot)
            row = QHBoxLayout()
            row.addWidget(y_label)
            row.addWidget(self.y_inputs[i])
            row.addWidget(x_label)
            row.addWidget(self.x_inputs[i])
            bottom_layout.addLayout(row)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(horizontal_splitter)
        layout.addLayout(bottom_layout)
        self.history_slider = QSlider(Qt.Horizontal)
        self.history_slider.setRange(0, 0)
        self.history_slider.valueChanged.connect(self.on_history_slider_change)
        layout.addWidget(self.history_slider)
        self.setCentralWidget(container)

    def update_static_labels(self):
        font_size = max(8, self.width() // 100)
        self.canvas.set_static_labels(font_size)
        self.canvas.draw()

    def add_channel(self, subplot_index):
        if self.channel_counts[subplot_index] >= MAX_CHANNELS:
            return
        ch = self.channel_counts[subplot_index]
        self.channel_counts[subplot_index] += 1
        if self.channel_counts[subplot_index] == MAX_CHANNELS:
            (
                self.add_signal_button
                if subplot_index == 0
                else self.add_inference_button
            ).setEnabled(False)
        default_port = 4000 + ch if subplot_index == 0 else 5000 + ch
        ip_input = QLineEdit("10.42.0.253")
        port_input = QLineEdit(str(default_port))
        checkbox = QPushButton("Hide")
        checkbox.setCheckable(True)
        checkbox.setChecked(True)
        checkbox.clicked.connect(
            lambda _, si=subplot_index, c=ch: self.toggle_visibility(si, c)
        )
        color_btn = QPushButton("Color")
        color_btn.clicked.connect(
            lambda _, si=subplot_index, c=ch: self.select_channel_color(si, c)
        )
        apply_btn = QPushButton("Apply")
        apply_btn.setEnabled(False)
        apply_btn.clicked.connect(
            lambda _, si=subplot_index, c=ch: self.restart_receiver(si, c)
        )
        delete_btn = QPushButton("✖")
        delete_btn.setStyleSheet("color: red; font-weight: bold;")
        delete_btn.clicked.connect(
            lambda _, si=subplot_index, c=ch: self.delete_channel(si, c)
        )
        ip_input.textChanged.connect(lambda: apply_btn.setEnabled(True))
        port_input.textChanged.connect(lambda: apply_btn.setEnabled(True))
        row = QHBoxLayout()
        row.addWidget(QLabel(f"CH{ch+1}"))
        for w in [ip_input, port_input, checkbox, color_btn, apply_btn, delete_btn]:
            row.addWidget(w)
        row_container = QWidget()
        row_container.setLayout(row)
        self.input_layout.addWidget(row_container)
        self.inputs[subplot_index][ch] = (ip_input, port_input)
        self.buffers[subplot_index][ch] = deque(
            [0.0] * int(self.x_inputs[subplot_index].value()),
            maxlen=int(self.x_inputs[subplot_index].value()),
        )
        self.full_buffers[subplot_index][ch] = []
        self.colors[subplot_index][ch] = self.default_color
        self.checkboxes[subplot_index][ch] = checkbox
        self.color_buttons[subplot_index][ch] = color_btn
        self.apply_buttons[subplot_index][ch] = apply_btn
        self.delete_buttons[subplot_index][ch] = delete_btn
        ax = self.canvas.ax1 if subplot_index == 0 else self.canvas.ax2
        (line,) = ax.plot(
            [], [], label=f"CH{ch+1}", color=self.default_color
        )
        self.lines[subplot_index][ch] = line
        ax.legend()
        label = "Current Voltage" if subplot_index == 0 else "Current Speed"
        stat = QLabel(f"Metrics\nMin: 0.0   Max: 0.0   Avg: 0.0   {label}: 0.0")
        self.stats_labels[subplot_index][ch] = stat
        self.input_layout.addWidget(stat)
        self.channel_rows[subplot_index][ch] = row
        self.restart_receiver(subplot_index, ch)
        self.redraw_plot()

    def delete_channel(self, subplot_index, ch):
        try:
            row = self.channel_rows[subplot_index].pop(ch, None)
            if row:
                for i in reversed(range(row.count())):
                    widget = row.itemAt(i).widget()
                    if widget is not None:
                        widget.setParent(None)
            stat_label = self.stats_labels[subplot_index].pop(ch, None)
            if stat_label:
                stat_label.setParent(None)
            if ch in self.lines[subplot_index]:
                try:
                    self.lines[subplot_index][ch].remove()
                except ValueError:
                    pass
                del self.lines[subplot_index][ch]
            for d in [
                self.buffers,
                self.full_buffers,
                self.inputs,
                self.checkboxes,
                self.color_buttons,
                self.apply_buttons,
                self.delete_buttons,
                self.colors,
                self.threads,
            ]:
                d[subplot_index].pop(ch, None)
            self.channel_counts[subplot_index] -= 1
            (
                self.add_signal_button
                if subplot_index == 0
                else self.add_inference_button
            ).setEnabled(True)
            self.redraw_plot()
        except Exception as e:
            print(f"Error deleting channel {ch} in subplot {subplot_index}: {e}")

    def start_timer(self):
        self.timer.start()
        self.viewing_history = False
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_timer(self):
        self.timer.stop()
        self.viewing_history = True
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def restart_receiver(self, subplot_index, ch):
        ip, port = self.inputs[subplot_index][ch]
        ip_str, port_int = ip.text().strip(), int(port.text().strip())
        self.apply_buttons[subplot_index][ch].setEnabled(False)
        thread = threading.Thread(
            target=self.tcp_receiver,
            args=(subplot_index, ch, ip_str, port_int),
            daemon=True,
        )
        self.threads[subplot_index][ch] = thread
        thread.start()

    def tcp_receiver(self, subplot_index, ch, ip, port):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((ip, port))
                    while True:
                        data = s.recv(4)
                        if not data:
                            break
                        if len(data) == 4:
                            val = struct.unpack("<f", data)[0]
                            self.buffers[subplot_index][ch].append(val)
                            self.full_buffers[subplot_index][ch].append(val)
            except:
                time.sleep(1)

    def redraw_plot(self):
        font_size = max(10, self.width() // 100)
        for subplot_index, ax in enumerate([self.canvas.ax1, self.canvas.ax2]):
            has_lines = False
            for ch, buffer in self.buffers[subplot_index].items():
                y = (
                    self.full_buffers[subplot_index][ch][
                        self.history_index : self.history_index + len(buffer)
                    ]
                    if self.viewing_history
                    else list(buffer)
                )
                x = list(range(len(y)))
                if y:
                    self.lines[subplot_index][ch].set_data(x, y)
                    self.lines[subplot_index][ch].set_color(self.colors[subplot_index][ch])
                    has_lines = True
                    label = "Current Speed" if subplot_index == 1 else "Current Voltage"
                    self.stats_labels[subplot_index][ch].setText(
                        f"Metrics\nMin: {min(y):.2f} Max: {max(y):.2f} Avg: {sum(y)/len(y):.2f} {label}: {y[-1]:.2f}"
                    )
            ax.set_xlim(0, self.x_inputs[subplot_index].value())
            ax.set_ylim(
                -self.y_inputs[subplot_index].value(),
                self.y_inputs[subplot_index].value(),
            )
            if has_lines:
                ax.legend()
            elif not self.buffers[subplot_index] and ax.get_legend():
                ax.get_legend().remove()
        self.canvas.draw()

    def select_background_color(self, subplot_index=None):
        color = QColorDialog.getColor()
        if color.isValid():
            if subplot_index is None:
                self.bg_colors[0] = color.name()
                self.bg_colors[1] = color.name()
            else:
                self.bg_colors[subplot_index] = color.name()
            self.apply_background_colors()
            self.redraw_plot()

    def apply_background_colors(self):
        self.canvas.ax1.set_facecolor(self.bg_colors[0])
        self.canvas.ax2.set_facecolor(self.bg_colors[1])

    def select_channel_color(self, subplot_index, ch):
        color = QColorDialog.getColor()
        if color.isValid():
            self.colors[subplot_index][ch] = color.name()
            self.redraw_plot()

    def toggle_visibility(self, subplot_index, ch):
        visible = self.checkboxes[subplot_index][ch].isChecked()
        if ch in self.lines[subplot_index]:
            self.lines[subplot_index][ch].set_visible(visible)
        self.redraw_plot()

    def on_history_slider_change(self, value):
        if self.viewing_history:
            self.history_index = value
            self.redraw_plot()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
