import sys
import socket
import struct
import threading
import time
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel,
    QLineEdit, QHBoxLayout, QDoubleSpinBox, QColorDialog, QMessageBox,
    QSizePolicy, QFrame, QSplitter, QSlider
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

MAX_CHANNELS = 2
DEFAULT_XMAX = 200
XMIN = 1
XMAX = 100000000
YMIN = 1
YMAX = 100000000

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 6), dpi=100, facecolor='#fafafa')
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax1.set_facecolor('#000000')
        self.ax2.set_facecolor('#000000')

        self.ax1.set_title("Signal (Voltage vs Time)", fontsize=16)
        self.ax2.set_title("Inference Output (Speed vs Time)", fontsize=16)

        self.ax1.set_xlabel("Time (samples)", fontsize=12)
        self.ax1.set_ylabel("Voltage", fontsize=12)
        self.ax2.set_xlabel("Time (samples)", fontsize=12)
        self.ax2.set_ylabel("Speed", fontsize=12)

        for ax in (self.ax1, self.ax2):
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray')

        super().__init__(self.fig)

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
        self.lines = {0: {}, 1: {}}
        self.colors = {0: {}, 1: {}}
        self.stats_labels = {0: {}, 1: {}}
        self.channel_counts = {0: 0, 1: 0}
        self.default_color = "yellow"
        self.canvas = MplCanvas()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.viewing_history = False
        self.history_index = 0
        self._setup_ui()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _setup_ui(self):
        self.setWindowTitle("TCP Data Stream")
        self.setMinimumSize(2100, 1600)
        self.input_layout = QVBoxLayout()

        self.add_signal_button = QPushButton("Add Signal Channel")
        self.add_signal_button.clicked.connect(lambda: self.add_channel(0))
        self.add_inference_button = QPushButton("Add Inference Channel")
        self.add_inference_button.clicked.connect(lambda: self.add_channel(1))
        self.bg_color_button = QPushButton("Background Color")
        self.bg_color_button.clicked.connect(self.select_background_color)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_timer)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_timer)

        button_bar = QHBoxLayout()
        for btn in [self.add_signal_button, self.add_inference_button, self.bg_color_button, self.start_button, self.stop_button]:
            button_bar.addWidget(btn)
        button_bar.addStretch(1)

        left = QVBoxLayout()
        left.addLayout(self.input_layout)
        left.addLayout(button_bar)
        left.addStretch(1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(300)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        horizontal_splitter = QSplitter(Qt.Horizontal)
        horizontal_splitter.addWidget(left_widget)
        horizontal_splitter.addWidget(self.canvas)
        horizontal_splitter.setStretchFactor(1, 1)

        self.y_label = QLabel("Y max:")
        self.y_input = QDoubleSpinBox()
        self.y_input.setRange(YMIN, YMAX)
        self.y_input.setValue(100)
        self.y_input.setFixedHeight(40)
        self.y_input.valueChanged.connect(self.update_ylim)

        self.x_label = QLabel("X max:")
        self.x_input = QDoubleSpinBox()
        self.x_input.setRange(XMIN, XMAX)
        self.x_input.setValue(DEFAULT_XMAX)
        self.x_input.setFixedHeight(40)
        self.x_input.valueChanged.connect(self.update_xlim)

        bottom_splitter = QSplitter(Qt.Horizontal)
        left_widget_bot = QWidget()
        left_layout_bot = QHBoxLayout(left_widget_bot)
        left_layout_bot.setContentsMargins(10, 10, 10, 10)
        left_layout_bot.addStretch()
        left_layout_bot.addWidget(self.y_label)
        left_layout_bot.addWidget(self.y_input)

        right_widget_bot = QWidget()
        right_layout_bot = QHBoxLayout(right_widget_bot)
        right_layout_bot.setContentsMargins(10, 10, 10, 10)
        right_layout_bot.addWidget(self.x_label)
        right_layout_bot.addWidget(self.x_input)
        right_layout_bot.addStretch()

        bottom_splitter.addWidget(left_widget_bot)
        bottom_splitter.addWidget(right_widget_bot)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)

        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.addWidget(horizontal_splitter)
        vertical_splitter.addWidget(bottom_splitter)
        vertical_splitter.setStretchFactor(0, 5)
        vertical_splitter.setStretchFactor(1, 1)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(vertical_splitter)

        self.history_slider = QSlider(Qt.Horizontal)
        self.history_slider.setRange(0, 0)
        self.history_slider.valueChanged.connect(self.on_history_slider_change)
        layout.addWidget(self.history_slider)
        self.setCentralWidget(container)

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

    def add_channel(self, subplot_index):
        if self.channel_counts[subplot_index] >= MAX_CHANNELS:
            QMessageBox.warning(self, "Limit reached", f"Maximum of {MAX_CHANNELS} channels allowed per subplot.")
            return

        ch = self.channel_counts[subplot_index]
        self.channel_counts[subplot_index] += 1
        default_port = 4000 + ch if subplot_index == 0 else 5000 + ch

        ip_input = QLineEdit("10.42.0.253")
        port_input = QLineEdit(str(default_port))
        checkbox = QPushButton("Hide")
        checkbox.setCheckable(True)
        checkbox.setChecked(True)
        checkbox.clicked.connect(lambda _, si=subplot_index, c=ch: self.toggle_visibility(si, c))

        color_btn = QPushButton("Color")
        color_btn.clicked.connect(lambda _, si=subplot_index, c=ch: self.select_channel_color(si, c))

        apply_btn = QPushButton("Apply Changes")
        apply_btn.setEnabled(False)
        apply_btn.clicked.connect(lambda _, si=subplot_index, c=ch: self.restart_receiver(si, c))

        ip_input.textChanged.connect(lambda: apply_btn.setEnabled(True))
        port_input.textChanged.connect(lambda: apply_btn.setEnabled(True))

        row = QHBoxLayout()
        row.addWidget(QLabel(f"CH{subplot_index}-{ch+1}:"))
        for widget in [ip_input, port_input, checkbox, color_btn, apply_btn]:
            row.addWidget(widget)
        self.input_layout.addLayout(row)

        points = int(self.x_input.value())
        self.inputs[subplot_index][ch] = (ip_input, port_input)
        self.buffers[subplot_index][ch] = deque([0.0] * points, maxlen=points)
        self.full_buffers[subplot_index][ch] = []
        self.colors[subplot_index][ch] = self.default_color
        self.checkboxes[subplot_index][ch] = checkbox
        self.color_buttons[subplot_index][ch] = color_btn
        self.apply_buttons[subplot_index][ch] = apply_btn

        ax = self.canvas.ax1 if subplot_index == 0 else self.canvas.ax2
        line, = ax.plot([], [], label=f"CH{subplot_index}-{ch+1}", color=self.default_color)
        self.lines[subplot_index][ch] = line
        ax.legend()

        label = "Current Voltage" if subplot_index == 0 else "Current Speed"
        stats_label = QLabel(f"Metrics\nMin: 0.0   Max: 0.0   Avg: 0.0   {label}: 0.0")
        stats_label.setStyleSheet("color: black; font-size: 30px;")
        self.input_layout.addWidget(stats_label)
        self.stats_labels[subplot_index][ch] = stats_label

        self.restart_receiver(subplot_index, ch)

    def restart_receiver(self, subplot_index, ch):
        ip_input, port_input = self.inputs[subplot_index][ch]
        ip = ip_input.text().strip()
        port = int(port_input.text().strip())
        self.apply_buttons[subplot_index][ch].setEnabled(False)
        thread = threading.Thread(target=self.tcp_receiver, args=(subplot_index, ch, ip, port), daemon=True)
        self.threads[subplot_index][ch] = thread
        thread.start()

    def tcp_receiver(self, subplot_index, ch, ip, port):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.connect((ip, port))
                    with s:
                        while True:
                            data = s.recv(4)
                            if not data:
                                break
                            if len(data) == 4:
                                value = struct.unpack('<f', data)[0]
                                self.buffers[subplot_index][ch].append(value)
                                self.full_buffers[subplot_index][ch].append(value)
            except Exception:
                time.sleep(1)

    def toggle_visibility(self, subplot_index, ch):
        visible = self.checkboxes[subplot_index][ch].isChecked()
        self.lines[subplot_index][ch].set_visible(visible)
        self.checkboxes[subplot_index][ch].setText("Hide" if visible else "Show")
        self.canvas.draw()

    def select_channel_color(self, subplot_index, ch):
        color = QColorDialog.getColor()
        if color.isValid():
            self.colors[subplot_index][ch] = color.name()
            self.lines[subplot_index][ch].set_color(color.name())
            self.canvas.draw()

    def select_background_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            for ax in (self.canvas.ax1, self.canvas.ax2):
                ax.set_facecolor(color.name())
            self.canvas.draw()

    def update_ylim(self):
        for ax in (self.canvas.ax1, self.canvas.ax2):
            ax.set_ylim(-self.y_input.value(), self.y_input.value())

    def update_xlim(self):
        new_x = int(self.x_input.value())
        for subplot_index in (0, 1):
            for ch in self.buffers[subplot_index]:
                old_buffer = self.buffers[subplot_index][ch]
                new_buffer = deque(old_buffer, maxlen=new_x)
                while len(new_buffer) < new_x:
                    new_buffer.appendleft(0.0)
                self.buffers[subplot_index][ch] = new_buffer
        self.update_plot()

    def update_plot(self):
        for subplot_index, ax in enumerate([self.canvas.ax1, self.canvas.ax2]):
            ax.clear()
            for ch, buffer in self.buffers[subplot_index].items():
                y = self.full_buffers[subplot_index][ch][self.history_index:self.history_index + len(buffer)] if self.viewing_history else list(buffer)
                if len(y) < len(buffer):
                    y = [0.0] * (len(buffer) - len(y)) + y
                x = list(range(len(y)))
                ax.plot(x, y, color=self.colors[subplot_index][ch], label=f"CH{subplot_index}-{ch+1}")
                if y:
                    label = "Current Speed" if subplot_index == 1 else "Current Voltage"
                    self.stats_labels[subplot_index][ch].setText(
                        f"Metrics\nMin: {min(y):.2f}   Max: {max(y):.2f}   Avg: {sum(y)/len(y):.2f}   {label}: {y[-1]:.2f}"
                    )
            ax.set_xlim(0, self.x_input.value())
            ax.set_ylim(-self.y_input.value(), self.y_input.value())
            ax.grid(True, linestyle=':', linewidth=0.5, color='gray')
            ax.legend()
        self.canvas.draw()
        if not self.viewing_history:
            max_len = max((len(buf) for d in self.full_buffers.values() for buf in d.values()), default=0)
            new_value = max(0, max_len - int(self.x_input.value()))
            self.history_slider.blockSignals(True)
            self.history_slider.setValue(new_value)
            self.history_slider.blockSignals(False)
            self.history_slider.setRange(0, new_value)
            self.history_index = self.history_slider.value()

    def on_history_slider_change(self, value):
        if self.viewing_history:
            self.history_index = value
            self.update_plot()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
