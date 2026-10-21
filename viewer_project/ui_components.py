import math
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QFrame, QSizePolicy, QGridLayout, QSpinBox, QLineEdit, QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import (
    QPixmap, QImage, QFont, QPainter, QColor, QLinearGradient,
    QRadialGradient, QPainterPath, QPen, QBrush
)

from config import *


# https://doc.qt.io/qtforpython-6.5/overviews/stylesheet-reference.html
# https://doc.qt.io/qtforpython-6/PySide6/QtGui/QPainter.html
# https://doc.qt.io/qtforpython-6.5/overviews/animation-overview.html
# https://doc.qt.io/qtforpython-6/PySide6/QtGui/QImage.html

def _lbl(text, size=12, weight=400, color=None, align=None):
    l = QLabel(text)
    col = color or P["text"]
    l.setStyleSheet(
        f"color:{col};font-size:{size}px;font-weight:{weight};"
        f"background:transparent;border:none;"
        f"font-family:'Inter','Segoe UI',sans-serif;"
    )
    if align: l.setAlignment(align)
    return l

def _div():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background:{P['divider']};border:none;max-height:1px;margin:4px 0;")
    return f

_COMBO_SS = f"""
QComboBox {{ background:white; color:{P["text"]}; border:1.5px solid {P["border2"]}; border-radius:17px; padding:0 14px; font-size:12px; font-weight:500; font-family:'Inter','Segoe UI',sans-serif; min-height:34px; }}
QComboBox:hover {{ border-color:{P["accent"]}; }}
QComboBox::drop-down {{ border:none; width:26px; }}
QComboBox::down-arrow {{ border-left:4px solid transparent; border-right:4px solid transparent; border-top:5px solid {P["accent"]}; margin-right:8px; }}
QComboBox QAbstractItemView {{ background:white; color:{P["text"]}; border:1.5px solid {P["border2"]}; border-radius:12px; selection-background-color:rgba(223,95,74,0.10); selection-color:{P["accent"]}; padding:4px; outline:none; font-family:'Inter','Segoe UI',sans-serif; font-size:12px; }}
"""
_SPIN_SS = f"""
QSpinBox {{ background:white; color:{P["text"]}; border:1.5px solid {P["border2"]}; border-radius:10px; padding:0 10px; font-size:12px; font-family:'Inter','Segoe UI',sans-serif; min-height:34px; }}
QSpinBox:focus {{ border-color:{P["accent"]}; }}
QSpinBox::up-button, QSpinBox::down-button {{ background:transparent; border:none; width:18px; }}
QSpinBox::up-arrow {{ border-left:4px solid transparent; border-right:4px solid transparent; border-bottom:5px solid {P["muted"]}; }}
QSpinBox::down-arrow {{ border-left:4px solid transparent; border-right:4px solid transparent; border-top:5px solid {P["muted"]}; }}
"""
_INPUT_SS = f"""
QLineEdit {{ background:white; color:{P["text"]}; border:1.5px solid {P["border2"]}; border-radius:17px; padding:0 14px; font-size:12px; font-family:'Inter','Segoe UI',sans-serif; min-height:34px; }}
QLineEdit:focus {{ border-color:{P["accent"]}; }}
"""
def _combo(): c = QComboBox(); c.setStyleSheet(_COMBO_SS); return c
def _spin(): s = QSpinBox(); s.setStyleSheet(_SPIN_SS); return s
def _input(ph=""): i = QLineEdit(); i.setPlaceholderText(ph); i.setStyleSheet(_INPUT_SS); return i

class PillBadge(QLabel):
    def __init__(self, text, rgb, parent=None):
        super().__init__(text, parent)
        self._rgb = rgb; self.setFixedHeight(22); self.set_active(False)
    def set_active(self, v):
        r, g, b = self._rgb
        if v: self.setStyleSheet(f"background:rgba({r},{g},{b},0.14); color:rgb({max(0,r-45)},{max(0,g-45)},{max(0,b-45)}); border:1px solid rgba({r},{g},{b},0.38); border-radius:11px; padding:1px 10px; font-size:11px; font-weight:600; font-family:'Inter','Segoe UI',sans-serif;")
        else: self.setStyleSheet(f"background:{P['bg2']}; color:{P['muted']}; border:1px solid {P['border']}; border-radius:11px; padding:1px 10px; font-size:11px; font-weight:500; font-family:'Inter','Segoe UI',sans-serif;")

class TopMetricsBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(350)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(5)
        
        self.class_card = GlassCard(10)
        self.class_card.setFixedHeight(48)
        class_layout = QHBoxLayout(self.class_card)
        class_layout.setContentsMargins(12, 6, 12, 6)
        class_layout.setSpacing(10)
        
        self.class_dot = QLabel("●")
        self.class_dot.setStyleSheet("font-size: 16px; background:transparent; border:none;")
        self.class_dot.setFixedWidth(20)
        self.class_name = QLabel("—")
        self.class_name.setStyleSheet(f"color:{P['text']}; font-size:14px; font-weight:700; background:transparent; border:none;")
        self.class_conf = QLabel("—")
        self.class_conf.setStyleSheet(f"color:{P['accent']}; font-size:14px; font-weight:700; background:transparent; border:none;")
        
        class_layout.addWidget(self.class_dot)
        class_layout.addWidget(self.class_name, 1)
        class_layout.addWidget(self.class_conf)
        
        self.shape_card = GlassCard(10)
        self.shape_card.setFixedHeight(100)
        shape_layout = QVBoxLayout(self.shape_card)
        shape_layout.setContentsMargins(10, 8, 10, 8)
        shape_layout.setSpacing(6)
        
        shape_title = QLabel("SHAPE METRICS")
        shape_title.setStyleSheet(f"color:{P['accent']}; font-size:10px; font-weight:800; letter-spacing:1px; background:transparent; border:none;")
        shape_layout.addWidget(shape_title)
        
        shape_grid = QGridLayout()
        shape_grid.setSpacing(8)
        shape_grid.setHorizontalSpacing(16)
        
        self._shape_items = {}
        shape_labels = [("Area (mm²)", "area_mm2", ""), ("Perimeter (mm)", "perimeter_mm", ""), ("Eccentricity", "eccentricity", ""), ("Compactation", "compactation", "")]
        for i, (label, key, unit) in enumerate(shape_labels):
            row, col = i // 2, i % 2
            container = QWidget()
            cl = QVBoxLayout(container); cl.setContentsMargins(0, 2, 0, 2); cl.setSpacing(2)
            val = QLabel("—")
            val.setStyleSheet(f"color:{P['accent']}; font-size:14px; font-weight:700; background:transparent; border:none;")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{P['text2']}; font-size:9px; font-weight:500; background:transparent; border:none;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cl.addWidget(val, 0, Qt.AlignmentFlag.AlignCenter)
            cl.addWidget(lbl)
            shape_grid.addWidget(container, row, col)
            self._shape_items[key] = val
        
        shape_layout.addLayout(shape_grid)
        
        self.texture_card = GlassCard(10)
        self.texture_card.setFixedHeight(90)
        texture_layout = QVBoxLayout(self.texture_card)
        texture_layout.setContentsMargins(10, 8, 10, 8)
        texture_layout.setSpacing(6)
        
        texture_title = QLabel("TEXTURE METRICS")
        texture_title.setStyleSheet(f"color:{P['rose']}; font-size:10px; font-weight:800; letter-spacing:1px; background:transparent; border:none;")
        texture_layout.addWidget(texture_title)
        
        texture_row = QHBoxLayout()
        texture_row.setSpacing(12)
        
        self._texture_items = {}
        texture_labels = [("Homogeneity", "glcm_homogeneity"), ("Energy", "glcm_energy"), ("Entropy", "glcm_entropy")]
        for label, key in texture_labels:
            container = QWidget()
            cl = QVBoxLayout(container); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(2)
            val = QLabel("—")
            val.setStyleSheet(f"color:{P['rose']}; font-size:14px; font-weight:700; background:transparent; border:none;")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{P['text2']}; font-size:9px; font-weight:500; background:transparent; border:none;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cl.addWidget(val); cl.addWidget(lbl)
            texture_row.addWidget(container, 1)
            self._texture_items[key] = val
        
        texture_layout.addLayout(texture_row)

        self.depth_card = GlassCard(10)
        self.depth_card.setFixedHeight(86)
        depth_layout = QVBoxLayout(self.depth_card)
        depth_layout.setContentsMargins(10, 8, 10, 8)
        depth_layout.setSpacing(6)

        depth_title = QLabel("POINT CLOUD DEPTH")
        depth_title.setStyleSheet(f"color:{P['green']}; font-size:10px; font-weight:800; letter-spacing:1px; background:transparent; border:none;")
        depth_layout.addWidget(depth_title)

        depth_grid = QGridLayout()
        depth_grid.setSpacing(4)
        depth_grid.setHorizontalSpacing(12)
        self._depth_items = {}
        depth_labels = [
            ("Mean depth (mm)", "depth_mean_mm"),
            ("P95 depth (mm)", "depth_p95_mm"),
            ("Max depth (mm)", "depth_max_mm"),
            ("Volume (mm³)", "depth_volume_mm3"),
        ]
        for index, (label, key) in enumerate(depth_labels):
            row, col = divmod(index, 2)
            container = QWidget()
            item_layout = QVBoxLayout(container)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(1)
            value = QLabel("—")
            value.setStyleSheet(f"color:{P['green']}; font-size:12px; font-weight:700; background:transparent; border:none;")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption = QLabel(label)
            caption.setStyleSheet(f"color:{P['text2']}; font-size:9px; font-weight:500; background:transparent; border:none;")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            item_layout.addWidget(value)
            item_layout.addWidget(caption)
            depth_grid.addWidget(container, row, col)
            self._depth_items[key] = value
        depth_layout.addLayout(depth_grid)

        
        self._lay.addWidget(self.class_card)
        self._lay.addWidget(self.shape_card)
        self._lay.addWidget(self.texture_card)
        self._lay.addWidget(self.depth_card)
        self.reset()
    
    def reset(self):
        self.class_name.setText("—"); self.class_conf.setText("—")
        self.class_dot.setStyleSheet("font-size: 16px; color: #b09488; background:transparent; border:none;")
        for val in self._shape_items.values(): val.setText("—")
        for val in self._texture_items.values(): val.setText("—")
        self.class_card.hide(); self.shape_card.hide(); self.texture_card.hide()
        self.depth_card.hide()
    
    def update_metrics(self, m: dict):
        if not m: 
            self.reset()
            return
            
        self.class_card.show()
        self.shape_card.show()
        self.texture_card.show()
        self.depth_card.show()
        
        self.class_name.setText(str(m.get("class_name", "UNKNOWN")))
        self.class_conf.setText(str(m.get("conf", "0%")))
        
        c = m.get("color", (100, 100, 100))
        self.class_dot.setStyleSheet(f"font-size: 16px; color: rgb({c[0]},{c[1]},{c[2]}); background:transparent; border:none;")
        
        self._shape_items["area_mm2"].setText(str(m.get("area_mm2", "0")))
        self._shape_items["perimeter_mm"].setText(str(m.get("perimeter_mm", "0")))
        self._shape_items["eccentricity"].setText(str(m.get("eccentricity", "0")))
        self._shape_items["compactation"].setText(str(m.get("compactation", "0")))
        
        self._texture_items["glcm_homogeneity"].setText(str(m.get("glcm_homogeneity", "0")))
        self._texture_items["glcm_energy"].setText(str(m.get("glcm_energy", "0")))
        self._texture_items["glcm_entropy"].setText(str(m.get("glcm_entropy", "0")))
        for key, value in self._depth_items.items():
            value.setText(str(m.get(key, "—")))


class StatusBadge(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22); self.setAlignment(Qt.AlignmentFlag.AlignCenter); self.hide()
    def set_status(self, text, rgb):
        r, g, b = rgb
        self.setText(text.upper())
        self.setStyleSheet(f"background:rgba({r},{g},{b},0.15); color:rgb({max(0,r-40)},{max(0,g-40)},{max(0,b-40)}); border:1px solid rgba({r},{g},{b},0.4); border-radius:11px; padding:0 12px; font-size:10px; font-weight:800; letter-spacing: 0.5px; font-family:'Inter','Segoe UI',sans-serif;")
        self.show()


class ImgPanel(QWidget):
    def __init__(self, title, accent=None, parent=None):
        super().__init__(parent)
        self._pix = None; self._pending_arr = None; self._pending_txt = ""; self._pending_col = None; self._pending_metrics = None
        self._acc = accent or P["accent"]
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._eff = QGraphicsOpacityEffect(self); self._eff.setOpacity(1.0); self.setGraphicsEffect(self._eff)
        self._fade = QPropertyAnimation(self._eff, b"opacity"); self._fade.setDuration(200); self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(4)
        self._card = GlassCard(16); cl = QVBoxLayout(self._card); cl.setContentsMargins(12, 10, 12, 12); cl.setSpacing(6)
        hr = QHBoxLayout(); hr.setSpacing(7)
        dot = _lbl("●", 10, 400, self._acc); dot.setFixedWidth(12)
        self._tlbl = _lbl(title, 11, 600, P["text2"])
        self._status_badge = StatusBadge()
        self._status_badge_3d = StatusBadge()
        hr.addWidget(dot); hr.addWidget(self._tlbl); hr.addStretch(); hr.addWidget(self._status_badge); hr.addWidget(self._status_badge_3d); cl.addLayout(hr)
        ln = QFrame(); ln.setFixedHeight(2); ln.setStyleSheet(f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {self._acc},stop:0.55 rgba(220,100,80,0.25),stop:1 transparent); border:none;border-radius:1px;"); cl.addWidget(ln)
        self._metrics_bar = TopMetricsBar(); cl.addWidget(self._metrics_bar)
        self._il = QLabel(); self._il.setAlignment(Qt.AlignmentFlag.AlignCenter); self._il.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self._il.setMinimumHeight(200); cl.addWidget(self._il, 1)
        outer.addWidget(self._card, 1); self.setMinimumWidth(300); self.setMinimumHeight(500); self._ph()

    @staticmethod
    def _tag(text, prefix):
        return text if text == "NO DEFECTS" else f"{prefix}: {text}"

    def _set_3d(self, metrics):
        status = metrics.get('status_3d') if metrics else None
        if status:
            self._status_badge_3d.set_status(self._tag(status[0], '3D'), status[1])
        else:
            self._status_badge_3d.hide()

    def _ph(self):
        self._pix = None; self._il.setPixmap(QPixmap()); self._il.setText("· · ·"); self._il.setStyleSheet(f"background:{P['bg']};border-radius:10px;border:none; color:{P['muted']};font-size:18px;letter-spacing:8px;")
        self._status_badge.hide(); self._status_badge_3d.hide(); self._metrics_bar.reset()

    def set_image(self, arr, status_text="", status_color=(155,160,165), metrics=None, animated=True):
        if arr is None: self._ph(); return
        
        # Si NO queremos animación (para vídeo en directo), lo pintamos al instante
        if not animated:
            h, w, c = arr.shape
            self._pix = QPixmap.fromImage(QImage(arr.data, w, h, w * c, QImage.Format.Format_RGB888))
            self._ref()
            self._il.setStyleSheet(f"background:{P['bg']};border-radius:10px;border:none;")
            self._il.setText("")
            if status_text: self._status_badge.set_status(self._tag(status_text, '2D'), status_color)
            else: self._status_badge.hide()
            self._set_3d(metrics)
            self._metrics_bar.update_metrics(metrics)
            self._eff.setOpacity(1.0) # Forzamos opacidad máxima sin animar
            return

        # Si SÍ queremos animación (para modo Offline)
        self._pending_arr = arr; self._pending_txt = status_text; self._pending_col = status_color; self._pending_metrics = metrics
        try: self._fade.finished.disconnect()
        except: pass
        self._fade.finished.connect(self._swap); self._fade.setStartValue(self._eff.opacity()); self._fade.setEndValue(0.1); self._fade.start()

    def _swap(self):
        try: self._fade.finished.disconnect()
        except: pass
        if self._pending_arr is not None:
            a = self._pending_arr; self._pending_arr = None
            h, w, c = a.shape
            self._pix = QPixmap.fromImage(QImage(a.data, w, h, w * c, QImage.Format.Format_RGB888))
            self._ref(); self._il.setStyleSheet(f"background:{P['bg']};border-radius:10px;border:none;"); self._il.setText("")
            if self._pending_txt: self._status_badge.set_status(self._tag(self._pending_txt, '2D'), self._pending_col)
            else: self._status_badge.hide()
            self._set_3d(self._pending_metrics)
            self._metrics_bar.update_metrics(self._pending_metrics)
        self._fade.setStartValue(0.1); self._fade.setEndValue(1.0); self._fade.start()
    def _ref(self):
        if self._pix is None: return
        target_w = self._il.width() - 8; target_h = self._il.height() - 8
        if target_w > 0 and target_h > 0:
            self._il.setPixmap(self._pix.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e): super().resizeEvent(e); self._ref()


class AnimatedBG(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents); self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground); self.setAutoFillBackground(False); self.lower()
        import random; rng = random.Random(7)
        self._orbs = []
        for r, g, b, a in ORB_COLS:
            self._orbs.append({"x":rng.uniform(0.05, 0.95), "y":rng.uniform(0.05, 0.95), "r":rng.uniform(0.20, 0.35), "dx":rng.uniform(-1.4e-4, 1.4e-4), "dy":rng.uniform(-1.1e-4, 1.1e-4), "ph":rng.uniform(0, math.tau), "col":(r,g,b,a)})
        self._t = 0.0; QTimer(self, timeout=self._tick, interval=34).start()
    def _tick(self):
        self._t += 0.017
        for o in self._orbs:
            o["x"] += o["dx"] + math.sin(self._t * 0.28 + o["ph"]) * 2.8e-4
            o["y"] += o["dy"] + math.cos(self._t * 0.22 + o["ph"]) * 2.2e-4
            if o["x"] < 0.0 or o["x"] > 1.0: o["dx"] *= -1
            if o["y"] < 0.0 or o["y"] > 1.0: o["dy"] *= -1
            o["x"] = max(0.0, min(1.0, o["x"])); o["y"] = max(0.0, min(1.0, o["y"]))
        self.update()
    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); W, H = self.width(), self.height()
        g = QLinearGradient(0, 0, W * 0.55, H); g.setColorAt(0.0, C_BG); g.setColorAt(0.5, QColor("#fef5ef")); g.setColorAt(1.0, QColor("#fce6dc"))
        p.fillRect(0, 0, W, H, QBrush(g))
        for o in self._orbs:
            cx, cy = o["x"] * W, o["y"] * H; pulse = 1.0 + 0.045 * math.sin(self._t * 0.48 + o["ph"]); r = o["r"] * max(W, H) * pulse; r2, g2, b2, a2 = o["col"]
            rg = QRadialGradient(cx, cy, r); rg.setColorAt(0.0, QColor(r2, g2, b2, a2)); rg.setColorAt(1.0, QColor(r2, g2, b2, 0))
            p.setBrush(QBrush(rg)); p.setPen(Qt.PenStyle.NoPen); p.drawEllipse(QPointF(cx, cy), r, r)
        p.end()

class GlassCard(QWidget):
    def __init__(self, radius=16, parent=None):
        super().__init__(parent); self._r = radius; self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); rect = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        for i in range(7, 0, -1): p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(180, 90, 70, 9 + i)); p.drawRoundedRect(rect.adjusted(0, i*0.55, 0, i*0.55), self._r, self._r)
        p.setBrush(QBrush(C_CARD)); p.setPen(QPen(C_BORDER, 1.0)); p.drawRoundedRect(rect, self._r, self._r)
        sg = QLinearGradient(0, 0, 0, self.height() * 0.30); sg.setColorAt(0.0, QColor(255, 255, 255, 80)); sg.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setBrush(QBrush(sg)); p.setPen(Qt.PenStyle.NoPen); p.drawRoundedRect(rect, self._r, self._r); p.end()

class PillBtn(QWidget):
    clicked = pyqtSignal()
    def __init__(self, text, accent=False, danger=False, success=False, parent=None):
        super().__init__(parent); self._text = text; self._accent = accent; self._danger = danger; self._success = success
        self._hover = False; self._press = False; self._ripple = 0.0; self._rip_x = 0.5; self._rip_y = 0.5; self._enabled2 = True
        self.setFixedHeight(34); self.setCursor(Qt.CursorShape.PointingHandCursor); self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._rtimer = QTimer(self, timeout=self._rtick, interval=15)
    def _rtick(self):
        self._ripple += 0.07
        if self._ripple >= 1.0: self._ripple = 0.0; self._rtimer.stop()
        self.update()
    def setEnabled(self, v): self._enabled2 = v; self.setCursor(Qt.CursorShape.PointingHandCursor if v else Qt.CursorShape.ArrowCursor); self.update()
    def isEnabled(self): return self._enabled2
    def setText(self, t): self._text = t; self.update()
    def _base(self):
        if not self._enabled2: return QColor(P["border"])
        if self._danger: return QColor(P["red"])
        if self._success: return QColor(P["green"])
        if self._accent: return C_ACCENT
        return QColor("white")
    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); W, H = self.width(), self.height(); r = H / 2; rect = QRectF(1, 1, W - 2, H - 2); bg = QColor(self._base())
        if self._enabled2:
            if self._press: bg = bg.darker(112)
            elif self._hover: bg = bg.lighter(110) if (self._accent or self._danger or self._success) else QColor(223, 95, 74, 22)
        if self._enabled2 and not self._press:
            sh = QColor(180, 90, 70, 14) if (self._accent or self._danger or self._success) else QColor(0, 0, 0, 7)
            for i in range(5, 0, -1): p.setPen(Qt.PenStyle.NoPen); p.setBrush(sh); p.drawRoundedRect(rect.adjusted(0, i*0.5, 0, i*0.5), r, r)
        solid = self._accent or self._danger or self._success
        p.setPen(Qt.PenStyle.NoPen if solid else QPen(QColor(P["border2"]), 1.0)); p.setBrush(QBrush(bg)); p.drawRoundedRect(rect, r, r)
        if self._ripple > 0:
            rr = self._ripple * max(W, H) * 1.3; alp = int(70 * (1.0 - self._ripple))
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(255, 255, 255, alp)); p.drawEllipse(QPointF(self._rip_x, self._rip_y), rr, rr * 0.6)
        if not self._enabled2: p.setPen(QColor(P["muted"]))
        elif solid: p.setPen(QColor("white"))
        else: p.setPen(QColor(P["accent"]) if self._hover else QColor(P["text"]))
        p.setFont(QFont("Inter", 11, QFont.Weight.DemiBold)); p.drawText(rect.toRect(), Qt.AlignmentFlag.AlignCenter, self._text); p.end()
    def mousePressEvent(self, e):
        if self._enabled2: self._press = True; self._rip_x = e.position().x(); self._rip_y = e.position().y(); self.update()
    def mouseReleaseEvent(self, e):
        if self._enabled2:
            self._press = False; self._ripple = 0.01; self._rtimer.start()
            if self.rect().contains(e.position().toPoint()): self.clicked.emit()
            self.update()
    def enterEvent(self, e): self._hover = True; self.update()
    def leaveEvent(self, e): self._hover = False; self.update()

class SmoothSlider(QWidget):
    valueChanged = pyqtSignal(int)
    def __init__(self, vmin=0, vmax=100, val=50, parent=None):
        super().__init__(parent); self._min = vmin; self._max = vmax; self._val = val; self._drag = False; self._ts = 1.0; self._tts = 1.0
        self.setFixedHeight(34); self.setCursor(Qt.CursorShape.PointingHandCursor); self._at = QTimer(self, timeout=self._atick, interval=15)
    def _atick(self):
        diff = self._tts - self._ts; self._ts += diff * 0.22
        if abs(diff) < 0.003: self._ts = self._tts; self._at.stop()
        self.update()
    def value(self): return self._val
    def setValue(self, v):
        v = max(self._min, min(self._max, v))
        if v != self._val: self._val = v; self.valueChanged.emit(v); self.update()
    TH = 15
    def _fx(self):
        span = self.width() - self.TH * 2; t = (self._val - self._min) / max(self._max - self._min, 1)
        return self.TH + int(t * span)
    def _vfx(self, x):
        span = self.width() - self.TH * 2; t = (x - self.TH) / max(span, 1)
        return self._min + int(max(0.0, min(1.0, t)) * (self._max - self._min))
    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); W, H = self.width(), self.height(); TH, ty, tr = self.TH, H // 2, 4
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(QColor(P["border"]))); p.drawRoundedRect(QRectF(TH, ty - tr, W - TH*2, tr*2), tr, tr); fx = self._fx()
        if fx > TH:
            gr = QLinearGradient(TH, 0, fx, 0); gr.setColorAt(0.0, QColor(P["accentL"])); gr.setColorAt(1.0, C_ACCENT)
            p.setBrush(QBrush(gr)); p.drawRoundedRect(QRectF(TH, ty - tr, fx - TH, tr*2), tr, tr)
        sc = self._ts; glr = QRadialGradient(fx, ty, TH + 10); glr.setColorAt(0.0, QColor(223, 95, 74, int(65 * sc))); glr.setColorAt(1.0, QColor(223, 95, 74, 0))
        p.setBrush(QBrush(glr)); p.drawEllipse(QPointF(fx, ty), TH+10, TH+10)
        p.setBrush(QBrush(QColor("white"))); p.setPen(QPen(C_ACCENT, 2.0)); p.drawEllipse(QPointF(fx, ty), TH * sc, TH * sc); p.end()
    def mousePressEvent(self, e): self._drag = True; self._tts = 1.3; self._at.start(); self.setValue(self._vfx(int(e.position().x())))
    def mouseMoveEvent(self, e):
        if self._drag: self.setValue(self._vfx(int(e.position().x())))
    def mouseReleaseEvent(self, e): self._drag = False; self._tts = 1.0; self._at.start()
    def enterEvent(self, e): self._tts = 1.12; self._at.start()
    def leaveEvent(self, e):
        if not self._drag: self._tts = 1.0; self._at.start()

class ColSec(QWidget):
    def __init__(self, title, open=True, parent=None):
        super().__init__(parent); self._open = open; self.setStyleSheet("background:transparent;")
        vl = QVBoxLayout(self); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)
        self._hdr = QWidget(); self._hdr.setFixedHeight(30); self._hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hdr.setStyleSheet("QWidget{background:transparent;border-radius:6px;} QWidget:hover{background:rgba(223,95,74,0.07);}")
        hl = QHBoxLayout(self._hdr); hl.setContentsMargins(4,0,4,0); hl.setSpacing(6)
        self._arr = _lbl("▾", 11, 700, P["accent"]); self._arr.setFixedWidth(12)
        tl = QLabel(title.upper()); tl.setStyleSheet(f"color:{P['text2']};font-size:9px;font-weight:700;letter-spacing:1px; background:transparent;border:none;font-family:'Inter','Segoe UI',sans-serif;")
        hl.addWidget(self._arr); hl.addWidget(tl); hl.addStretch(); vl.addWidget(self._hdr)
        self._body = QWidget(); self._body.setStyleSheet("background:transparent;")
        self._bl = QVBoxLayout(self._body); self._bl.setContentsMargins(6,4,4,8); self._bl.setSpacing(6); vl.addWidget(self._body)
        self._anim = QPropertyAnimation(self._body, b"maximumHeight"); self._anim.setDuration(200); self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        if not open: self._body.setMaximumHeight(0); self._arr.setText("▸")
        self._hdr.mousePressEvent = lambda e: self._toggle()
    def _toggle(self):
        self._open = not self._open; self._arr.setText("▾" if self._open else "▸")
        if self._open: self._anim.setStartValue(0); self._anim.setEndValue(self._body.sizeHint().height() + 28)
        else: self._anim.setStartValue(self._body.height()); self._anim.setEndValue(0)
        self._anim.start()
    def add(self, w): self._bl.addWidget(w)
    def addL(self, l): self._bl.addLayout(l)

class ResultCard(QWidget):
    def __init__(self, model_name, accent, parent=None):
        super().__init__(parent); self._name = model_name; self._accent = accent
        self._label = "—"; self._conf = 0.0; self._anim_v = 0.0; self._target = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground); self.setMinimumHeight(100)
        self._timer = QTimer(self, timeout=self._tick, interval=16)
    def set_result(self, label, conf): self._label = label; self._conf = conf; self._target = conf; self._timer.start()
    def reset(self): self._label = "—"; self._conf = 0.0; self._target = 0.0; self._anim_v = 0.0; self._timer.stop(); self.update()
    def _tick(self):
        diff = self._target - self._anim_v; self._anim_v += diff * 0.12
        if abs(diff) < 0.002: self._anim_v = self._target; self._timer.stop()
        self.update()
    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); W, H = self.width(), self.height(); r = 14; rect = QRectF(1.5, 1.5, W-3, H-3)
        for i in range(6,0,-1): p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(180,90,70,8+i)); p.drawRoundedRect(rect.adjusted(0,i*0.5,0,i*0.5), r, r)
        p.setBrush(QBrush(C_CARD)); p.setPen(QPen(C_BORDER, 1.0)); p.drawRoundedRect(rect, r, r)
        sg = QLinearGradient(0,0,W,0); sg.setColorAt(0, QColor(self._accent)); sg.setColorAt(1, QColor(self._accent[:7]+"44"))
        p.setBrush(QBrush(sg)); p.setPen(Qt.PenStyle.NoPen)
        sr = QRectF(1.5, 1.5, W-3, 4); pp2 = QPainterPath(); pp2.moveTo(sr.left()+r, sr.top()); pp2.lineTo(sr.right()-r, sr.top()); pp2.arcTo(QRectF(sr.right()-r*2, sr.top(), r*2, r*2), 90, -90)
        pp2.lineTo(sr.right(), sr.bottom()); pp2.lineTo(sr.left(), sr.bottom()); pp2.arcTo(QRectF(sr.left(), sr.top(), r*2, r*2), 180, -90); pp2.closeSubpath(); p.drawPath(pp2)
        p.setPen(QColor(self._accent)); p.setFont(QFont("Inter", 9, QFont.Weight.Bold)); p.drawText(QRectF(14,10,W-28,16), Qt.AlignmentFlag.AlignLeft, self._name.upper())
        p.setPen(QColor(P["text"])); p.setFont(QFont("Inter", 17, QFont.Weight.Bold)); p.drawText(QRectF(14,26,W-90,32), Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter, self._label)
        p.setPen(QColor(self._accent)); p.setFont(QFont("Inter", 13, QFont.Weight.DemiBold)); conf_str = f"{self._conf*100:.1f}%" if self._conf > 0 else "—"
        p.drawText(QRectF(W-80,26,66,32), Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter, conf_str)
        bx, by, bw, bh = 14, H-20, W-28, 5; p.setBrush(QColor(P["border"])); p.setPen(Qt.PenStyle.NoPen); p.drawRoundedRect(QRectF(bx,by,bw,bh), 3, 3)
        if self._anim_v > 0:
            gr = QLinearGradient(bx,0,bx+bw,0); gr.setColorAt(0, QColor(self._accent[:7]+"88")); gr.setColorAt(1, QColor(self._accent)); p.setBrush(QBrush(gr)); p.drawRoundedRect(QRectF(bx,by,bw*self._anim_v,bh), 3, 3)
        p.end()