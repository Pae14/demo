import sys, os, glob, time
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QComboBox, QFrame, QSizePolicy,
    QStatusBar, QFileDialog, QProgressBar, QGridLayout,
    QCompleter, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase, QFont, QPalette, QColor, QPainter, QBrush, QPen

# Módulos propios
import config
from config import *
from ui_components import *
from workers import *
from ui_components import _lbl, _div, _combo, _spin, _input


# https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QApplication.html
# https://doc.qt.io/qtforpython-6/tutorials/basictutorial/signals_and_slots.html
# https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QLayout.html

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
class Sidebar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(230)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        lb = QWidget(); lb.setFixedHeight(54); lb.setStyleSheet(f"background:rgba(255,255,255,0.78);border-bottom:1px solid {P['border']};")
        ll = QHBoxLayout(lb); ll.setContentsMargins(16,0,16,0); ll.setSpacing(6)
        ll.addWidget(_lbl("●",11,400,P["accent"])); ll.addWidget(_lbl("Defect Detector Viewer",13,700,P["text"])); ll.addStretch(); outer.addWidget(lb)
        pb = QWidget(); pb.setFixedHeight(46); pb.setStyleSheet(f"background:rgba(255,255,255,0.72);border-bottom:1px solid {P['border']};")
        pl = QHBoxLayout(pb); pl.setContentsMargins(12,7,12,7); pl.setSpacing(6)
        self._poff = self._pill("Offline",  True); self._pon  = self._pill("Online",   False)
        pl.addWidget(self._poff); pl.addWidget(self._pon); pl.addStretch(); outer.addWidget(pb)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setStyleSheet(f"QScrollArea {{ background:rgba(255,246,242,0.72); border:none; }} QScrollBar:vertical {{ background:transparent; width:5px; }} QScrollBar::handle:vertical {{ background:{P['border2']}; border-radius:3px; min-height:20px; }} QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}")
        self._inner = QWidget(); self._inner.setStyleSheet("background:transparent;")
        self._lay   = QVBoxLayout(self._inner); self._lay.setContentsMargins(10,12,10,20); self._lay.setSpacing(2)
        sc.setWidget(self._inner); outer.addWidget(sc, 1)

    def _pill(self, text, active):
        b = QPushButton(text); b.setFixedHeight(26); b.setCheckable(True); b.setChecked(active)
        if active: b.setStyleSheet(f"QPushButton{{background:{P['accent']};color:white;border:none;border-radius:13px;font-size:11px;font-weight:600;padding:0 14px;font-family:'Inter','Segoe UI',sans-serif;}}")
        else: b.setStyleSheet(f"QPushButton{{background:{P['border']};color:{P['text2']};border:none;border-radius:13px;font-size:11px;font-weight:500;padding:0 14px;font-family:'Inter','Segoe UI',sans-serif;}}QPushButton:hover{{background:rgba(223,95,74,0.12);color:{P['accent']};}}")
        return b

    def set_callbacks(self, cb_off, cb_on): self._poff.clicked.connect(cb_off); self._pon.clicked.connect(cb_on)
    def sec(self, title, open=True): s = ColSec(title, open=open); self._lay.addWidget(s); self._lay.addWidget(_div()); return s
    def stretch(self): self._lay.addStretch()
    def paintEvent(self, _): p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.setBrush(QBrush(QColor(255,246,242,185))); p.setPen(QPen(C_BORDER,1.0)); p.drawRect(0,0,self.width()-1,self.height()); p.end()

# ─────────────────────────────────────────────────────────────────────────────
#  TOPBAR
# ─────────────────────────────────────────────────────────────────────────────
class Topbar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52); self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._lay = QHBoxLayout(self); self._lay.setContentsMargins(20,0,20,0); self._lay.setSpacing(10)
    def lay(self): return self._lay
    def paintEvent(self, _): p = QPainter(self); p.setBrush(QBrush(QColor(255,255,255,205))); p.setPen(QPen(C_BORDER,1.0)); p.drawRect(0,0,self.width(),self.height()); p.end()


# ─────────────────────────────────────────────────────────────────────────────
#  OFFLINE TAB
# ─────────────────────────────────────────────────────────────────────────────
class OfflineTab(QWidget):
    status_msg = pyqtSignal(str)
    SUB_DET = 0; SUB_CLS = 1
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    def __init__(self, mr, cls_ref, sb, tb, parent=None):
        super().__init__(parent)
        self.mr = mr; self.cr = cls_ref; self._sb = sb; self._tb = tb
        self.paths = []; self.all_paths = []; self.idx = 0
        self._arrs = [None]*5; self._wk = None; self._ntp = {}; self._sub = self.SUB_DET
        self._seg_idx = 0
        self.img_dir = config.IMG_DIR; self.lbl_dir = config.LBL_DIR
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._bsb(); self._btb(); self._bc(); self._load()

    def _bsb(self):
        nav = self._sb.sec("Navigation", open=True)
        br = QHBoxLayout(); br.setSpacing(6)
        self.b_prev = PillBtn("← Prev"); self.b_next = PillBtn("Next →")
        self.b_prev.clicked.connect(self._prev); self.b_next.clicked.connect(self._next)
        br.addWidget(self.b_prev); br.addWidget(self.b_next); nav.addL(br)
        jr = QHBoxLayout(); jr.setSpacing(6)
        self.sp_idx = _spin(); self.sp_idx.setMinimum(1)
        go = PillBtn("Go"); go.setFixedWidth(54); go.clicked.connect(self._gos)
        jr.addWidget(self.sp_idx); jr.addWidget(go); nav.addL(jr)

        lr = QHBoxLayout(); lr.setSpacing(6)
        self.b_load_fld = PillBtn("Load Folder"); self.b_load_img = PillBtn("Load Image")
        self.b_load_fld.clicked.connect(self._choose_folder); self.b_load_img.clicked.connect(self._choose_image)
        lr.addWidget(self.b_load_fld); lr.addWidget(self.b_load_img); nav.addL(lr)

        flt = self._sb.sec("Filter by class", open=True)
        self.cls_cb = _combo(); self.cls_cb.addItem("All classes", -1)
        [self.cls_cb.addItem(v, k) for k, v in CLASS_NAMES.items()]
        self.cls_cb.currentIndexChanged.connect(self._on_flt); flt.add(self.cls_cb)

        cfs = self._sb.sec("Confidence threshold", open=True)
        self.cf_big = _lbl(f"{int(CONF_THRESHOLD*100)}%", 30, 700, P["accent"])
        self.cf_big.setAlignment(Qt.AlignmentFlag.AlignCenter); cfs.add(self.cf_big)
        self.cf_sl = SmoothSlider(5, 95, int(CONF_THRESHOLD*100))
        self.cf_sl.valueChanged.connect(lambda v: self.cf_big.setText(f"{v}%")); cfs.add(self.cf_sl)
        ab = PillBtn("Apply & reload", accent=True); ab.clicked.connect(self._run); cfs.add(ab)

        cm = self._sb.sec("Classification model", open=True)
        self.cls_model_cb = _combo()
        self.cls_model_cb.addItem("ResNet18", 0); self.cls_model_cb.addItem("CNN (3-layer)", 1); self.cls_model_cb.addItem("Both", 2)
        cm.add(self.cls_model_cb)

        cls = self._sb.sec("Classes", open=False)
        for cid, name in CLASS_NAMES.items():
            r, g, b = CLASS_COLORS_RGB[cid]
            row = QHBoxLayout(); row.setSpacing(8); row.setContentsMargins(0,0,0,0)
            dot = _lbl("●", 10, 400, f"rgb({r},{g},{b})"); dot.setFixedWidth(14)
            row.addWidget(dot); row.addWidget(_lbl(name, 12, 500, P["text2"])); row.addStretch()
            cls.addL(row)
        self._sb.stretch()

    def _btb(self):
        t = self._tb.lay()
        t.addWidget(_lbl("Offline", 12, 600, P["text"]))
        t.addWidget(_lbl("·", 12, 400, P["muted"]))
        self.ctr  = _lbl("— / —", 12, 700, P["accent"]); t.addWidget(self.ctr)
        self.nlbl = _lbl("", 11, 400, P["muted"]);         t.addWidget(self.nlbl)
        t.addStretch()
        self._pill_det = self._mk_pill("Detection",      True)
        self._pill_cls = self._mk_pill("Classification", False)
        self._pill_det.clicked.connect(lambda: self._switch_sub(self.SUB_DET))
        self._pill_cls.clicked.connect(lambda: self._switch_sub(self.SUB_CLS))
        t.addWidget(self._pill_det); t.addWidget(self._pill_cls); t.addSpacing(10)
        self.srch = _input("Search image name…"); self.srch.setFixedWidth(180)
        self.srch.returnPressed.connect(self._search); t.addWidget(self.srch)
        sb = PillBtn("Search"); sb.setFixedWidth(76); sb.clicked.connect(self._search); t.addWidget(sb)
        eb = PillBtn("Export", accent=True); eb.setFixedWidth(86); eb.clicked.connect(self._export); t.addWidget(eb)

    def _mk_pill(self, text, active):
        b = QPushButton(text); b.setFixedHeight(28); b.setCheckable(True); b.setChecked(active)
        self._apply_pill_ss(b, active); return b

    def _apply_pill_ss(self, btn, active):
        if active:
            btn.setStyleSheet(f"QPushButton{{background:{P['accent']};color:white;border:none;border-radius:14px;font-size:11px;font-weight:600;padding:0 14px;font-family:'Inter','Segoe UI',sans-serif;}}")
        else:
            btn.setStyleSheet(f"QPushButton{{background:{P['border']};color:{P['text2']};border:none;border-radius:14px;font-size:11px;font-weight:500;padding:0 14px;font-family:'Inter','Segoe UI',sans-serif;}}QPushButton:hover{{background:rgba(223,95,74,0.14);color:{P['accent']};}}")

    def _bc(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        self._det_w = QWidget(); self._det_w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        dl = QVBoxLayout(self._det_w); dl.setContentsMargins(18, 14, 18, 14); dl.setSpacing(12)

        sel_card = GlassCard(14)
        sel_lay = QVBoxLayout(sel_card); sel_lay.setContentsMargins(14, 12, 14, 12); sel_lay.setSpacing(10)
        self.seg_label = _lbl("Model 1 / 4 · YOLO12n-seg", 11, 700, P["accent"])
        self.seg_slider = SmoothSlider(0, 3, 0)
        self.seg_slider.valueChanged.connect(self._on_seg_slider)
        sel_lay.addWidget(self.seg_label)
        sel_lay.addWidget(self.seg_slider)
        dl.addWidget(sel_card)

        row = QHBoxLayout(); row.setSpacing(12)
        self.input_panel = ImgPanel("Tactile Input", accent=P["muted"])
        self.input_panel.setMinimumWidth(260)
        row.addWidget(self.input_panel, 1)

        self.output_panels = []
        for i, title in enumerate(SEGMENTATION_TITLES.values()):
            pp = ImgPanel(f"Output: {title}", accent=PANEL_ACCENTS[i])
            pp.setVisible(i == 0)
            self.output_panels.append(pp)
            row.addWidget(pp, 1)

        self.panels = [self.input_panel] + self.output_panels
        dl.addLayout(row, 1)
        outer.addWidget(self._det_w, 1)

        self._cls_w = QWidget(); self._cls_w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        cls_outer = QVBoxLayout(self._cls_w); cls_outer.setContentsMargins(18, 14, 18, 14); cls_outer.setSpacing(12)

        self._gt_bar = QWidget(); self._gt_bar.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._gt_bar.setFixedHeight(44)
        self._gt_bar_lay = QHBoxLayout(self._gt_bar); self._gt_bar_lay.setContentsMargins(0,0,0,0); self._gt_bar_lay.setSpacing(10)
        self._gt_bar_lay.addWidget(_lbl("Ground truth:", 12, 600, P["text2"]))
        self._gt_pills_w = QWidget(); self._gt_pills_w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._gt_pills_lay = QHBoxLayout(self._gt_pills_w); self._gt_pills_lay.setContentsMargins(0,0,0,0); self._gt_pills_lay.setSpacing(6); self._gt_pills_lay.addStretch()
        self._gt_bar_lay.addWidget(self._gt_pills_w, 1); cls_outer.addWidget(self._gt_bar)

        cl = QHBoxLayout(); cl.setSpacing(16)
        self._cls_orig = ImgPanel("Tactile Input", accent=P["muted"]); cl.addWidget(self._cls_orig, 1)

        right_w = QWidget(); right_w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        rl = QVBoxLayout(right_w); rl.setContentsMargins(0,0,0,0); rl.setSpacing(12)

        sel_card = GlassCard(14); sel_lay = QVBoxLayout(sel_card); sel_lay.setContentsMargins(14,12,14,14); sel_lay.setSpacing(8)
        sel_lay.addWidget(_lbl("Model", 10, 700, P["muted"]))
        self.cls_model_cb = _combo()
        self.cls_model_cb.addItem("ResNet18", 0); self.cls_model_cb.addItem("CNN (3-layer)", 1); self.cls_model_cb.addItem("Both", 2)
        sel_lay.addWidget(self.cls_model_cb)
        run_btn = PillBtn("Run classification", accent=True); run_btn.clicked.connect(self._run_cls); sel_lay.addWidget(run_btn)
        rl.addWidget(sel_card)
        self._rc_resnet = ResultCard("ResNet18", P["accent"]); self._rc_cnn = ResultCard("CNN 3-layer", P["rose"])
        rl.addWidget(self._rc_resnet); rl.addWidget(self._rc_cnn); rl.addStretch()
        right_w.setFixedWidth(290); cl.addWidget(right_w)
        cls_outer.addLayout(cl, 1); outer.addWidget(self._cls_w, 1)
        self._cls_w.hide()

    def _on_seg_slider(self, idx):
        self._seg_idx = int(idx)
        self.seg_label.setText(f"Model {idx + 1} / 4 · {SEGMENTATION_TITLES[idx]}")
        for i, pp in enumerate(self.output_panels):
            pp.setVisible(i == idx)
        if self._arrs and len(self._arrs) > idx + 1 and self._arrs[idx + 1] is not None:
            data = self._arrs[idx + 1]
            if data is not None:
                self.output_panels[idx].set_image(data[0], data[1], data[2], data[3])
        else:
            self.output_panels[idx]._ph()
        if self._sub == self.SUB_DET and self.paths and self.mr[idx] is not None:
            self._run_det()

    def _switch_sub(self, idx):
        self._sub = idx
        self._apply_pill_ss(self._pill_det, idx == self.SUB_DET)
        self._apply_pill_ss(self._pill_cls, idx == self.SUB_CLS)
        if idx == self.SUB_DET: self._cls_w.hide(); self._det_w.show()
        else: self._det_w.hide(); self._cls_w.show()
        self._run()

    def _load(self):
        def collect_images(directory):
            if not os.path.isdir(directory):
                return []
            return sorted(
                os.path.join(root, filename)
                for root, _, filenames in os.walk(directory)
                for filename in filenames
                if os.path.splitext(filename)[1].lower() in self.IMAGE_EXTENSIONS
            )

        self.all_paths = collect_images(self.img_dir)
        if not self.all_paths and os.path.abspath(self.img_dir) == os.path.abspath(config.IMG_DIR):
            data_root = os.path.join(config.BASE_DIR, "data")
            for root, _, _ in os.walk(data_root):
                if os.path.basename(root).lower() != "images":
                    continue
                discovered = collect_images(root)
                if discovered:
                    self.img_dir = root
                    self.all_paths = discovered
                    sibling_labels = os.path.join(os.path.dirname(root), "labels")
                    if os.path.isdir(sibling_labels):
                        self.lbl_dir = sibling_labels
                    break
        self.paths = self.all_paths[:]
        self._ntp = {os.path.splitext(os.path.basename(p))[0]: p for p in self.all_paths}
        c = QCompleter(sorted(self._ntp.keys()), self); c.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.srch.setCompleter(c); self.sp_idx.setMaximum(max(1, len(self.paths)))
        self._uc()
        if self.paths:
            self.status_msg.emit(f"{len(self.paths)} images found in {self.img_dir}")
            self.input_panel._ph(); self.output_panels[self._seg_idx]._ph()
            self._run()
        else:
            self.nlbl.setText("No images found")
            self.input_panel._ph(); self.output_panels[self._seg_idx]._ph()
            self.status_msg.emit(f"No images found in {self.img_dir}. Use Load Folder to select the dataset images directory.")

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder", self.img_dir)
        if d:
            self.img_dir = d
            # Try to find labels: sibling 'labels' or subfolder 'labels'
            p = os.path.dirname(d)
            cl_sib = os.path.join(p, "labels")
            cl_sub = os.path.join(d, "labels")
            if os.path.exists(cl_sib): self.lbl_dir = cl_sib
            elif os.path.exists(cl_sub): self.lbl_dir = cl_sub
            else: self.lbl_dir = d
            self.idx = 0; self._load()
            if self.paths: self._run()

    def _choose_image(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Image", self.img_dir, "Images (*.png *.jpg *.jpeg)")
        if f:
            d = os.path.dirname(f)
            self.img_dir = d
            p = os.path.dirname(d)
            cl_sib = os.path.join(p, "labels")
            cl_sub = os.path.join(d, "labels")
            if os.path.exists(cl_sib): self.lbl_dir = cl_sib
            elif os.path.exists(cl_sub): self.lbl_dir = cl_sub
            else: self.lbl_dir = d
            self._load()
            if f in self.paths: self.idx = self.paths.index(f)
            else: self.idx = 0
            self._run()

    def _cls4(self, p):
        b = os.path.splitext(os.path.basename(p))[0]; lp = os.path.join(self.lbl_dir, b + '.txt')
        if not os.path.exists(lp): return set()
        with open(lp) as f: lines = f.readlines()
        return {int(float(l.split()[0])) for l in lines if l.strip()}

    def _on_flt(self):
        cls = self.cls_cb.currentData()
        self.paths = (self.all_paths[:] if cls == -1 else [p for p in self.all_paths if cls in self._cls4(p)])
        self.idx = 0; self.sp_idx.setMaximum(max(1, len(self.paths))); self._uc()
        if self.paths: self._run()

    def _search(self):
        q = self.srch.text().strip()
        if not q: return
        t = self._ntp.get(q)
        if t is None:
            for pad in range(1, 5):
                zp = q.zfill(pad)
                if zp in self._ntp: t = self._ntp[zp]; break
        if t is None: self.status_msg.emit(f"'{q}' not found."); return
        if t not in self.paths: self.status_msg.emit(f"'{q}' filtered out."); return
        self.idx = self.paths.index(t); self.srch.clear(); self._run()

    def _prev(self):
        if self.idx > 0: self.idx -= 1; self._run()
    def _next(self):
        if self.idx < len(self.paths)-1: self.idx += 1; self._run()
    def _gos(self):
        i = self.sp_idx.value() - 1
        if 0 <= i < len(self.paths): self.idx = i; self._run()

    def _run(self):
        if not self.paths: return
        self._uc()
        if self._sub == self.SUB_DET: self._run_det()
        else: self._run_cls()

    def _run_det(self):
        model = self.mr[self._seg_idx]
        if not model: self.status_msg.emit(f"{SEGMENTATION_TITLES[self._seg_idx]} is not loaded."); return
        if self._wk and self._wk.isRunning(): return
        ip = self.paths[self.idx]; cf = self.cf_sl.value() / 100.0
        self.status_msg.emit(f"Inference on {os.path.basename(ip)}…")
        for pp in self.panels: pp._ph()
        self._wk_model_idx = self._seg_idx
        self._wk = InferenceWorker(self.mr, ip, cf, self._seg_idx)
        self._wk.finished.connect(self._done_det); self._wk.start()

    def _done_det(self, arrs):
        self._arrs = [None] * 5
        if arrs and arrs[0] is not None:
            self._arrs[0] = arrs[0][0]
        for i, data in enumerate(arrs[1:5], start=1):
            self._arrs[i] = data[0] if data is not None else None

        if self._arrs[0] is not None:
            self.input_panel.set_image(self._arrs[0], animated=False)
        else:
            self.input_panel._ph()

        for pp, data in zip(self.output_panels, arrs[1:5]):
            if data is not None:
                pp.set_image(data[0], data[1], data[2], data[3])
            else:
                pp._ph()
        self.output_panels[self._seg_idx].setVisible(True)
        fn = os.path.basename(self.paths[self.idx])
        self.nlbl.setText(fn); self.status_msg.emit(f"✓  {fn}")
        if self._wk_model_idx != self._seg_idx:
            self._run_det()

    def _run_cls(self):
        if not self.paths: return
        ip = self.paths[self.idx]; bgr = cv2.imread(ip)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._cls_orig.set_image(rgb)
        gt = self._cls4(ip)
        while self._gt_pills_lay.count() > 1:
            item = self._gt_pills_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        if gt:
            for cid in sorted(gt):
                name = CLASS_NAMES.get(cid, str(cid)); r, g, b = CLASS_COLORS_RGB.get(cid, (160,160,160))
                pill = QLabel(name)
                pill.setStyleSheet(f"background:rgba({r},{g},{b},0.15); color:rgb({max(0,r-40)},{max(0,g-40)},{max(0,b-40)}); border:1.5px solid rgba({r},{g},{b},0.4); border-radius:12px; padding:2px 12px; font-size:12px; font-weight:700; font-family:'Inter','Segoe UI',sans-serif;")
                self._gt_pills_lay.insertWidget(self._gt_pills_lay.count()-1, pill)
        else:
            no_lbl = QLabel("No label"); no_lbl.setStyleSheet(f"color:{P['muted']};font-size:12px;font-weight:500; background:transparent;border:none; font-family:'Inter','Segoe UI',sans-serif;")
            self._gt_pills_lay.insertWidget(0, no_lbl)

        self._rc_resnet.reset(); self._rc_cnn.reset()
        sel = self.cls_model_cb.currentData()
        rn  = self.cr[0] if sel in (0, 2) else None
        cnn = self.cr[1] if sel in (1, 2) else None
        if rn is None and cnn is None:
            self.status_msg.emit("Classification models not loaded yet."); return
        self.status_msg.emit(f"Classifying {os.path.basename(ip)}…")
        self._wk = ClassifyWorker(rn, cnn, ip); self._wk.finished.connect(self._done_cls); self._wk.start()

    def _done_cls(self, res):
        lr, cr = res.get("resnet", ("—", 0.)); lc, cc = res.get("cnn", ("—", 0.))
        self._rc_resnet.set_result(lr, cr); self._rc_cnn.set_result(lc, cc)
        fn = os.path.basename(self.paths[self.idx]); self.nlbl.setText(fn)
        self.status_msg.emit(f"✓  {fn}  ·  ResNet: {lr} ({cr*100:.1f}%)  ·  CNN: {lc} ({cc*100:.1f}%)")

    def _export(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.paths[self.idx]))[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export", os.path.join(SAVE_DIR, f"cmp_{stem}.png"), "PNG (*.png)")
        if not path: return
        if self._sub == self.SUB_DET:
            arrs = [a for a in self._arrs[1:] if a is not None]
            if not arrs: return
            th  = arrs[0].shape[0]
            row = np.hstack([cv2.resize(a,(int(a.shape[1]*th/a.shape[0]),th)) if a.shape[0]!=th else a for a in arrs])
            hdr = np.full((38, row.shape[1], 3), (253,240,234), dtype=np.uint8)
            labels = ["YOLO12n","YOLO26n","RF-DETR","DINOv3"]
            for i, l in enumerate(labels[:len(arrs)]):
                cv2.putText(hdr, l, (i*(row.shape[1]//len(arrs))+10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,50,40), 2)
            cv2.imwrite(path, cv2.cvtColor(np.vstack([hdr, row]), cv2.COLOR_RGB2BGR))
        else:
            if self._cls_orig._pix: self._cls_orig._pix.save(path)
        self.status_msg.emit(f"Exported → {path}")

    def _uc(self):
        total = len(self.paths); idx = self.idx+1 if total > 0 else 0
        self.ctr.setText(f"{idx} / {total}")
        self.b_prev.setEnabled(self.idx > 0)
        self.b_next.setEnabled(self.idx < total-1)
        self.sp_idx.setValue(idx)

    def keydown(self, k):
        if k == Qt.Key.Key_Right: self._next()
        elif k == Qt.Key.Key_Left: self._prev()
        elif k == Qt.Key.Key_E: self._export()


# ─────────────────────────────────────────────────────────────────────────────
#  ONLINE TAB
# ─────────────────────────────────────────────────────────────────────────────
class OnlineTab(QWidget):
    status_msg = pyqtSignal(str)
    MDL = {0:"YOLO12n-seg", 1:"YOLO26n-seg", 2:"RF-DETR", 3:"DINOv3"}

    def __init__(self, mr, sb, tb, parent=None):
        super().__init__(parent)
        self.mr = mr; self._sb = sb; self._tb = tb
        self._th = None; self._lf = None; self._streaming = False
        self._last_seg_result = None; self._cloud_image = None; self._cloud_metrics = None
        self._fusion_image = None; self._pcl_image = None; self._pcl_win = None
        self._sd = os.path.join(SAVE_DIR, "online"); self._lt = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._bsb(); self._btb(); self._bc()

    def _bsb(self):
        cam = self._sb.sec("Source Selection", open=True)

        self.src_cb = _combo()
        self.src_cb.addItem("Local Camera (V4L2)", 0)
        self.src_cb.addItem("ROS 2 Topic", 1)
        self.src_cb.currentIndexChanged.connect(self._on_src_changed)
        cam.add(self.src_cb)

        # Local Cam settings
        self.cam_fr = QFrame()
        cl = QHBoxLayout(self.cam_fr); cl.setContentsMargins(0,0,0,0)
        cl.addWidget(_lbl("Device ID", 11, 400, P["text2"]))
        self.cam_sp = _spin(); self.cam_sp.setMinimum(0); self.cam_sp.setValue(GELSIGHT_CAM)
        cl.addWidget(self.cam_sp); cam.add(self.cam_fr)

        # ROS 2 settings
        self.ros_fr = QFrame(); self.ros_fr.hide()
        rl = QVBoxLayout(self.ros_fr); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(_lbl("Topic Name", 11, 400, P["text2"]))
        self.topic_in = _input("/gs_mini_img")
        rl.addWidget(self.topic_in); cam.add(self.ros_fr)

        self.b_start = PillBtn("▶  Start", success=True); self.b_stop = PillBtn("■  Stop", danger=True)
        self.b_stop.setEnabled(False); self.b_start.clicked.connect(self._start); self.b_stop.clicked.connect(self._stop)
        cam.add(self.b_start); cam.add(self.b_stop)
        self.dot = _lbl("⬤  Not connected", 11, 500, P["muted"]); cam.add(self.dot)

        mdl = self._sb.sec("Detection model", open=True)
        self.mdl_cb = _combo(); [self.mdl_cb.addItem(v, k) for k, v in self.MDL.items()]
        self.mdl_cb.currentIndexChanged.connect(self._on_mdl); mdl.add(self.mdl_cb)
        out = self._sb.sec("Output view", open=True)
        self.view_cb = _combo()
        self.view_cb.addItem("Segmentation", 0)
        self.view_cb.addItem("Point cloud depth", 1)
        self.view_cb.addItem("Depth + segmentation", 2)
        self.view_cb.addItem("PCL normals (vectors)", 3)
        self.view_cb.currentIndexChanged.connect(self._on_view_changed)
        out.add(self.view_cb)

        cfs = self._sb.sec("Confidence", open=True)
        self.cf_big = _lbl(f"{int(CONF_THRESHOLD*100)}%", 28, 700, P["accent"])
        self.cf_big.setAlignment(Qt.AlignmentFlag.AlignCenter); cfs.add(self.cf_big)
        self.cf_sl = SmoothSlider(5, 95, int(CONF_THRESHOLD*100))
        self.cf_sl.valueChanged.connect(self._on_cf); cfs.add(self.cf_sl)

        ivl = self._sb.sec("Inference interval", open=False)
        self.ivl_sp = _spin(); self.ivl_sp.setMinimum(1); self.ivl_sp.setMaximum(60)
        self.ivl_sp.setValue(int(INFER_INTERVAL*10)); self.ivl_sp.setSuffix(" × 0.1s")
        ivl.add(self.ivl_sp); ivl.add(_lbl("Higher = less GPU load", 10, 400, P["muted"]))

        det = self._sb.sec("Detections", open=True)
        self._sbadges = {}
        for cid, name in CLASS_NAMES.items():
            b = PillBadge(name, CLASS_COLORS_RGB[cid]); det.add(b); self._sbadges[cid] = b
        self._sb.stretch()
        self.src_cb.setCurrentIndex(1)

    def _btb(self):
        t = self._tb.lay()
        t.addWidget(_lbl("Online", 12, 600, P["text"]))
        t.addWidget(_lbl("·", 12, 400, P["muted"]))
        self.fps_lbl = _lbl("— fps", 12, 400, P["muted"]); t.addWidget(self.fps_lbl)
        t.addStretch()
        self.cap_btn = PillBtn("Snapshot", accent=True); self.cap_btn.setFixedWidth(100)
        self.cap_btn.setEnabled(False); self.cap_btn.clicked.connect(self._cap); t.addWidget(self.cap_btn)
        self.export_pcd_btn = PillBtn("Export PCD + PFH/SHOT", accent=True)
        self.export_pcd_btn.setEnabled(False)
        self.export_pcd_btn.clicked.connect(self._export_pointclouds)
        t.addWidget(self.export_pcd_btn)
        self.pcl_win_btn = PillBtn("PCL window", accent=True)
        self.pcl_win_btn.clicked.connect(self._open_pcl_window)
        t.addWidget(self.pcl_win_btn)

    def _bc(self):
        lay = QHBoxLayout(self); lay.setContentsMargins(18,14,18,14); lay.setSpacing(12)
        self.raw_p = ImgPanel("GelSight Live Feed", P["muted"])
        self.res_p = ImgPanel("Output: YOLO12n-seg", P["accent"])
        lay.addWidget(self.raw_p, 3); lay.addWidget(self.res_p, 3)

        dc = GlassCard(14); dc.setFixedWidth(158)
        dcl = QVBoxLayout(dc); dcl.setContentsMargins(12,14,12,14); dcl.setSpacing(8)
        dcl.addWidget(_lbl("Detections", 11, 600, P["text2"])); dcl.addWidget(_div())
        self._dbadges = {}
        for cid, name in CLASS_NAMES.items():
            b = PillBadge(name, CLASS_COLORS_RGB[cid]); dcl.addWidget(b); self._dbadges[cid] = b
        dcl.addStretch(); lay.addWidget(dc)

    def _on_src_changed(self):
        is_ros = self.src_cb.currentData() == 1
        self.cam_fr.setVisible(not is_ros)
        self.ros_fr.setVisible(is_ros)
        self._cloud_image = None
        self._cloud_metrics = None
        self._fusion_image = None
        if hasattr(self, 'export_pcd_btn'):
            self.export_pcd_btn.setEnabled(
                self._streaming and is_ros and isinstance(self._th, ROS2Thread)
            )
        if self.view_cb.currentData() != 0:
            self._on_view_changed()

    def _start(self):
        if self._streaming: return
        midx = self.mdl_cb.currentData()
        if midx is None or self.mr[midx] is None:
            self.status_msg.emit("The selected model is not loaded."); return
        config.INFER_INTERVAL = self.ivl_sp.value() * 0.1
        
        models = (self.mr[0], self.mr[1], self.mr[2], self.mr[3])
        conf = self.cf_sl.value()/100.0
        
        if self.src_cb.currentData() == 0:
            self._th = GSThread(models, conf, midx, self.cam_sp.value())
        else:
            self._th = ROS2Thread(models, conf, midx, self.topic_in.text())
            
        self._th.frame_ready.connect(lambda r: [setattr(self,'_lf',r), self.raw_p.set_image(r, animated=False)])
        self._th.result_ready.connect(self._on_result_ready)
        if isinstance(self._th, ROS2Thread):
            self._th.cloud_ready.connect(self._on_cloud_ready)
            self._th.fusion_ready.connect(self._on_fusion_ready)
            self._th.pcl_ready.connect(self._on_pcl_ready)
            self._th.export_finished.connect(self._on_export_finished)
        self._th.det_classes.connect(self._on_det)
        self._th.status.connect(self._on_st)
        self._th.start(); self._streaming = True
        self.b_start.setEnabled(False); self.b_stop.setEnabled(True); self.cap_btn.setEnabled(True)
        self.export_pcd_btn.setEnabled(isinstance(self._th, ROS2Thread))
        self.dot.setText("⬤  Connecting…")
        self.dot.setStyleSheet(f"color:{P['amber']};font-size:11px;background:transparent;border:none;font-family:'Inter','Segoe UI',sans-serif;")

    def _on_result_ready(self, img, txt, col, metrics):
        self._last_seg_result = (img, txt, col, metrics)
        if self.view_cb.currentData() == 0:
            self.res_p._tlbl.setText(f"Output: {self.MDL[self.mdl_cb.currentData()]}")
            self.res_p.set_image(img, txt, col, metrics, animated=False)

    def _on_cloud_ready(self, img, metrics):
        self._cloud_image = img
        self._cloud_metrics = metrics
        if self.view_cb.currentData() == 1:
            self.res_p.set_image(
                img, "POINT CLOUD Z", (60, 170, 145), metrics, animated=False
            )

    def _on_fusion_ready(self, img):
        self._fusion_image = img
        if self.view_cb.currentData() == 2:
            if img is None:
                self.res_p._ph()
                return
            self.res_p.set_image(
                img, "DEPTH + SEGMENTATION", (60, 170, 145), animated=False
            )

    def _on_pcl_ready(self, img):
        self._pcl_image = img
        if self.view_cb.currentData() == 3 and img is not None:
            self.res_p.set_image(img, "PCL NORMALS", (60, 170, 145), animated=False)
        if self._pcl_win is not None and self._pcl_win.isVisible() and img is not None:
            self._show_pcl_window_image(img)

    def _show_pcl_window_image(self, img):
        from PyQt6.QtGui import QImage, QPixmap
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pcl_lbl.setPixmap(QPixmap.fromImage(qimg).scaled(
            self._pcl_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _open_pcl_window(self):
        from PyQt6.QtWidgets import QDialog
        if self._pcl_win is None:
            self._pcl_win = QDialog(self)
            self._pcl_win.setWindowTitle("PCL processing - normals")
            self._pcl_win.resize(1100, 460)
            lay = QVBoxLayout(self._pcl_win)
            self._pcl_lbl = QLabel("Waiting for synchronized point cloud...")
            self._pcl_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(self._pcl_lbl)
        self._pcl_win.show(); self._pcl_win.raise_()
        if self._pcl_image is not None:
            self._show_pcl_window_image(self._pcl_image)

    def _on_export_finished(self, message):
        if message:
            self.status_msg.emit(message)
        self.export_pcd_btn.setEnabled(
            self._streaming and isinstance(self._th, ROS2Thread)
        )

    def _export_pointclouds(self):
        if not isinstance(self._th, ROS2Thread):
            self.status_msg.emit("Start a ROS 2 stream with synchronized depth data first.")
            return
        output_root = os.path.join(SAVE_DIR, "pointclouds")
        if not self._th.request_export(output_root):
            self.status_msg.emit("A PCD export is already pending.")
            return
        self.export_pcd_btn.setEnabled(False)
        self.status_msg.emit("Export queued; waiting for a synchronized depth cloud.")

    def _on_view_changed(self):
        if self.view_cb.currentData() == 0:
            self.res_p._tlbl.setText(f"Output: {self.MDL[self.mdl_cb.currentData()]}")
            if self._last_seg_result is not None:
                self.res_p.set_image(*self._last_seg_result, animated=False)
            else:
                self.res_p._ph()
            return

        if self.view_cb.currentData() == 1:
            self.res_p._tlbl.setText("Output: Point cloud depth")
            if self._cloud_image is None:
                self.res_p._ph()
                self.status_msg.emit("Waiting for synchronized /gs_mini_pcd data.")
                return
            self.res_p.set_image(
                self._cloud_image, "POINT CLOUD Z", (60, 170, 145),
                self._cloud_metrics, animated=False
            )
            return

        if self.view_cb.currentData() == 3:
            self.res_p._tlbl.setText("Output: PCL normals")
            if self._pcl_image is None:
                self.res_p._ph()
                self.status_msg.emit("Waiting for image and point cloud with identical timestamps.")
                return
            self.res_p.set_image(self._pcl_image, "PCL NORMALS", (60, 170, 145), animated=False)
            return

        self.res_p._tlbl.setText("Output: Depth + segmentation")
        if self._fusion_image is None:
            self.res_p._ph()
            self.status_msg.emit(
                "Waiting for image and point cloud with identical timestamps."
            )
            return
        self.res_p.set_image(
            self._fusion_image, "DEPTH + SEGMENTATION", (60, 170, 145),
            animated=False
        )

    def _stop(self):
        if self._th: self._th.stop(); self._th.wait(3000); self._th = None
        self._streaming = False
        self._last_seg_result = None; self._cloud_image = None; self._cloud_metrics = None
        self._fusion_image = None; self._pcl_image = None
        self.b_start.setEnabled(True); self.b_stop.setEnabled(False); self.cap_btn.setEnabled(False)
        self.export_pcd_btn.setEnabled(False)
        self.dot.setText("⬤  Stopped")
        self.dot.setStyleSheet(f"color:{P['muted']};font-size:11px;background:transparent;border:none;font-family:'Inter','Segoe UI',sans-serif;")
        self.raw_p._ph(); self.res_p._ph(); self._on_det(set())

    def _on_cf(self, v):
        self.cf_big.setText(f"{v}%")
        if self._th: self._th.set_conf(v/100.0)

    def _on_mdl(self):
        i = self.mdl_cb.currentData()
        if i is None: return
        if self.view_cb.currentData() == 0:
            self.res_p._tlbl.setText(f"Output: {self.MDL[i]}"); self.res_p._ph()
        elif self.view_cb.currentData() == 2:
            self._fusion_image = None
            self._on_view_changed()
        if self._th: self._th.set_model(i)

    def _on_st(self, msg):
        self.status_msg.emit(msg)
        if msg.startswith("ERROR"):
            self.dot.setText("⬤  Error")
            self.dot.setStyleSheet(f"color:{P['red']};font-size:11px;background:transparent;border:none;")
            self._stop()
        elif "connected" in msg.lower():
            self.dot.setText("⬤  Streaming")
            self.dot.setStyleSheet(f"color:{P['green']};font-size:11px;background:transparent;border:none;")

    def _on_det(self, det):
        now = time.time()
        if self._lt > 0: self.fps_lbl.setText(f"{1/max(now-self._lt,1e-3):.1f} fps")
        self._lt = now
        for cid, b in self._sbadges.items(): b.set_active(cid in det)
        for cid, b in self._dbadges.items(): b.set_active(cid in det)

    def _cap(self):
        if self._lf is None: return
        os.makedirs(self._sd, exist_ok=True)
        path = os.path.join(self._sd, f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(path, cv2.cvtColor(self._lf, cv2.COLOR_RGB2BGR))
        self.status_msg.emit(f"Saved → {path}")

    def stop_if_running(self):
        if self._streaming: self._stop()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Detector Viewer")
        self.setMinimumSize(1440, 860)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._models = [None, None, None, None]; self._cls = [None, None]
        self._build(); self._load_models()

    def _build(self):
        c = QWidget(); c.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground); self.setCentralWidget(c)
        self._bg = AnimatedBG(c)
        vl = QVBoxLayout(c); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)
        self._toff = Topbar(); self._ton = Topbar()
        vl.addWidget(self._toff); vl.addWidget(self._ton); self._ton.hide()
        bl = QHBoxLayout(); bl.setContentsMargins(0,0,0,0); bl.setSpacing(0)
        self._soff = Sidebar(); self._son = Sidebar()
        self._soff.set_callbacks(lambda:self._sw(0), lambda:self._sw(1))
        self._son.set_callbacks( lambda:self._sw(0), lambda:self._sw(1))
        bl.addWidget(self._soff); bl.addWidget(self._son); self._son.hide()
        self._coff = QWidget(); self._coff.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._con  = QWidget(); self._con.setAttribute( Qt.WidgetAttribute.WA_TranslucentBackground)
        bl.addWidget(self._coff, 1); bl.addWidget(self._con, 1); self._con.hide()
        vl.addLayout(bl, 1)

        self.sb = QStatusBar(); self.setStatusBar(self.sb)
        self.sb.setStyleSheet(f"background:rgba(255,246,242,0.92);color:{P['muted']};font-size:11px;border-top:1px solid {P['border']};font-family:'Inter','Segoe UI',sans-serif;")
        self._slbl = QLabel("● Loading models…")
        self._slbl.setStyleSheet(f"color:{P['amber']};font-size:11px;margin-right:10px;font-family:'Inter','Segoe UI',sans-serif;")
        self.sb.addPermanentWidget(self._slbl)
        self._spb = QProgressBar(); self._spb.setRange(0,0); self._spb.setFixedSize(100,5)
        self._spb.setStyleSheet(f"QProgressBar{{background:{P['border']};border:none;border-radius:3px;}}QProgressBar::chunk{{background:{P['accent']};border-radius:3px;}}")
        self.sb.addPermanentWidget(self._spb)

        self.off_tab = OfflineTab(self._models, self._cls, self._soff, self._toff)
        ol = QVBoxLayout(self._coff); ol.setContentsMargins(0,0,0,0); ol.addWidget(self.off_tab)
        self.off_tab.status_msg.connect(self.sb.showMessage)

        self.on_tab = OnlineTab(self._models, self._son, self._ton)
        nl = QVBoxLayout(self._con); nl.setContentsMargins(0,0,0,0); nl.addWidget(self.on_tab)
        self.on_tab.status_msg.connect(self.sb.showMessage)

    def resizeEvent(self, e):
        super().resizeEvent(e); self._bg.setGeometry(0, 0, self.width(), self.height())

    def _sw(self, idx):
        pairs = [(self._soff, self._toff, self._coff), (self._son, self._ton, self._con)]
        for i, (s, t, c) in enumerate(pairs):
            if i == idx: s.show(); t.show(); c.show()
            else: s.hide(); t.hide(); c.hide()

    def _load_models(self):
        self._ldr = ModelLoader()
        self._ldr.progress.connect(lambda m: self._slbl.setText(f"● {m}"))
        self._ldr.finished.connect(self._on_loaded)
        self._ldr.error.connect(self._on_err)
        self._ldr.start()
        self._cldr = ClassifyLoader()
        self._cldr.progress.connect(lambda m: self.sb.showMessage(m))
        self._cldr.finished.connect(self._on_cls_loaded)
        self._cldr.start()

    def _on_loaded(self, y12, y26, rf, dino):
        self._models[0]=y12; self._models[1]=y26; self._models[2]=rf; self._models[3]=dino
        available = sum(model is not None for model in self._models)
        if available == len(self._models):
            self._slbl.setText("● Models ready")
            self._slbl.setStyleSheet(f"color:{P['green']};font-size:11px;margin-right:10px;font-family:'Inter','Segoe UI',sans-serif;")
        else:
            self._slbl.setText(f"● {available}/4 models ready")
            self._slbl.setStyleSheet(f"color:{P['amber']};font-size:11px;margin-right:10px;font-family:'Inter','Segoe UI',sans-serif;")
        online_model = self.on_tab.mdl_cb.model()
        first_available = None
        for index, model in enumerate(self._models):
            item = online_model.item(index)
            if item is not None:
                item.setEnabled(model is not None)
            if model is not None and first_available is None:
                first_available = index
        if first_available is not None and self._models[self.on_tab.mdl_cb.currentData()] is None:
            self.on_tab.mdl_cb.setCurrentIndex(first_available)
        self._spb.setRange(0,1); self._spb.setValue(1)
        errors = getattr(self._ldr, "load_errors", {})
        if errors:
            self.sb.showMessage("Unavailable models: " + ", ".join(errors))
        else:
            self.sb.showMessage("Detection models loaded — ready.")
        if self.off_tab.paths: self.off_tab._run()

    def _on_cls_loaded(self, rn, cnn):
        self._cls[0]=rn; self._cls[1]=cnn; parts = []
        if rn:  parts.append("ResNet18 ✓")
        if cnn: parts.append("CNN ✓")
        self.sb.showMessage("Classification: " + ("  |  ".join(parts) if parts else "models not found"))

    def _on_err(self, msg):
        self._slbl.setText(f"● Error: {msg[:50]}")
        self._slbl.setStyleSheet(f"color:{P['red']};font-size:11px;margin-right:10px;")
        self._spb.setRange(0,1)
        self.sb.showMessage(f"Error: {msg}")

    def keyPressEvent(self, e):
        k = e.key()
        if self._coff.isVisible(): self.off_tab.keydown(k)
        if k == Qt.Key.Key_F and e.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.off_tab.srch.setFocus(); self.off_tab.srch.selectAll()
        super().keyPressEvent(e)

    def closeEvent(self, e):
        self.on_tab.stop_if_running(); super().closeEvent(e)


if __name__ == '__main__':
    app = QApplication(sys.argv); app.setStyle('Fusion')
    for v in ["Inter-Regular","Inter-Medium","Inter-SemiBold","Inter-Bold"]:
        QFontDatabase.addApplicationFont(f"{v}.ttf")
    app.setFont(QFont("Inter", 11))
    pal = app.palette(); pal.setColor(QPalette.ColorRole.Window, QColor(0,0,0,0)); app.setPalette(pal)
    win = MainWindow(); win.show(); sys.exit(app.exec())