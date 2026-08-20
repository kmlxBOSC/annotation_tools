"""
Image Annotation Desktop Application
=====================================
PyQt6 + OpenCV tool for annotating images with bounding boxes and
mathematically interpolated keypoints.

Canvas backend: QGraphicsView / QGraphicsScene
  - Scene coordinates == image pixel coordinates (1-to-1 mapping).
  - Annotation lines/rects use cosmetic pens (always 1 screen-pixel wide).
  - Keypoint circles use ItemIgnoresTransformations (fixed screen-pixel size).
  - Crosshairs are two cosmetic QGraphicsLineItems spanning the scene rect.
  - Ctrl+Scroll zooms, pivoting under the mouse (AnchorUnderMouse).
  - Ctrl+F fits the image back into the viewport.
  - Scrollbars appear automatically when zoomed in.

UI layout (LabelMe-style)
  - Top QMenuBar (File / Edit / View) holds all general actions.
  - A slim, spacious side dock holds ONLY: tool selection + Bat settings,
    the Edit/Resize toggle, and the smart image file list.

Folder-driven workflow
  - File ▸ Open Image Dir   loads every image in a folder into the list.
  - File ▸ Open Labels Dir  (optional) points the loader at a folder of
    YOLO .txt or LabelMe .json files.
  - The image list shows a status checkbox per image; it is ticked when a
    matching label file is found.  Selecting an image auto-loads its label.

Edit / Resize mode
  - OFF (default): click-drag draws new annotations.
  - ON: drag keypoints to move them; drag a box edge/corner to resize it, or
    drag inside a box to move the whole box.

Mode:
  Bat is the only mode — a bounding box plus interpolated keypoints.

Bat workflow (bbox is mandatory):
  1. draw bbox — drag or two corner-clicks  (skipped if one already exists)
  2. click TOP of object
  3. click TIP — n keypoints auto-interpolate evenly along the straight line
  4. drag the red anchor markers to fine-tune
  Deleting the bbox resets the object and forces a fresh bbox before keypoints.

Per-keypoint visibility (0 = not visible · 1 = occluded · 2 = fully visible):
  - right-click any keypoint → pick a value, or
  - use the "Keypoint Visibility" panel in the sidebar, or
  - select a keypoint and press 0 / 1 / 2.

Outputs (saved simultaneously, base filename == image name):
  <parent>/labels/<name>.txt       YOLO-pose format, normalised coords
  <parent>/json_labels/<name>.json LabelMe format, absolute pixel coords

Keys: A/D prev/next · Space skip · Backspace delete selected · Delete delete all
      Ctrl+C copy · Ctrl+V paste · Ctrl+E edit · Ctrl+S save · Ctrl+Z undo
      Ctrl+Y redo · Ctrl+F fit · B zoom-to-bbox · ESC cancel · Ctrl+Scroll zoom
      Shift+left-drag pan (hand/grab)
      Shift+Scroll horizontal pan · Scroll vertical pan
"""

import json
import os
import sys

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import (QAction, QActionGroup, QBrush, QColor, QFont, QImage,
                         QKeySequence, QPainter, QPainterPath, QPen, QPixmap,
                         QShortcut)
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QDialog, QFileDialog, QFormLayout,
    QFrame, QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, 
    QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QVBoxLayout, QWidget,
)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# A mouse move smaller than this (viewport pixels) is treated as a click.
CLICK_TOLERANCE = 6
# Hit radius (viewport pixels) for grabbing a draggable bat anchor.
ANCHOR_HIT_RADIUS = 12
# Edit-mode hit radii (viewport pixels).
KP_HIT_RADIUS     = 12   # grab a keypoint / anchor
CORNER_HIT_RADIUS = 10   # grab a box corner handle
EDGE_HIT_RADIUS   = 7    # grab a box edge
# Zoom multiplier per Ctrl+Scroll notch.
ZOOM_FACTOR = 1.15
# Screen-pixel radii for keypoint markers (scale-invariant).
POINT_RADIUS  = 5
ANCHOR_RADIUS = 8
# Screen-pixel half-size of an edit handle square.
HANDLE_HALF = 4
# Bounding-box pen width in screen pixels (cosmetic — constant at any zoom).
BOX_PEN_WIDTH = 4
# Per-keypoint visibility palette (the marker's outer RING reflects the
# stored flag; this matches the sidebar legend).
VIS_COLORS = {
    2: QColor(52, 199, 89),    # fully visible  — iOS green
    1: QColor(255, 159, 10),   # occluded       — iOS orange
    0: QColor(142, 142, 147),  # not visible    — iOS gray
}
# Selection halo colour (iOS blue).
SELECT_COLOR = QColor(10, 132, 255)

# Distinct per-keypoint identity colours (the marker's inner DISC).  Chosen to
# stay legible over complex/cluttered backgrounds and to avoid the green /
# orange / gray used by the visibility rings.  KP1 = red, KP2 = blue,
# KP3 = brown, … then the palette cycles.
INDEX_COLORS = [
    QColor("#FF3B30"),  # 1  red
    QColor("#1E90FF"),  # 2  blue
    QColor("#B5651D"),  # 3  brown
    QColor("#FFD60A"),  # 4  yellow
    QColor("#AF52DE"),  # 5  purple
    QColor("#FF2D95"),  # 6  magenta
    QColor("#00C7BE"),  # 7  teal
    QColor("#5AC8FA"),  # 8  sky blue
    QColor("#FF7AB6"),  # 9  pink
    QColor("#5E5CE6"),  # 10 indigo
    QColor("#D4A017"),  # 11 goldenrod
    QColor("#64D2FF"),  # 12 cyan
]


def color_for_index(i):
    """Identity colour for the 0-based keypoint index (cycles the palette)."""
    return INDEX_COLORS[i % len(INDEX_COLORS)]


# ---------------------------------------------------------------------------
# Interpolation math (equal spacing only)
# ---------------------------------------------------------------------------
def default_fractions(n):
    """Parametric positions t∈[0,1] of the n keypoints, evenly spaced.

    t_i = i/(n-1), so for n=5 the keypoints sit at 0 %, 25 %, 50 %, 75 %,
    100 % of the Top→Tip segment.
    """
    if n <= 1:
        return [0.0]
    return [i / (n - 1) for i in range(n)]


def inner_fractions(n):
    """The fractions of the intermediate (non-endpoint) keypoints only."""
    return default_fractions(n)[1:-1] if n >= 2 else []


def points_from_fractions(p_top, p_tip, fractions):
    """Map parametric positions t∈[0,1] to absolute points on Top→Tip."""
    dx, dy = p_tip[0] - p_top[0], p_tip[1] - p_top[1]
    return [[p_top[0] + t * dx, p_top[1] + t * dy] for t in fractions]


def interpolate_line(p_top, p_tip, n):
    """n evenly spaced points on the segment p_top → p_tip (equal spacing)."""
    return points_from_fractions(p_top, p_tip, default_fractions(n))


# ---------------------------------------------------------------------------
# Scale-invariant keypoint marker
# ---------------------------------------------------------------------------
class ScaleInvariantPoint(QGraphicsItem):
    """Keypoint circle that stays a fixed size in screen pixels.

    ItemIgnoresTransformations strips the view's zoom from the item's local
    coordinate system at paint time.  setPos() still uses scene (= image
    pixel) coordinates, so the marker tracks the correct image location at
    any zoom level while the drawn radius never changes.
    """

    def __init__(self, index, color, is_anchor=False, vis=2, selected=False,
                 parent=None):
        super().__init__(parent)
        self.index    = index
        self.color    = color
        self.vis      = int(vis)
        self.selected = bool(selected)
        self.r        = float(ANCHOR_RADIUS if is_anchor else POINT_RADIUS)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setZValue(21 if selected else 20)

    def boundingRect(self):
        # Local coords are screen pixels due to ItemIgnoresTransformations.
        # Extra margin for the selection halo and the index label.
        m = self.r + 6.0
        return QRectF(-m, -m, 2 * m + 22, 2 * m + 8)

    def paint(self, painter, option, widget=None):
        r = self.r
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Selection halo — a soft blue ring around the marker.
        if self.selected:
            halo = QPen(SELECT_COLOR, 2.5)
            painter.setPen(halo)
            painter.setBrush(QBrush())
            painter.drawEllipse(QPointF(0.0, 0.0), r + 4.0, r + 4.0)

        # The border colour encodes the visibility flag (0/1/2); a "not
        # visible" keypoint is also drawn dashed and muted so it reads at a
        # glance without needing to open the panel.
        vis_color = VIS_COLORS.get(self.vis, VIS_COLORS[2])
        if self.vis == 0:
            border = QPen(vis_color, 2.0, Qt.PenStyle.DashLine)
            fill   = QColor(self.color.red(), self.color.green(),
                            self.color.blue(), 90)
        elif self.vis == 1:
            border = QPen(vis_color, 2.0)
            fill   = self.color
        else:
            border = QPen(vis_color, 2.0)
            fill   = self.color
        painter.setPen(border)
        painter.setBrush(fill)
        painter.drawEllipse(QPointF(0.0, 0.0), r, r)

        # Index label, with a subtle dark outline for legibility on any bg.
        painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        painter.setPen(QColor(0, 0, 0, 160))
        painter.drawText(QPointF(r + 3.0, 5.0), str(self.index))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QPointF(r + 2.0, 4.0), str(self.index))
        painter.setBrush(QBrush())


# ---------------------------------------------------------------------------
# Scale-invariant box resize handle (drawn only in Edit mode)
# ---------------------------------------------------------------------------
class HandleItem(QGraphicsItem):
    """Small filled square at a box corner/edge, fixed size in screen pixels.

    Purely a visual affordance — hit-testing is done independently in the
    view's mouse handlers (in viewport coordinates) so it works whether or
    not the cursor lands exactly on the painted square.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setZValue(22)

    def boundingRect(self):
        m = HANDLE_HALF + 1.0
        return QRectF(-m, -m, 2 * m, 2 * m)

    def paint(self, painter, option, widget=None):
        painter.setPen(QPen(QColor(0, 0, 0), 1.0))
        painter.setBrush(QColor(255, 255, 255))
        painter.drawRect(QRectF(-HANDLE_HALF, -HANDLE_HALF,
                                2 * HANDLE_HALF, 2 * HANDLE_HALF))
        painter.setBrush(QBrush())


# ---------------------------------------------------------------------------
# Annotation canvas — QGraphicsView
# ---------------------------------------------------------------------------
class AnnotationView(QGraphicsView):
    """Image canvas with zoom/pan, crosshairs, drawing and editing.

    The QGraphicsScene uses image-pixel coordinates so that
    mapToScene(viewport_pos) directly yields image pixel coordinates —
    no manual scale/offset arithmetic needed.

    Annotation state lives in self.history (action-replay model).
    _rebuild_state() replays history → data fields, then _rebuild_scene()
    pushes the result into QGraphicsItems.

    Edit mode mutates the derived data fields directly during a drag and,
    on release, rewrites history to a canonical equivalent so undo/redo and
    saving stay consistent.
    """

    def __init__(self, status_cb, parent=None):
        super().__init__(parent)
        self.status_cb = status_cb
        # Optional no-arg callback fired when the annotation *structure*
        # changes (keypoints added/removed/visibility), so the sidebar
        # keypoint panel can refresh.  Set by MainWindow.
        self.state_cb  = None
        # Navigation hooks wired up by MainWindow for the canvas key handler.
        self.request_prev = None
        self.request_next = None
        self.request_skip = None
        # Copy/paste hooks (Ctrl+C / Ctrl+V) wired up by MainWindow.
        self.request_copy = None
        self.request_paste = None

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        # The canvas must accept focus so single-key shortcuts (A/D/Space/
        # Backspace/0-1-2) reach keyPressEvent without disturbing text fields.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # ── view configuration ──────────────────────────────────────────
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # Zoom pivot is computed manually in wheelEvent using viewport-local
        # coordinates. AnchorUnderMouse relies on QCursor::pos() (global
        # desktop coords), which can drift out of sync with the viewport on
        # multi-monitor / HiDPI setups and made the zoom pivot jump around.
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(35, 35, 40))
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        # ── persistent scene items (survive _rebuild_scene calls) ────────
        # Crosshair: two lines with a cosmetic pen = constant screen width at
        # any zoom.  Dark black and a bit thicker for clear visibility.
        _ch_pen = QPen(QColor(0, 0, 0), 2.0)
        _ch_pen.setCosmetic(True)
        self._crosshair_h = QGraphicsLineItem()
        self._crosshair_v = QGraphicsLineItem()
        for _ch in (self._crosshair_h, self._crosshair_v):
            _ch.setPen(QPen(_ch_pen))
            _ch.setZValue(30)
            _ch.setVisible(False)
            self._scene.addItem(_ch)

        # Rubber-band: drag-draw preview and two-click corner preview.
        # Solid line (was dashed) to match the committed bounding-box style.
        _rb_pen = QPen(QColor(0, 220, 90), BOX_PEN_WIDTH, Qt.PenStyle.SolidLine)
        _rb_pen.setCosmetic(True)
        self._rubber_rect = QGraphicsRectItem()
        self._rubber_rect.setPen(_rb_pen)
        self._rubber_rect.setBrush(QBrush())
        self._rubber_rect.setZValue(25)
        self._rubber_rect.setVisible(False)
        self._scene.addItem(self._rubber_rect)

        self._pixmap_item      = None   # QGraphicsPixmapItem for the image
        self._annotation_items = []     # items created by _rebuild_scene

        # ── annotation state ─────────────────────────────────────────────
        self.img_w = 0
        self.img_h = 0
        # Bat is the only annotation mode.  A bounding box is mandatory:
        # keypoints cannot be placed until one exists, and deleting it forces a
        # fresh box (see spec 13/14).
        self.bat_n        = 3
        self.edit_mode    = False
        # Absolute (x, y) positions of the intermediate Bat keypoints (len == n-2).
        # Stored as free coordinates so KP2 can be placed anywhere off-axis.
        self.bat_inner_pts = []
        # Per-keypoint visibility for Bat keypoints (len == bat_n), independent
        # of the general-keypoint visibilities stored inline in self.points.
        self.bat_vis      = []
        # YOLOv8-pose visibility default applied to newly drawn keypoints.
        self.kp_visibility = 2

        # Currently selected annotation (for Backspace / 0-1-2). One of:
        #   ('point', i) | ('bat_kp', i) | ('box', ('gen', i)) | ('box', ('bat',))
        self._selected  = None

        self.history    = []
        self.redo_stack = []
        # One-shot snapshot taken by delete_all() so a single Ctrl+Z restores
        # every annotation at once.  Reset whenever a new action is recorded or
        # a different image is loaded (undo is per-frame only).
        self._cleared_snapshot = None

        # Derived from history; rebuilt by _rebuild_state().
        self.boxes          = []
        self.points         = []
        self.bat_box        = None
        self.bat_top        = None
        self.bat_tip        = None
        self.pending_corner = None

        # Transient mouse state.
        self._press_vp       = None   # viewport QPoint at mouse press
        self._press_scene_pt = None   # scene [x,y] at mouse press
        self._drag_anchor    = None   # 'bat_top' / 'bat_tip' / None
        self._cursor_scene   = None   # last cursor scene [x, y]

        # Pan state: Shift+left-drag moves the view like a hand/grab tool,
        # in every mode, without disturbing draw/edit (which use plain left).
        self._panning        = False
        self._pan_last_vp    = None   # viewport QPoint of the last pan step

        # Edit-mode grab descriptor and last scene pos (for box moves).
        # ('point', i) | ('anchor', name) | ('box', ref, handle)
        #   ref:    ('gen', i) | ('bat',)
        #   handle: 'move' | 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw'
        self._edit_target     = None
        self._edit_last_scene = None
        self._edit_moved      = False

        # True while a drag is mid-flight, so _rebuild_state suppresses the
        # (relatively heavy) sidebar refresh until the drag commits.
        self._dragging        = False
        # History snapshot taken at the start of a drag so ESC can abort it.
        self._drag_snapshot   = None

        self.dirty = False

    # ── image loading ────────────────────────────────────────────────────

    def load_image(self, path):
        """Load image via OpenCV (np.fromfile handles Unicode paths on Win)."""
        data = np.fromfile(path, dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            self.status_cb(f"Failed to load {path}")
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        qimg   = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        # Replace the pixmap item without using scene.clear() (that would
        # also remove the persistent crosshair and rubber-band items).
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.addItem(self._pixmap_item)

        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self.img_w, self.img_h = w, h
        self.clear_annotations()
        self.fit_to_screen()
        self._update_idle_cursor()
        return True

    def clear_annotations(self):
        self.history    = []
        self.redo_stack = []
        self.dirty      = False
        self._selected  = None
        self._cleared_snapshot = None
        self.bat_inner_pts = [] 
        self.bat_vis = []
        self._rebuild_state()

    def set_history(self, actions):
        """Install a pre-built action list (used when loading saved files)."""
        self.history    = list(actions)
        self.redo_stack = []
        self.dirty      = False
        self._selected  = None
        self._cleared_snapshot = None
        self._rebuild_state()

    def set_edit_mode(self, on):
        self.edit_mode = bool(on)
        # Drop any in-progress grab and clear transient previews.
        self._edit_target = None
        self._drag_anchor = None
        self._selected    = None
        self._rubber_rect.setVisible(False)
        if self.edit_mode:
            self._crosshair_h.setVisible(False)
            self._crosshair_v.setVisible(False)
        self._update_idle_cursor()
        self._rebuild_scene()

    # ── state / scene management ─────────────────────────────────────────

    def _rebuild_state(self):
        """Replay history → update data fields → synchronise scene items."""
        self.boxes          = []
        self.points         = []
        self.bat_box        = None
        self.bat_top        = None
        self.bat_tip        = None
        self.pending_corner = None

        for kind, data in self.history:
            if kind == 'point':
                self.points.append(data)
            elif kind == 'bat_top':
                self.bat_top = data
            elif kind == 'bat_tip':
                self.bat_tip = data
            elif kind == 'box':
                self._commit_box(data)
            elif kind == 'corner':
                if self.pending_corner is None:
                    self.pending_corner = data
                else:
                    x1, y1 = self.pending_corner
                    x2, y2 = data
                    self._commit_box([min(x1, x2), min(y1, y2),
                                      max(x1, x2), max(y1, y2)])
                    self.pending_corner = None

        self._sync_bat_vis()
        self._rebuild_scene()
        self._notify_state()

    def _notify_state(self):
        """Refresh the sidebar keypoint panel, unless a drag is in flight."""
        if self.state_cb is not None and not self._dragging:
            self.state_cb()

    def _rebuild_scene(self):
        """Remove all annotation QGraphicsItems and recreate from state.

        Cosmetic pens: width=1 screen pixel at every zoom level.
        ScaleInvariantPoint: fixed screen-pixel radius via
        ItemIgnoresTransformations.  In Edit mode, each box also gets eight
        scale-invariant handle squares.
        """
        for item in self._annotation_items:
            self._scene.removeItem(item)
        self._annotation_items.clear()

        if self._pixmap_item is None:
            return

        def add(item):
            self._scene.addItem(item)
            self._annotation_items.append(item)

        def cpen(color, style=Qt.PenStyle.SolidLine, width=1.0):
            """Return a cosmetic pen (constant screen-pixel width at any zoom)."""
            p = QPen(color, width, style)
            p.setCosmetic(True)
            return p

        # Thicker, more distinct bounding boxes (cosmetic → constant on screen).
        box_pen = cpen(QColor(0, 220, 90), width=BOX_PEN_WIDTH)
        sel_pen = cpen(SELECT_COLOR, width=BOX_PEN_WIDTH + 1)

        def add_box(box, selected=False):
            r = QGraphicsRectItem(
                QRectF(box[0], box[1], box[2] - box[0], box[3] - box[1]))
            r.setPen(sel_pen if selected else box_pen)
            if selected:
                r.setBrush(QBrush(QColor(10, 132, 255, 28)))
            else:
                r.setBrush(QBrush())
            r.setZValue(6 if selected else 5)
            add(r)
            if self.edit_mode:
                self._add_box_handles(box, add)

        # General-mode bounding boxes.
        for i, box in enumerate(self.boxes):
            add_box(box, selected=(self._selected == ('box', ('gen', i))))

        # Bat manual bounding box.
        if self.bat_box is not None:
            add_box(self.bat_box, selected=(self._selected == ('box', ('bat',))))

        # Bat axis: curved dashed line through all keypoints.
        if self.bat_top is not None and self.bat_tip is not None:
            self._sync_inner_pts()
            pen = cpen(QColor(80, 160, 255), Qt.PenStyle.DashLine)
            all_pts = [self.bat_top] + self.bat_inner_pts + [self.bat_tip]
            n_pts = len(all_pts)
            path = QPainterPath()
            path.moveTo(all_pts[0][0], all_pts[0][1])
            if n_pts == 2:
                path.lineTo(all_pts[1][0], all_pts[1][1])
            elif n_pts == 3:
                # Quadratic Bézier that passes exactly through the middle point.
                p0, p1, p2 = all_pts
                cx = 2 * p1[0] - (p0[0] + p2[0]) / 2
                cy = 2 * p1[1] - (p0[1] + p2[1]) / 2
                path.quadTo(cx, cy, p2[0], p2[1])
            else:
                # Catmull-Rom spline (passes through every point).
                for i in range(n_pts - 1):
                    p0 = all_pts[max(i - 1, 0)]
                    p1 = all_pts[i]
                    p2 = all_pts[i + 1]
                    p3 = all_pts[min(i + 2, n_pts - 1)]
                    cp1x = p1[0] + (p2[0] - p0[0]) / 6
                    cp1y = p1[1] + (p2[1] - p0[1]) / 6
                    cp2x = p2[0] - (p3[0] - p1[0]) / 6
                    cp2y = p2[1] - (p3[1] - p1[1]) / 6
                    path.cubicTo(cp1x, cp1y, cp2x, cp2y, p2[0], p2[1])
            ln = QGraphicsPathItem(path)
            ln.setPen(pen)
            ln.setZValue(8)
            add(ln)

        # General-mode keypoints — each gets a distinct identity colour.
        for i, pt in enumerate(self.points):
            vis = pt[2] if len(pt) > 2 else self.kp_visibility
            p = ScaleInvariantPoint(
                i + 1, color_for_index(i), vis=vis,
                selected=(self._selected == ('point', i)))
            p.setPos(pt[0], pt[1])
            add(p)

        # Bat keypoints — anchors drawn larger; colour is per-index identity.
        kps = self.bat_keypoints()
        if kps is not None:
            self._sync_bat_vis()
            for i, pt in enumerate(kps):
                is_anchor = (i == 0 or i == len(kps) - 1)
                vis = self.bat_vis[i] if i < len(self.bat_vis) else 2
                p = ScaleInvariantPoint(
                    i + 1, color_for_index(i), is_anchor=is_anchor, vis=vis,
                    selected=(self._selected == ('bat_kp', i)))
                p.setPos(pt[0], pt[1])
                add(p)
        elif self.bat_top is not None:
            # Only the first anchor has been placed yet.
            vis = self.bat_vis[0] if self.bat_vis else 2
            p = ScaleInvariantPoint(
                1, color_for_index(0), is_anchor=True, vis=vis,
                selected=(self._selected == ('bat_kp', 0)))
            p.setPos(self.bat_top[0], self.bat_top[1])
            add(p)

    def _add_box_handles(self, box, add):
        """Place eight handle squares (4 corners + 4 edge midpoints)."""
        x1, y1, x2, y2 = box
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2),
                       (mx, y1), (mx, y2), (x1, my), (x2, my)):
            h = HandleItem()
            h.setPos(hx, hy)
            add(h)

    def _commit_box(self, box):
        # Bat is the only mode: a committed box is always the Bat bounding box.
        self.bat_box = box

    def _record(self, kind, data):
        self.history.append((kind, data))
        self.redo_stack = []
        # A fresh action supersedes a pending delete-all; drop its snapshot so
        # Ctrl+Z follows the new action chain instead of restoring stale state.
        self._cleared_snapshot = None
        self.dirty = True
        self._rebuild_state()

    # ── undo / redo / cancel ───────────────────────────────────────────────

    def undo(self):
        """Ctrl+Z: pop the last committed action onto the redo stack."""
        if not self.history:
            # A delete-all is restored in one step from its snapshot.
            if self._cleared_snapshot is not None:
                self.history = list(self._cleared_snapshot)
                self._cleared_snapshot = None
                self.redo_stack = []
                self.dirty = True
                self._selected = None
                self._rebuild_state()
                self.status_cb("Restored all annotations (undo delete-all)")
                return
            self.status_cb("Nothing to undo")
            return
        self.redo_stack.append(self.history.pop())
        self.dirty = True
        self._selected = None
        self._rebuild_state()
        self.status_cb("Undo (Ctrl+Z) — Ctrl+Y to redo")

    def redo(self):
        """Ctrl+Y: restore the most recently undone action."""
        if not self.redo_stack:
            self.status_cb("Nothing to redo")
            return
        self.history.append(self.redo_stack.pop())
        self.dirty = True
        self._rebuild_state()
        self.status_cb("Redo (Ctrl+Y)")

    def cancel_operation(self):
        """ESC: cancel the operation in progress WITHOUT deleting committed
        annotations.  Aborts an active drag, a half-drawn two-click bbox, or a
        rubber-band drag — in that priority order."""
        # 1. Abort an in-flight drag (anchor / edit) by restoring the snapshot.
        if self._drag_snapshot is not None and \
                (self._drag_anchor or self._edit_target):
            self.history    = list(self._drag_snapshot)
            self.redo_stack = []
            self._drag_anchor = None
            self._edit_target = None
            self._drag_snapshot = None
            self._dragging  = False
            self._rubber_rect.setVisible(False)
            self._rebuild_state()
            self.status_cb("Drag cancelled")
            return
        # 2. Cancel a half-placed two-click bbox (the unmatched corner).
        if self.pending_corner is not None:
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i][0] == 'corner':
                    self.history.pop(i)
                    break
            self.redo_stack = []
            self.dirty = True
            self._rubber_rect.setVisible(False)
            self._rebuild_state()
            self.status_cb("Bounding-box creation cancelled")
            return
        # 3. Cancel a rubber-band drag that hasn't been released yet.
        if self._press_scene_pt is not None or self._rubber_rect.isVisible():
            self._press_vp = None
            self._press_scene_pt = None
            self._rubber_rect.setVisible(False)
            self.status_cb("Drawing cancelled")
            return
        # 4. Clear the current selection (e.g. a selected inner keypoint).
        if self._selected is not None:
            self._selected = None
            self._rebuild_scene()
            self.status_cb("Selection cleared")
            return
        self.status_cb("Nothing to cancel")

    # ── bat keypoints (derived on demand) ────────────────────────────────

    def bat_keypoints(self):
        if self.bat_top is None or self.bat_tip is None:
            return None
        self._sync_inner_pts()
        if self.bat_n >= 2:
            return ([list(self.bat_top)]
                    + [list(p) for p in self.bat_inner_pts]
                    + [list(self.bat_tip)])
        return [list(self.bat_top)]

    def _sync_inner_pts(self):
        """Keep bat_inner_pts length == n-2; initialize to evenly-spaced
        positions along Top→Tip on mismatch (e.g. after n changed)."""
        need = max(self.bat_n - 2, 0)
        if len(self.bat_inner_pts) != need:
            if self.bat_top is not None and self.bat_tip is not None:
                fracs = inner_fractions(self.bat_n)
                ax, ay = self.bat_top
                bx, by = self.bat_tip
                self.bat_inner_pts = [
                    [ax + f * (bx - ax), ay + f * (by - ay)] for f in fracs
                ]
            else:
                self.bat_inner_pts = [[0.0, 0.0]] * need

    def reset_inner_pts(self):
        """Force the intermediate keypoints back to even spacing along Top→Tip
        (called when n changes)."""
        if self.bat_top is not None and self.bat_tip is not None:
            fracs = inner_fractions(self.bat_n)
            ax, ay = self.bat_top
            bx, by = self.bat_tip
            self.bat_inner_pts = [
                [ax + f * (bx - ax), ay + f * (by - ay)] for f in fracs
            ]
        else:
            self.bat_inner_pts = []

    # ── per-keypoint visibility helpers ────────────────────────────────────

    def _sync_bat_vis(self):
        """Keep bat_vis length == bat_n, preserving existing per-keypoint
        values and seeding new slots with the session default."""
        if len(self.bat_vis) != self.bat_n:
            old = self.bat_vis
            self.bat_vis = [old[i] if i < len(old) else self.kp_visibility
                            for i in range(self.bat_n)]

    def reset_bat_vis(self):
        """Reset every Bat keypoint to the current session default."""
        self.bat_vis = [self.kp_visibility for _ in range(self.bat_n)]

    def kp_vis_for(self, ref):
        """Return the stored visibility flag for a keypoint reference."""
        if ref[0] == 'point':
            p = self.points[ref[1]]
            return p[2] if len(p) > 2 else self.kp_visibility
        self._sync_bat_vis()
        i = ref[1]
        return self.bat_vis[i] if 0 <= i < len(self.bat_vis) else 2

    def set_keypoint_visibility(self, ref, vis):
        """Set one keypoint's visibility (right-click menu / panel / 0-1-2)."""
        vis = int(vis)
        if ref[0] == 'point':
            i = ref[1]
            if not (0 <= i < len(self.points)):
                return
            p = self.points[i]
            self.points[i] = [p[0], p[1], vis]
            # Fold the change back into history so undo/save stay coherent.
            self.history    = self._canonical_history()
            self.redo_stack = []
            label = f"KP{i + 1}"
        else:
            self._sync_bat_vis()
            i = ref[1]
            if not (0 <= i < len(self.bat_vis)):
                return
            self.bat_vis[i] = vis
            label = f"KP{i + 1}"
        self.dirty = True
        self._rebuild_scene()
        self._notify_state()
        self.status_cb(f"{label} visibility → {vis}")

    def set_selected_visibility(self, vis):
        """Apply a visibility value to the currently selected keypoint."""
        if self._selected is None or self._selected[0] not in ('point', 'bat_kp'):
            self.status_cb("Select a keypoint first, then press 0 / 1 / 2")
            return
        self.set_keypoint_visibility(self._selected, vis)

    # ── selection / deletion ────────────────────────────────────────────────

    def _set_selection(self, ref):
        """Update the current selection and redraw the highlight."""
        if ref == self._selected:
            return
        self._selected = ref
        self._rebuild_scene()

    def _sel_from_edit_target(self, target):
        """Map an edit-mode grab descriptor to a stable selection reference."""
        if target is None:
            return None
        kind = target[0]
        if kind == 'point':
            return ('point', target[1])
        if kind == 'anchor':
            return (('bat_kp', 0) if target[1] == 'bat_top'
                    else ('bat_kp', self.bat_n - 1))
        if kind == 'bat_inner':
            return ('bat_kp', target[1] + 1)
        if kind == 'box':
            return ('box', target[1])
        return None

    def delete_selected(self):
        """Backspace: delete whatever annotation is currently selected.

        Bbox → whole box removed (a Bat box also resets its keypoints, forcing
        a fresh box).  General keypoint → that point removed.  Bat anchor →
        the anchor is cleared and the object reverts to placing it again.
        Interpolated (inner) Bat keypoints are not individually deletable.
        """
        sel = self._selected
        if sel is None:
            self.status_cb("Nothing selected — turn on Edit (Ctrl+E), click "
                           "an annotation, then Backspace")
            return
        kind = sel[0]
        if kind == 'point':
            i = sel[1]
            if 0 <= i < len(self.points):
                self.points.pop(i)
                self.status_cb("Keypoint deleted")
        elif kind == 'bat_kp':
            i = sel[1]
            n = self.bat_n
            last = n - 1
            if i in (0,) :
                self.bat_top = None
                self.status_cb("Top anchor removed — click to place it again")
            elif i in (last, -1):
                self.bat_tip = None
                self.status_cb("Tip anchor removed — click to place it again")
            else:
                self.status_cb("Interpolated keypoints can't be deleted — "
                               "change n or delete an anchor")
                return
        elif kind == 'box':
            ref = sel[1]
            if ref[0] == 'bat':
                # Spec 14: deleting the bbox resets the Bat object entirely so
                # the user must draw a new box before placing keypoints again.
                self.bat_box = None
                self.bat_top = None
                self.bat_tip = None
                self.status_cb("BBox deleted — draw a new bounding box to "
                               "restart keypoint annotation")
            else:
                i = ref[1]
                if 0 <= i < len(self.boxes):
                    self.boxes.pop(i)
                    self.status_cb("BBox deleted")
        self._selected  = None
        self.history    = self._canonical_history()
        self.redo_stack = []
        self.dirty      = True
        self._rebuild_state()

    def delete_all(self):
        """Delete key: wipe every annotation on this frame at once.

        Unlike Clear Annotations, this stays undoable: the prior state is
        snapshotted so a single Ctrl+Z restores all annotations.  The undo is
        per-frame — navigating to another image discards the snapshot."""
        if (not self.boxes and not self.points and self.bat_box is None
                and self.bat_top is None and self.bat_tip is None):
            self.status_cb("Nothing to delete")
            return
        self._cleared_snapshot = self._canonical_history()
        self.history    = []
        self.redo_stack = []
        self._selected  = None
        self.dirty      = True
        self._rebuild_state()
        self.status_cb("All annotations deleted — Ctrl+Z to undo")

    def _project_t(self, sp):
        """Vector-project a scene point onto the Top→Tip segment, returning
        the clamped parameter t∈[0,1].  Used to constrain inner-point drags."""
        if self.bat_top is None or self.bat_tip is None:
            return 0.0
        ax, ay = self.bat_top
        bx, by = self.bat_tip
        dx, dy = bx - ax, by - ay
        denom  = dx * dx + dy * dy
        if denom < 1e-9:
            return 0.0
        t = ((sp[0] - ax) * dx + (sp[1] - ay) * dy) / denom
        return min(max(t, 0.0), 1.0)

    # ── zoom / fit ───────────────────────────────────────────────────────

    def wheelEvent(self, event):
        """Ctrl+Scroll  → zoom, pivoting exactly under the mouse pointer.
        Shift+Scroll → horizontal pan.
        Plain scroll → vertical pan (built-in scrollbars)."""
        mods  = event.modifiers()
        delta = event.angleDelta().y() or event.angleDelta().x()
        if mods & Qt.KeyboardModifier.ControlModifier:
            factor = ZOOM_FACTOR if delta > 0 else 1.0 / ZOOM_FACTOR

            # Anchor the zoom on the scene point under the cursor, using
            # only viewport-local coordinates (no QCursor::pos()).
            anchor_vp = event.position().toPoint()
            anchor_scene = self.mapToScene(anchor_vp)

            self.scale(factor, factor)

            # Re-pan so that scene point lands back under the cursor.
            shift = self.mapFromScene(anchor_scene) - anchor_vp
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() + shift.x())
            vbar.setValue(vbar.value() + shift.y())

            event.accept()
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)

    def fit_to_screen(self):
        """Fit the whole image into the viewport (Ctrl+F)."""
        if self.img_w > 0 and self.img_h > 0:
            self.fitInView(QRectF(0, 0, self.img_w, self.img_h),
                           Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_to_bbox(self):
        """Zoom the view to the Bat bounding box (B key).

        A small margin is added so the box isn't flush against the viewport
        edges.  Does nothing (with a hint) when no box has been drawn yet."""
        if self.bat_box is None:
            self.status_cb("No bbox yet — draw the bounding box first")
            return
        x1, y1, x2, y2 = self.bat_box
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        pad_x, pad_y = w * 0.15, h * 0.15
        self.fitInView(QRectF(x1 - pad_x, y1 - pad_y,
                              w + 2 * pad_x, h + 2 * pad_y),
                       Qt.AspectRatioMode.KeepAspectRatio)
        self.status_cb("Zoomed to bbox (B)")

    # ── coordinate helpers ───────────────────────────────────────────────

    def _scene_pt(self, viewport_qpoint):
        """Viewport QPoint → image-pixel [x, y], clamped to image bounds."""
        sp = self.mapToScene(viewport_qpoint)
        return [min(max(sp.x(), 0.0), max(self.img_w - 1.0, 0.0)),
                min(max(sp.y(), 0.0), max(self.img_h - 1.0, 0.0))]

    def _hit_test_anchor(self, viewport_qpoint):
        """Return the draggable Bat target under the press, or None.

        'bat_top' / 'bat_tip'  → freely movable end anchors.
        ('inner', j)           → intermediate keypoint j (constrained to the
                                 line; handled via vector projection).
        End anchors take priority over inner points on a tie.
        """
        for name, pt in (('bat_top', self.bat_top), ('bat_tip', self.bat_tip)):
            if pt is None:
                continue
            # Map the scene anchor back to viewport screen pixels to compare.
            screen = self.mapFromScene(QPointF(pt[0], pt[1]))
            dist = ((QPointF(viewport_qpoint) - QPointF(screen))
                    .manhattanLength())
            if dist <= ANCHOR_HIT_RADIUS:
                return name
        kps = self.bat_keypoints()
        if kps is not None and len(kps) > 2:
            for k in range(1, len(kps) - 1):
                screen = self.mapFromScene(QPointF(kps[k][0], kps[k][1]))
                dist = ((QPointF(viewport_qpoint) - QPointF(screen))
                        .manhattanLength())
                if dist <= ANCHOR_HIT_RADIUS:
                    return ('inner', k - 1)
        return None

    def _hit_test_keypoint(self, viewport_qpoint):
        """Return the keypoint reference under the cursor, or None.

        ('point', i)  → general keypoint i
        ('bat_kp', i) → Bat keypoint i (i == 0 / n-1 are the Top / Tip anchors)
        Used for the right-click visibility menu and select-on-right-click.
        """
        best, best_d = None, KP_HIT_RADIUS
        for i, pt in enumerate(self.points):
            d = self._vp_dist(viewport_qpoint, pt)
            if d <= best_d:
                best, best_d = ('point', i), d
        kps = self.bat_keypoints()
        if kps is not None:
            for i, pt in enumerate(kps):
                d = self._vp_dist(viewport_qpoint, pt)
                if d <= best_d:
                    best, best_d = ('bat_kp', i), d
        elif self.bat_top is not None:
            if self._vp_dist(viewport_qpoint, self.bat_top) <= best_d:
                best = ('bat_kp', 0)
        return best

    # ── right-click visibility menu (Option A) ─────────────────────────────

    def contextMenuEvent(self, event):
        """Right-click a keypoint → set its visibility (0 / 1 / 2) or delete."""
        if self._pixmap_item is None:
            return
        ref = self._hit_test_keypoint(event.pos())
        if ref is None:
            return
        self._set_selection(ref)
        cur  = self.kp_vis_for(ref)
        kpno = ref[1] + 1
        menu = QMenu(self)
        header = menu.addAction(f"Keypoint {kpno} — visibility")
        header.setEnabled(False)
        menu.addSeparator()
        for val, txt in ((2, "2  ·  fully visible"),
                         (1, "1  ·  visible but occluded"),
                         (0, "0  ·  not visible")):
            act = menu.addAction(("✓  " if val == cur else "     ") + txt)
            act.triggered.connect(
                lambda _checked=False, v=val: self.set_keypoint_visibility(ref, v))
        menu.addSeparator()
        dele = menu.addAction("Delete keypoint    (Backspace)")
        dele.triggered.connect(self.delete_selected)
        menu.exec(event.globalPos())

    # ── edit-mode hit testing ─────────────────────────────────────────────

    def _vp_dist(self, viewport_qpoint, scene_xy):
        screen = self.mapFromScene(QPointF(scene_xy[0], scene_xy[1]))
        return (QPointF(viewport_qpoint) - QPointF(screen)).manhattanLength()

    def _editable_boxes(self):
        """Yield (ref, box) for every editable box.  Bat box first so it
        wins ties when overlapping a general box."""
        if self.bat_box is not None:
            yield ('bat',), self.bat_box
        for i, box in enumerate(self.boxes):
            yield ('gen', i), box

    def _hit_test_edit(self, viewport_qpoint):
        """Return the edit grab descriptor under the cursor, or None.

        Priority: keypoints/anchors → box corners → box edges → box interior.
        Distances are measured in viewport pixels so tolerances stay constant
        across zoom levels.
        """
        # Keypoints (general) — closest within radius.
        best = None
        best_d = KP_HIT_RADIUS
        for i, pt in enumerate(self.points):
            d = self._vp_dist(viewport_qpoint, pt)
            if d <= best_d:
                best, best_d = ('point', i), d
        # Bat anchors (top / tip).
        for name, pt in (('bat_top', self.bat_top), ('bat_tip', self.bat_tip)):
            if pt is None:
                continue
            d = self._vp_dist(viewport_qpoint, pt)
            if d <= best_d:
                best, best_d = ('anchor', name), d
        # Bat intermediate keypoints (constrained to the line on drag).
        kps = self.bat_keypoints()
        if kps is not None and len(kps) > 2:
            for k in range(1, len(kps) - 1):
                d = self._vp_dist(viewport_qpoint, kps[k])
                if d <= best_d:
                    best, best_d = ('bat_inner', k - 1), d
        if best is not None:
            return best

        # Box corners, then edges, then interior.
        vx, vy = viewport_qpoint.x(), viewport_qpoint.y()
        for ref, box in self._editable_boxes():
            tl = self.mapFromScene(QPointF(box[0], box[1]))
            br = self.mapFromScene(QPointF(box[2], box[3]))
            left, right = tl.x(), br.x()
            top, bottom = tl.y(), br.y()

            corners = (('nw', left, top), ('ne', right, top),
                       ('sw', left, bottom), ('se', right, bottom))
            for handle, cx, cy in corners:
                if abs(vx - cx) <= CORNER_HIT_RADIUS and \
                        abs(vy - cy) <= CORNER_HIT_RADIUS:
                    return ('box', ref, handle)

            in_x = left - EDGE_HIT_RADIUS <= vx <= right + EDGE_HIT_RADIUS
            in_y = top - EDGE_HIT_RADIUS <= vy <= bottom + EDGE_HIT_RADIUS
            if in_y and abs(vx - left) <= EDGE_HIT_RADIUS:
                return ('box', ref, 'w')
            if in_y and abs(vx - right) <= EDGE_HIT_RADIUS:
                return ('box', ref, 'e')
            if in_x and abs(vy - top) <= EDGE_HIT_RADIUS:
                return ('box', ref, 'n')
            if in_x and abs(vy - bottom) <= EDGE_HIT_RADIUS:
                return ('box', ref, 's')
            if left < vx < right and top < vy < bottom:
                return ('box', ref, 'move')
        return None

    _CURSORS = {
        'nw': Qt.CursorShape.SizeFDiagCursor, 'se': Qt.CursorShape.SizeFDiagCursor,
        'ne': Qt.CursorShape.SizeBDiagCursor, 'sw': Qt.CursorShape.SizeBDiagCursor,
        'n':  Qt.CursorShape.SizeVerCursor,   's':  Qt.CursorShape.SizeVerCursor,
        'e':  Qt.CursorShape.SizeHorCursor,   'w':  Qt.CursorShape.SizeHorCursor,
        'move': Qt.CursorShape.SizeAllCursor,
    }

    def _cursor_for_target(self, target):
        if target is None:
            return Qt.CursorShape.ArrowCursor
        if target[0] in ('point', 'anchor', 'bat_inner'):
            return Qt.CursorShape.OpenHandCursor
        return self._CURSORS.get(target[2], Qt.CursorShape.ArrowCursor)

    def _update_idle_cursor(self):
        if self._pixmap_item is None:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        elif self.edit_mode:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    # ── edit-mode mutation ─────────────────────────────────────────────────

    def _box_for_ref(self, ref):
        if ref[0] == 'bat':
            return self.bat_box
        return self.boxes[ref[1]]

    def _apply_edit(self, scene_pt):
        """Mutate the derived data field referenced by the current grab."""
        kind = self._edit_target[0]
        if kind == 'point':
            old = self.points[self._edit_target[1]]
            vis = old[2] if len(old) > 2 else self.kp_visibility
            self.points[self._edit_target[1]] = [scene_pt[0], scene_pt[1], vis]
        elif kind == 'anchor':
            if self._edit_target[1] == 'bat_top':
                self.bat_top = list(scene_pt)
            else:
                self.bat_tip = list(scene_pt)
        elif kind == 'bat_inner':
            # Free placement: store absolute position (no line constraint).
            j = self._edit_target[1]
            self._sync_inner_pts()
            if 0 <= j < len(self.bat_inner_pts):
                self.bat_inner_pts[j] = list(scene_pt)
        elif kind == 'box':
            box    = self._box_for_ref(self._edit_target[1])
            handle = self._edit_target[2]
            if box is not None:
                self._modify_box(box, handle, scene_pt)
        self._edit_last_scene = list(scene_pt)

    def _modify_box(self, box, handle, sp):
        """Resize (edge/corner) or translate (interior) a box in place,
        clamped to the image and to a 1-px minimum size."""
        x, y = sp[0], sp[1]
        if handle == 'move':
            dx = x - self._edit_last_scene[0]
            dy = y - self._edit_last_scene[1]
            w, h = box[2] - box[0], box[3] - box[1]
            nx1 = min(max(box[0] + dx, 0.0), self.img_w - w)
            ny1 = min(max(box[1] + dy, 0.0), self.img_h - h)
            box[0], box[1] = nx1, ny1
            box[2], box[3] = nx1 + w, ny1 + h
            return
        # Edge/corner: 'w'/'e' move x, 'n'/'s' move y (corners combine both).
        if 'w' in handle:
            box[0] = min(x, box[2] - 1.0)
        if 'e' in handle:
            box[2] = max(x, box[0] + 1.0)
        if 'n' in handle:
            box[1] = min(y, box[3] - 1.0)
        if 's' in handle:
            box[3] = max(y, box[1] + 1.0)

    def _canonical_history(self):
        """Rebuild a minimal history equivalent to the current derived state.

        Boxes/points become single 'box'/'point'/'bat_*' actions so that a
        replay through _rebuild_state() reproduces the edited geometry
        exactly.  Called after an edit so undo/redo/save stay coherent.
        """
        h = []
        for box in self.boxes:
            h.append(('box', list(box)))
        for p in self.points:
            h.append(('point', list(p)))
        if self.bat_box is not None:
            h.append(('box', list(self.bat_box)))   # → bat_box in Bat mode
        if self.bat_top is not None:
            h.append(('bat_top', list(self.bat_top)))
        if self.bat_tip is not None:
            h.append(('bat_tip', list(self.bat_tip)))
        return h

    # ── mouse events ────────────────────────────────────────────────────

    def leaveEvent(self, event):
        self._crosshair_h.setVisible(False)
        self._crosshair_v.setVisible(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if self._pixmap_item is None:
            super().mousePressEvent(event)
            return

        # ── Shift + left-drag = pan the view (hand/grab), in every mode ───
        # Plain left-click stays free for drawing/editing; Shift is what
        # turns the same button into a grab-and-drag of the image.
        if (event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._panning     = True
            self._pan_last_vp = event.pos()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        # ── Edit mode: grab an existing keypoint / box, never draw ────────
        if self.edit_mode:
            self._edit_target = self._hit_test_edit(event.pos())
            self._edit_moved  = False
            # Clicking an item selects it (for Backspace / 0-1-2); clicking
            # empty space clears the selection.
            self._set_selection(self._sel_from_edit_target(self._edit_target))
            if self._edit_target is not None:
                self._edit_last_scene = self._scene_pt(event.pos())
                self._drag_snapshot   = list(self.history)
                self._dragging        = True
                self.viewport().setCursor(
                    self._cursor_for_target(self._edit_target))
            return

        self._press_vp       = event.pos()
        self._press_scene_pt = self._scene_pt(event.pos())
        # Anchor drag takes priority over a fresh click.
        self._drag_anchor = self._hit_test_anchor(event.pos())
        if self._drag_anchor:
            self._drag_snapshot = list(self.history)
            self._dragging      = True

    def mouseMoveEvent(self, event):
        if self._pixmap_item is None:
            super().mouseMoveEvent(event)
            return

        # ── pan in progress (Shift+left-drag): scroll the view, nothing else
        if self._panning and (event.buttons() & Qt.MouseButton.LeftButton):
            delta = event.pos() - self._pan_last_vp
            self._pan_last_vp = event.pos()
            hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            event.accept()
            return

        sp = self.mapToScene(event.pos())
        self._cursor_scene = [sp.x(), sp.y()]
        lmb = bool(event.buttons() & Qt.MouseButton.LeftButton)

        # ── Edit mode ─────────────────────────────────────────────────────
        if self.edit_mode:
            if self._edit_target is not None and lmb:
                self._apply_edit(self._scene_pt(event.pos()))
                self._edit_moved = True
                self.dirty = True
                self._rebuild_scene()    # light: redraw from mutated state
            elif not lmb:
                # Hover feedback only.
                self.viewport().setCursor(
                    self._cursor_for_target(self._hit_test_edit(event.pos())))
            return

        # ── crosshairs: stretch to scene (image) edges ───────────────────
        in_img = (0 <= sp.x() <= self.img_w and 0 <= sp.y() <= self.img_h)
        if in_img:
            self._crosshair_h.setLine(0.0, sp.y(), float(self.img_w), sp.y())
            self._crosshair_v.setLine(sp.x(), 0.0, sp.x(), float(self.img_h))
            self._crosshair_h.setVisible(True)
            self._crosshair_v.setVisible(True)
        else:
            self._crosshair_h.setVisible(False)
            self._crosshair_v.setVisible(False)

        # ── anchor / inner-point drag ────────────────────────────────────
        if self._drag_anchor and lmb:
            new_pt = self._scene_pt(event.pos())
            if (isinstance(self._drag_anchor, tuple)
                    and self._drag_anchor[0] == 'inner'):
                # Free drag: store absolute position for the inner keypoint.
                j = self._drag_anchor[1]
                self._sync_inner_pts()
                if 0 <= j < len(self.bat_inner_pts):
                    self.bat_inner_pts[j] = list(new_pt)
                self.dirty = True
                self._rebuild_scene()
            else:
                # Update the matching history entry in-place so that undo/redo
                # replay stays consistent with the dragged position.
                for i in range(len(self.history) - 1, -1, -1):
                    if self.history[i][0] == self._drag_anchor:
                        self.history[i] = (self._drag_anchor, new_pt)
                        break
                self.dirty = True
                self._rebuild_state()

        # ── drag-draw rubber band ────────────────────────────────────────
        elif (self._press_scene_pt is not None and lmb
              and self._box_drawing_allowed()):
            p1, p2 = self._press_scene_pt, self._scene_pt(event.pos())
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            self._rubber_rect.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            self._rubber_rect.setVisible(True)

        # ── two-click corner preview (first corner placed, mouse free) ───
        elif self.pending_corner is not None and not lmb:
            c  = self.pending_corner
            cx = min(max(sp.x(), 0.0), float(self.img_w))
            cy = min(max(sp.y(), 0.0), float(self.img_h))
            x1, y1 = min(c[0], cx), min(c[1], cy)
            x2, y2 = max(c[0], cx), max(c[1], cy)
            self._rubber_rect.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            self._rubber_rect.setVisible(True)

        else:
            self._rubber_rect.setVisible(False)

    def mouseReleaseEvent(self, event):
        if (self._pixmap_item is None
                or event.button() != Qt.MouseButton.LeftButton):
            super().mouseReleaseEvent(event)
            return

        # ── end a pan (Shift+left-drag) ──────────────────────────────────
        if self._panning:
            self._panning     = False
            self._pan_last_vp = None
            self.viewport().unsetCursor()
            event.accept()
            return

        # ── Edit mode: commit the grab by canonicalising history ─────────
        if self.edit_mode:
            self._dragging      = False
            self._drag_snapshot = None
            if self._edit_target is not None and self._edit_moved:
                # Normalise any inverted box, then snapshot to history.
                for box in list(self.boxes) + \
                        ([self.bat_box] if self.bat_box is not None else []):
                    x1, y1, x2, y2 = box
                    box[0], box[2] = min(x1, x2), max(x1, x2)
                    box[1], box[3] = min(y1, y2), max(y1, y2)
                self.history    = self._canonical_history()
                self.redo_stack = []
                self.dirty      = True
                self._rebuild_state()
                self.status_cb("Annotation edited")
            self._edit_target = None
            self._edit_moved  = False
            self.viewport().setCursor(
                self._cursor_for_target(self._hit_test_edit(event.pos())))
            return

        moved = 0.0
        if self._press_vp is not None:
            moved = ((QPointF(event.pos()) - QPointF(self._press_vp))
                     .manhattanLength())

        if self._drag_anchor:
            if moved > CLICK_TOLERANCE:
                self.status_cb("Anchor moved — curve updated")
            else:
                # Tap (no drag) on an inner keypoint → select it for
                # click-to-reposition.
                if (isinstance(self._drag_anchor, tuple)
                        and self._drag_anchor[0] == 'inner'):
                    j = self._drag_anchor[1]
                    self._selected = ('bat_kp', j + 1)
                    self.status_cb(
                        f"KP{j + 2} selected — click anywhere to reposition it")
            self._drag_anchor   = None
            self._dragging      = False
            self._drag_snapshot = None
            self._rebuild_state()   # refresh sidebar now the drag has settled

        elif moved <= CLICK_TOLERANCE:
            self._handle_click(self._scene_pt(event.pos()))

        elif self._box_drawing_allowed():
            p1, p2 = self._press_scene_pt, self._scene_pt(event.pos())
            box = [min(p1[0], p2[0]), min(p1[1], p2[1]),
                   max(p1[0], p2[0]), max(p1[1], p2[1])]
            if box[2] > box[0] and box[3] > box[1]:   # reject zero-area
                self._record('box', box)
                self.status_cb(self._next_step_hint())

        self._press_vp       = None
        self._press_scene_pt = None
        self._rubber_rect.setVisible(False)

    # ── keyboard (canvas-focused single-key shortcuts) ──────────────────────

    def keyPressEvent(self, event):
        """Single-key workflow shortcuts handled while the canvas has focus.

        Routing them here (rather than as global QShortcuts) keeps A/D/Space/
        Backspace usable for annotation without hijacking text-entry fields.
        """
        key  = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl and key == Qt.Key.Key_C and self.request_copy:
            self.request_copy()
        elif ctrl and key == Qt.Key.Key_V and self.request_paste:
            self.request_paste()
        elif key == Qt.Key.Key_Escape:
            self.cancel_operation()
        elif key == Qt.Key.Key_A and self.request_prev:
            self.request_prev()
        elif key == Qt.Key.Key_D and self.request_next:
            self.request_next()
        elif key == Qt.Key.Key_Space and self.request_skip:
            self.request_skip()
        elif key == Qt.Key.Key_B:
            self.zoom_to_bbox()
        elif key == Qt.Key.Key_Backspace:
            self.delete_selected()
        elif key == Qt.Key.Key_Delete:
            self.delete_all()
        elif key in (Qt.Key.Key_0, Qt.Key.Key_1, Qt.Key.Key_2):
            self.set_selected_visibility(key - Qt.Key.Key_0)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    # ── annotation click routing ─────────────────────────────────────────

    def _box_drawing_allowed(self):
        # Bat is the only mode and its box is mandatory: drawing is allowed
        # only while no box exists yet.
        return (not self.edit_mode) and self.bat_box is None

    def _handle_click(self, pt):
        # A bounding box is mandatory: keypoints cannot be placed until one
        # exists (spec 13/14).
        if self.bat_box is None:
            self._record('corner', pt)
        elif self.bat_top is None:
            self._record('bat_top', pt)
        elif self.bat_tip is None:
            self._record('bat_tip', pt)
        elif (isinstance(self._selected, tuple)
              and self._selected[0] == 'bat_kp'
              and 0 < self._selected[1] < self.bat_n - 1):
            # Click-to-reposition: move the selected inner keypoint freely.
            j = self._selected[1] - 1
            self._sync_inner_pts()
            if 0 <= j < len(self.bat_inner_pts):
                self.bat_inner_pts[j] = list(pt)
                self.dirty = True
                self._rebuild_state()
                self.status_cb(
                    f"KP{j + 2} moved — click again to reposition, "
                    "or ESC to deselect")
            return
        else:
            self.status_cb("Bat object complete — click KP2 to select it, "
                           "then click anywhere to reposition; "
                           "Ctrl+S to save")
            return
        self.status_cb(self._next_step_hint())

    def _next_step_hint(self):
        if self.bat_box is None:
            if self.pending_corner is not None:
                return "BBox: click the bottom-right corner"
            return "BBox required: click top-left corner (or click-drag the box)"
        if self.bat_top is None:
            return "BBox set ✓ — keypoint click 1 of 2: click the TOP"
        if self.bat_tip is None:
            return "Keypoint click 2 of 2: click the TIP (bottom)"
        return (f"{self.bat_n} keypoints interpolated — "
                f"drag the Top/Tip anchors to fine-tune")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Annotation Tool — Bat Keypoint Interpolator")
        self.resize(1400, 880)

        self.images_dir      = None
        self.labels_dir      = None   # default .txt output dir
        self.json_dir        = None   # default .json output dir
        self.user_labels_dir = None   # optional dir chosen via Open Labels Dir
        self.image_files     = []
        self.current_index   = -1
        # Copy/paste buffer for one annotation (Ctrl+C → Ctrl+V); persists
        # across image navigation so it can be pasted onto another frame.
        self._clipboard      = None

        self.theme = 'dark'
        self._build_menu()
        self._build_ui()
        self._build_shortcuts()
        self._sync_edit_controls()
        self._refresh_kp_panel()
        self.apply_theme('dark')   # sleek dark theme by default


    @staticmethod
    def _clean_json(obj):
        """Recursively strip whitespace from all JSON keys and string values.
        Fixes compatibility with malformed LabelMe exports that add trailing spaces."""
        if isinstance(obj, dict):
            return {k.strip(): (v.strip() if isinstance(v, str) else MainWindow._clean_json(v)) 
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MainWindow._clean_json(elem) for elem in obj]
        return obj
    # ── menu bar ───────────────────────────────────────────────────────────

    def _build_menu(self):
        bar = self.menuBar()

        # File ────────────────────────────────────────────────────────────
        file_menu = bar.addMenu("&File")
        act_open_img = QAction("Open Image Dir…", self)
        act_open_img.setShortcut("Ctrl+O")
        act_open_img.triggered.connect(self.open_image_dir)
        file_menu.addAction(act_open_img)

        act_open_lbl = QAction("Open Labels Dir…", self)
        act_open_lbl.triggered.connect(self.open_labels_dir)
        file_menu.addAction(act_open_lbl)

        file_menu.addSeparator()
        act_save = QAction("Save", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(lambda: self.save_current())
        file_menu.addAction(act_save)

        # Auto-save is ON by default (spec 1).  Kept as a checkable QAction
        # named autosave_check (a QAction exposes isChecked()/setChecked()).
        self.autosave_check = QAction("Auto-save on navigate", self)
        self.autosave_check.setCheckable(True)
        self.autosave_check.setChecked(True)
        file_menu.addAction(self.autosave_check)

        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Edit ──────────────────────────────────────────────────────────────
        edit_menu = bar.addMenu("&Edit")
        self.act_edit_mode = QAction("Edit / Resize Mode", self)
        self.act_edit_mode.setCheckable(True)
        self.act_edit_mode.setShortcut("Ctrl+E")
        self.act_edit_mode.toggled.connect(self._on_edit_toggled)
        edit_menu.addAction(self.act_edit_mode)

        edit_menu.addSeparator()
        act_undo = QAction("Undo", self)
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(lambda: self.canvas.undo())
        edit_menu.addAction(act_undo)

        act_redo = QAction("Redo", self)
        act_redo.setShortcut("Ctrl+Y")
        act_redo.triggered.connect(lambda: self.canvas.redo())
        edit_menu.addAction(act_redo)

        act_cancel = QAction("Cancel Current Operation", self)
        act_cancel.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        act_cancel.triggered.connect(lambda: self.canvas.cancel_operation())
        edit_menu.addAction(act_cancel)

        edit_menu.addSeparator()
        # Backspace is handled by the canvas key handler (not a global
        # shortcut) so it never hijacks the class-label text field.
        act_del_sel = QAction("Delete Selected Annotation\tBackspace", self)
        act_del_sel.triggered.connect(lambda: self.canvas.delete_selected())
        edit_menu.addAction(act_del_sel)

        # Delete is handled by the canvas key handler; deletes everything on the
        # frame but stays undoable (Ctrl+Z) until you leave the image.
        act_del_all = QAction("Delete All Annotations\tDelete", self)
        act_del_all.triggered.connect(lambda: self.canvas.delete_all())
        edit_menu.addAction(act_del_all)

        act_clear = QAction("Clear Annotations", self)
        act_clear.triggered.connect(lambda: self.canvas.clear_annotations())
        edit_menu.addAction(act_clear)

        edit_menu.addSeparator()
        # Ctrl+C / Ctrl+V are handled by the canvas key handler (not global
        # shortcuts) so they never hijack copy/paste in the class-label field.
        act_copy = QAction("Copy Annotation\tCtrl+C", self)
        act_copy.triggered.connect(self.copy_annotation)
        edit_menu.addAction(act_copy)

        act_paste = QAction("Paste Annotation\tCtrl+V", self)
        act_paste.triggered.connect(self.paste_annotation)
        edit_menu.addAction(act_paste)

        # View ────────────────────────────────────────────────────────────
        view_menu = bar.addMenu("&View")
        act_fit = QAction("Fit to Screen", self)
        act_fit.setShortcut("Ctrl+F")
        act_fit.triggered.connect(lambda: self.canvas.fit_to_screen())
        view_menu.addAction(act_fit)

        # "B" itself is handled by the canvas key handler (not a global
        # shortcut, so it never hijacks typing in side-panel fields); this
        # menu entry is just for discoverability.
        act_zoom_box = QAction("Zoom to BBox\tB", self)
        act_zoom_box.triggered.connect(lambda: self.canvas.zoom_to_bbox())
        view_menu.addAction(act_zoom_box)

        view_menu.addSeparator()
        # A / D / Space are handled by the canvas key handler (canvas-focused)
        # so they don't interfere with typing in the side-panel fields; the
        # "\t<key>" suffix shows the hint in the menu without binding globally.
        act_prev = QAction("Previous Image\tA", self)
        act_prev.triggered.connect(lambda: self.step(-1))
        view_menu.addAction(act_prev)
        act_next = QAction("Next Image\tD", self)
        act_next.triggered.connect(lambda: self.step(1))
        view_menu.addAction(act_next)
        act_skip = QAction("Skip Image (no prompt)\tSpace", self)
        act_skip.triggered.connect(self.skip_image)
        view_menu.addAction(act_skip)

        # Theme submenu (Dark default · spec: light + dark) ─────────────────
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme")
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        self.act_theme_dark = QAction("Dark", self, checkable=True)
        self.act_theme_light = QAction("Light", self, checkable=True)
        self.act_theme_dark.setShortcut("Ctrl+T")
        self.act_theme_dark.triggered.connect(lambda: self.apply_theme('dark'))
        self.act_theme_light.triggered.connect(lambda: self.apply_theme('light'))
        for a in (self.act_theme_dark, self.act_theme_light):
            self._theme_group.addAction(a)
            theme_menu.addAction(a)
        self.act_theme_dark.setChecked(True)

        # Help ──────────────────────────────────────────────────────────────
        help_menu = bar.addMenu("&Help")
        act_keys = QAction("Keyboard Shortcuts…", self)
        act_keys.setShortcut(QKeySequence(Qt.Key.Key_F1))
        act_keys.triggered.connect(self.show_shortcuts)
        help_menu.addAction(act_keys)

    # ── side dock ──────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        layout  = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = AnnotationView(self.statusBar().showMessage)
        # Wire the canvas back to the window for sidebar refresh + navigation.
        self.canvas.state_cb     = self._refresh_kp_panel
        self.canvas.request_prev = lambda: self.step(-1)
        self.canvas.request_next = lambda: self.step(1)
        self.canvas.request_skip = self.skip_image
        self.canvas.request_copy  = self.copy_annotation
        self.canvas.request_paste = self.paste_annotation
        layout.addWidget(self.canvas, stretch=1)

        # ── side panel (scrollable so it never crowds the canvas) ─────────
        side = QVBoxLayout()
        side.setContentsMargins(14, 14, 14, 14)
        side.setSpacing(14)

        # Visibility legend — always at the very top (spec 10).  Built from
        # uniformly-sized colour dots so every row lines up perfectly.
        self.legend = QFrame()
        self.legend.setObjectName("legend")
        leg = QVBoxLayout(self.legend)
        leg.setContentsMargins(14, 12, 14, 12)
        leg.setSpacing(7)
        title = QLabel("Keypoint visibility")
        title.setObjectName("legendTitle")
        leg.addWidget(title)
        for hexcol, txt in (("#34c759", "2 — fully visible"),
                            ("#ff9f0a", "1 — visible but occluded"),
                            ("#8e8e93", "0 — not visible")):
            r = QHBoxLayout()
            r.setContentsMargins(0, 0, 0, 0)
            r.setSpacing(10)
            dot = QLabel("●")
            dot.setFixedSize(16, 16)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setStyleSheet(f"color: {hexcol}; font-size: 13px;")
            lab = QLabel(txt)
            r.addWidget(dot)
            r.addWidget(lab, 1)
            leg.addLayout(r)
        side.addWidget(self.legend)

        # Tool settings.  Bat is the only annotation mode, so there is no mode
        # selector — just the class identity for export.
        tool_box  = QGroupBox("Tool")
        tool_form = QFormLayout(tool_box)
        tool_form.setVerticalSpacing(10)
        self.class_id_spin = QSpinBox()
        self.class_id_spin.setRange(0, 999)
        tool_form.addRow("Class ID:", self.class_id_spin)
        self.class_name_edit = QLineEdit("bat")
        tool_form.addRow("Class label:", self.class_name_edit)
        side.addWidget(tool_box)

        # Bat mode settings (spacing + draw-bbox controls removed; the box is
        # mandatory and interpolation is always equal).
        self.bat_box_group = QGroupBox("Bat Mode Settings")
        bat_form = QFormLayout(self.bat_box_group)
        bat_form.setVerticalSpacing(10)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(2, 200)
        self.n_spin.setValue(5)
        self.n_spin.valueChanged.connect(self._on_n_changed)
        bat_form.addRow("n (keypoints):", self.n_spin)
        hint = QLabel("Box is required · keypoints spaced equally")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        bat_form.addRow(hint)
        side.addWidget(self.bat_box_group)

        # Edit / Resize toggle.
        self.edit_btn = QPushButton("Edit / Resize Annotations")
        self.edit_btn.setObjectName("editBtn")
        self.edit_btn.setCheckable(True)
        self.edit_btn.setMinimumHeight(38)
        self.edit_btn.setToolTip(
            "When ON: click an annotation to select it, drag keypoints to move "
            "them, drag a box edge/corner to resize, or drag inside a box to "
            "move it.\nWhen OFF: click-drag draws new annotations.  (Ctrl+E)")
        self.edit_btn.toggled.connect(self._on_edit_toggled)
        side.addWidget(self.edit_btn)

        # Keypoint visibility panel (Option B) — one row per keypoint.  The
        # scroll area is sized to its content so up to KP_VISIBLE_ROWS rows
        # never show a scrollbar; beyond that it scrolls.
        self.kp_group = QGroupBox("Keypoint Visibility")
        kp_outer = QVBoxLayout(self.kp_group)
        kp_outer.setContentsMargins(8, 6, 8, 6)
        self.kp_scroll = QScrollArea()
        self.kp_scroll.setObjectName("kpScroll")
        self.kp_scroll.setWidgetResizable(True)
        self.kp_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.kp_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Keep the scroll contents transparent so the group-box card shows
        # through (no grey rectangle behind the rows).
        self.kp_scroll.viewport().setAutoFillBackground(False)
        self.kp_holder = QWidget()
        self.kp_holder.setObjectName("kpHolder")
        self.kp_holder.setAutoFillBackground(False)
        self.kp_layout = QVBoxLayout(self.kp_holder)
        self.kp_layout.setContentsMargins(0, 0, 0, 0)
        self.kp_layout.setSpacing(3)
        self.kp_scroll.setWidget(self.kp_holder)
        kp_outer.addWidget(self.kp_scroll)
        self._kp_empty = QLabel(
            "No keypoints yet — place some to set per-point visibility.")
        self._kp_empty.setObjectName("kpEmpty")
        self._kp_empty.setWordWrap(True)
        self._kp_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.kp_layout.addWidget(self._kp_empty)
        self.kp_layout.addStretch(1)
        self._kp_rows = []
        side.addWidget(self.kp_group)

        # Image file list: each row is a custom widget with a status tick
        # (left), the file name, and a Delete button (right).
        side.addWidget(QLabel("Images"))
        self.file_list = QListWidget()
        self._status_boxes = []        # per-row status ✔ QLabel, aligned to rows
        self.file_list.currentRowChanged.connect(self._on_row_changed)
        side.addWidget(self.file_list, stretch=1)

        panel = QWidget()
        panel.setObjectName("sidePanel")
        panel.setLayout(side)
        panel.setFixedWidth(320)
        layout.addWidget(panel)
        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "File ▸ Open Image Dir to begin  ·  Bat mode, auto-save ON")

    def _build_shortcuts(self):
        # Ctrl+D deletes the currently selected image (and its label files).
        # A/D/Space/Backspace/0-1-2 are handled by the canvas key handler.
        QShortcut(QKeySequence("Ctrl+D"), self,
                  activated=self.delete_current_image)

    # ── edit-mode plumbing ───────────────────────────────────────────────

    def _on_edit_toggled(self, checked):
        """Single handler for both the side button and the Edit menu action;
        keeps the two controls and the canvas in sync without recursion."""
        if self.canvas.edit_mode == checked:
            return
        self.canvas.set_edit_mode(checked)
        self._sync_edit_controls()
        # Bat settings only make sense when drawing, not editing.
        self.bat_box_group.setEnabled(not checked)
        self.statusBar().showMessage(
            "Edit mode ON — drag keypoints / box handles to adjust"
            if checked else "Edit mode OFF — draw new annotations")

    def _sync_edit_controls(self):
        on = self.canvas.edit_mode
        for w in (self.edit_btn, self.act_edit_mode):
            w.blockSignals(True)
            w.setChecked(on)
            w.blockSignals(False)

    # ── settings handlers ────────────────────────────────────────────────

    def _on_n_changed(self, n):
        # Keypoints are derived from the anchors on every _rebuild_scene,
        # so changing n re-spaces them instantly (always equal spacing now).
        self.canvas.bat_n = n
        self.canvas.reset_inner_pts()        # re-seed positions for new n
        self.canvas._sync_bat_vis()          # grow/shrink per-kp visibility list
        self.canvas._rebuild_scene()
        self._refresh_kp_panel()

    # ── keypoint visibility panel (Option B) ───────────────────────────────

    def _current_keypoint_refs(self):
        """List (ref, label) for every visibility-editable keypoint on canvas,
        in display order, where ref matches AnnotationView's selection model."""
        c = self.canvas
        refs = []
        kps = c.bat_keypoints()
        if kps is not None:
            c._sync_bat_vis()
            for i in range(len(kps)):
                tag = (" (Top)" if i == 0
                       else " (Tip)" if i == len(kps) - 1 else "")
                refs.append((('bat_kp', i), f"KP{i + 1}{tag}"))
        return refs

    def _refresh_kp_panel(self):
        """Rebuild the per-keypoint visibility rows from the canvas state.

        Each row is:  [● identity colour] [KPn label] [ 0 | 1 | 2 ] segmented
        toggle, where the current visibility is highlighted.  The colour dot
        matches that keypoint's marker on the canvas."""
        # Guard against any state callback that fires before the panel exists.
        if not hasattr(self, 'kp_layout'):
            return
        # Clear existing rows.
        for row in self._kp_rows:
            row.setParent(None)
            row.deleteLater()
        self._kp_rows = []

        refs = self._current_keypoint_refs()
        self._kp_empty.setVisible(not refs)
        for ref, label in refs:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(2, 0, 2, 0)
            h.setSpacing(7)

            dot = QLabel("●")
            dot.setStyleSheet(
                f"color: {color_for_index(ref[1]).name()}; font-size: 12px;")
            dot.setFixedWidth(13)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)

            name = QLabel(label)
            name.setObjectName("kpName")
            name.setMinimumWidth(58)

            h.addWidget(dot)
            h.addWidget(name, 1)

            # Segmented 0 / 1 / 2 toggle (exclusive button group per row).
            cur = self.canvas.kp_vis_for(ref)
            grp = QButtonGroup(row)
            grp.setExclusive(True)
            for val in (0, 1, 2):
                b = QPushButton(str(val))
                b.setObjectName("segBtn")
                b.setCheckable(True)
                b.setFixedSize(26, 22)
                b.setChecked(val == cur)
                grp.addButton(b)
                b.clicked.connect(
                    lambda _checked=False, r=ref, v=val:
                        self.canvas.set_keypoint_visibility(r, v))
                h.addWidget(b)

            # Insert before the trailing stretch (last layout item).
            self.kp_layout.insertWidget(self.kp_layout.count() - 1, row)
            self._kp_rows.append(row)

        # Size the scroll area to fit its content, capped so that up to
        # KP_VISIBLE_ROWS keypoints never need a scrollbar.
        KP_VISIBLE_ROWS = 6
        ROW_H = 25                       # per-row height incl. spacing (upper bound)
        if refs:
            rows_shown = min(len(refs), KP_VISIBLE_ROWS)
            self.kp_scroll.setFixedHeight(rows_shown * ROW_H + 8)
        else:
            self.kp_scroll.setFixedHeight(46)

    # ── folder management ────────────────────────────────────────────────

    def open_image_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        self.images_dir = folder
        parent          = os.path.dirname(os.path.abspath(folder))
        self.labels_dir = os.path.join(parent, "labels")
        self.json_dir   = os.path.join(parent, "json_labels")
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.json_dir,   exist_ok=True)

        self.image_files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS)

        if not self.image_files:
            self._populate_list()
            QMessageBox.warning(self, "Empty folder",
                                "No images found in the selected folder.")
            return

        self._populate_list()
        self.current_index = -1
        self.file_list.setCurrentRow(0)
        self.statusBar().showMessage(
            f"{len(self.image_files)} images loaded")

    def open_labels_dir(self):
        """Optional: point the loader/scanner at a folder of .txt or .json."""
        folder = QFileDialog.getExistingDirectory(self, "Select labels folder")
        if not folder:
            return
        self.user_labels_dir = folder
        self._populate_list()          # re-tick status checkboxes
        # Reload the current image's annotation from the new directory.
        if self.current_index >= 0:
            self.load_current()
        n = sum(1 for f in self.image_files
                if self._find_label_for(os.path.splitext(f)[0]))
        self.statusBar().showMessage(
            f"Labels dir set — {n}/{len(self.image_files)} images annotated")

    # ── smart image list ───────────────────────────────────────────────────

    def _label_dirs(self):
        """Directories searched for an existing label, in priority order."""
        dirs = []
        if self.user_labels_dir:
            dirs.append(self.user_labels_dir)
        if self.json_dir:
            dirs.append(self.json_dir)
        if self.labels_dir:
            dirs.append(self.labels_dir)
        return dirs

    def _find_label_for(self, base):
        """Return (kind, path) for a base name, or None.  JSON wins over TXT."""
        dirs = self._label_dirs()
        for d in dirs:
            p = os.path.join(d, base + ".json")
            if os.path.isfile(p):
                return ('json', p)
        for d in dirs:
            p = os.path.join(d, base + ".txt")
            if os.path.isfile(p):
                return ('txt', p)
        return None

    def _class_id_from_txt(self, base):
        """First class id from a sibling YOLO .txt, or None.

        LabelMe JSON has no class-id field, so when auto-configuring the
        Tool panel from a JSON we look up the matching .txt (same base name,
        searched in the same priority order) to recover the numeric class id.
        """
        for d in self._label_dirs():
            p = os.path.join(d, base + ".txt")
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            return int(float(parts[0]))
            except (OSError, ValueError):
                return None
        return None

    def _auto_configure_tool(self, rects, pts, rect_label, base):
        """Set Class label / Class ID to match a just-loaded label file.

        Bat is the only mode, so there is no mode inference — this just keeps
        the export identity (label + numeric id) in sync with the loaded file.
        """
        # Class label — taken from the rectangle shape (points are 'point_N').
        if rect_label:
            self.class_name_edit.setText(rect_label)

        # Class ID — recovered from the sibling YOLO .txt if one exists.
        cid = self._class_id_from_txt(base)
        if cid is not None:
            self.class_id_spin.setValue(cid)

    @staticmethod
    def _set_status_tick(label, annotated):
        """Show a solid black ✔ when annotated, blank otherwise."""
        label.setText("✔" if annotated else "")
        label.setToolTip("Annotated" if annotated else "Not annotated")

    def _make_row_widget(self, name, annotated):
        """Build one list row: [status ✔] file name … [Delete].

        The tick is a display-only black check glyph (not a control); the
        Delete button removes the image and its labels (captures the file name,
        not the row index, so it stays correct as rows shift)."""
        w   = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 1, 4, 1)
        lay.setSpacing(6)

        tick = QLabel()
        tick.setFixedWidth(16)
        # iOS-green check reads on both light and dark themes.
        tick.setStyleSheet("color: #34c759; font-weight: bold;")
        tick.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_status_tick(tick, annotated)

        lbl = QLabel(name)
        lbl.setToolTip(name)

        # Red delete control.  Segoe UI Symbol renders the trash glyph as a
        # monochrome shape that takes the red colour (the emoji font ignores it).
        btn = QPushButton("\U0001F5D1")
        btn.setObjectName("deleteBtn")
        btn.setFixedSize(30, 26)
        btn.setToolTip(f"Delete {name} and its label files (Ctrl+D)")
        btn.clicked.connect(lambda _=False, n=name: self._delete_image_by_name(n))

        lay.addWidget(tick)
        lay.addWidget(lbl, 1)
        lay.addWidget(btn)
        return w, tick

    def _populate_list(self):
        """(Re)build the list with custom row widgets, showing a black ✔ tick
        where a matching label file exists."""
        self.file_list.blockSignals(True)
        self.file_list.clear()
        self._status_boxes = []
        for name in self.image_files:
            annotated = self._find_label_for(os.path.splitext(name)[0]) is not None
            item = QListWidgetItem()
            self.file_list.addItem(item)
            w, tick = self._make_row_widget(name, annotated)
            item.setSizeHint(w.sizeHint())
            self.file_list.setItemWidget(item, w)
            self._status_boxes.append(tick)
        self.file_list.blockSignals(False)

    def _mark_current_annotated(self):
        if 0 <= self.current_index < len(self._status_boxes):
            self._set_status_tick(self._status_boxes[self.current_index], True)

    def _mark_current_unannotated(self):
        if 0 <= self.current_index < len(self._status_boxes):
            self._set_status_tick(self._status_boxes[self.current_index], False)

    def _on_row_changed(self, row):
        if row < 0 or row == self.current_index:
            return
        if not self._confirm_discard():
            self.file_list.blockSignals(True)
            self.file_list.setCurrentRow(self.current_index)
            self.file_list.blockSignals(False)
            return
        self.current_index = row
        self.load_current()

    def _confirm_discard(self):
        if not self.canvas.dirty:
            return True
        # Auto-save (default ON): save silently and never interrupt the user
        # with a dialog (spec 1 / 8).  Always proceed with navigation.
        if self.autosave_check.isChecked():
            self.save_current(silent=True)
            return True
        resp = QMessageBox.question(
            self, "Unsaved changes",
            "Save annotation before leaving this image?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)
        if resp == QMessageBox.StandardButton.Save:
            return self.save_current()
        return resp == QMessageBox.StandardButton.Discard

    def step(self, delta):
        if not self.image_files:
            return
        row = min(max(self.current_index + delta, 0),
                  len(self.image_files) - 1)
        self.file_list.setCurrentRow(row)

    def skip_image(self):
        """Space: jump to the next image with no prompt whatsoever (spec 8).

        Auto-save ON  → save the current annotation, then advance.
        Auto-save OFF → discard the current annotation, then advance.
        """
        if not self.image_files:
            return
        if self.autosave_check.isChecked():
            self.save_current(silent=True)
        # Clear the dirty flag so navigation never raises a save/discard dialog.
        self.canvas.dirty = False
        if self.current_index >= len(self.image_files) - 1:
            self.statusBar().showMessage("Already at the last image")
            self.canvas.setFocus()
            return
        self.step(1)

    # ── copy / paste an annotation across frames ──────────────────────────

    def copy_annotation(self):
        """Ctrl+C: snapshot the current annotation so it can be pasted onto
        another frame at the identical pixel coordinates."""
        c = self.canvas
        c._sync_inner_pts()
        c._sync_bat_vis()
        has_bat = (c.bat_box is not None or c.bat_top is not None
                   or c.bat_tip is not None)
        if not has_bat and not c.boxes and not c.points:
            self.statusBar().showMessage(
                "Nothing to copy — annotate first, then press Ctrl+C")
            return
        self._clipboard = {
            'bat_box':     list(c.bat_box) if c.bat_box is not None else None,
            'bat_top':     list(c.bat_top) if c.bat_top is not None else None,
            'bat_tip':     list(c.bat_tip) if c.bat_tip is not None else None,
            'bat_n':       c.bat_n,
            'bat_inner_pts': [list(p) for p in c.bat_inner_pts],
            'bat_vis':     list(c.bat_vis),
            'boxes':       [list(b) for b in c.boxes],
            'points':      [list(p) for p in c.points],
        }
        self.statusBar().showMessage(
            "Annotation copied — go to another frame and press Ctrl+V to paste")

    def paste_annotation(self):
        """Ctrl+V: restore the copied annotation onto the current image at the
        same coordinates, replacing whatever is there."""
        snap = self._clipboard
        if not snap:
            self.statusBar().showMessage(
                "Clipboard empty — copy an annotation first with Ctrl+C")
            return
        if self.current_index < 0:
            self.statusBar().showMessage("Open an image before pasting")
            return
        c = self.canvas
        # Set n first (this resets inner fractions / visibility via
        # _on_n_changed), then restore the exact copied spacing and per-keypoint
        # visibility so the paste reproduces the original geometry precisely.
        self.n_spin.setValue(snap['bat_n'])
        c.bat_n       = snap['bat_n']
        c.bat_inner_pts = [list(p) for p in snap.get('bat_inner_pts', [])]
        c.bat_vis     = list(snap['bat_vis'])
        c.boxes       = [list(b) for b in snap['boxes']]
        c.points      = [list(p) for p in snap['points']]
        c.bat_box = list(snap['bat_box']) if snap['bat_box'] is not None else None
        c.bat_top = list(snap['bat_top']) if snap['bat_top'] is not None else None
        c.bat_tip = list(snap['bat_tip']) if snap['bat_tip'] is not None else None
        c._selected  = None
        c.history    = c._canonical_history()
        c.redo_stack = []
        c.dirty      = True
        c._rebuild_state()
        self._refresh_kp_panel()
        c.setFocus()
        self.statusBar().showMessage("Annotation pasted at copied coordinates")

    # ── image deletion (cascade to label files) ──────────────────────────

    def delete_current_image(self):
        """Ctrl+D handler — delete the image currently selected in the list."""
        if 0 <= self.current_index < len(self.image_files):
            self._delete_image_by_name(self.image_files[self.current_index])

    def _delete_image_by_name(self, name):
        """Permanently delete an image and its .txt/.json labels from disk,
        then refresh the list and selection."""
        if name not in self.image_files:
            return
        resp = QMessageBox.question(
            self, "Delete image",
            f"Permanently delete '{name}' and its label files?\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if resp != QMessageBox.StandardButton.Yes:
            return

        idx              = self.image_files.index(name)
        deleting_current = (idx == self.current_index)
        base             = os.path.splitext(name)[0]
        self._remove_files_for(name, base)

        self.image_files.pop(idx)
        self._populate_list()

        if not self.image_files:
            self.current_index = -1
            self.canvas.clear_annotations()
            self.setWindowTitle("Image Annotation Tool — Bat Keypoint "
                                "Interpolator")
            self.statusBar().showMessage(f"Deleted {name} — no images left")
            return

        if deleting_current:
            # The shown image is gone; load the one that slid into its place.
            self.canvas.dirty = False          # nothing left to save for it
            self.current_index = -1            # force _on_row_changed to load
            self.file_list.setCurrentRow(min(idx, len(self.image_files) - 1))
        else:
            # A different image went away; keep the current one untouched and
            # just fix the index / highlight.
            if idx < self.current_index:
                self.current_index -= 1
            self.file_list.blockSignals(True)
            self.file_list.setCurrentRow(self.current_index)
            self.file_list.blockSignals(False)
        self.statusBar().showMessage(f"Deleted {name}")

    def _remove_label_files_for(self, base):
        """Delete any matching .txt/.json across label dirs (image untouched).

        Used when an annotation is deleted so the stale label files do not
        reload the next time the image is opened."""
        targets = []
        for d in (self.labels_dir, self.json_dir, self.user_labels_dir):
            if not d:
                continue
            targets.append(os.path.join(d, base + ".txt"))
            targets.append(os.path.join(d, base + ".json"))
        for p in targets:
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError as e:
                self.statusBar().showMessage(
                    f"Could not delete {os.path.basename(p)}: {e}")

    def _remove_files_for(self, name, base):
        """Delete the image plus any matching .txt/.json across label dirs."""
        if self.images_dir:
            p = os.path.join(self.images_dir, name)
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError as e:
                self.statusBar().showMessage(
                    f"Could not delete {os.path.basename(p)}: {e}")
        self._remove_label_files_for(base)

    def current_image_path(self):
        if self.current_index < 0:
            return None
        return os.path.join(self.images_dir,
                            self.image_files[self.current_index])

    def load_current(self):
        path = self.current_image_path()
        if path and self.canvas.load_image(path):
            self.setWindowTitle(
                f"Annotating {self.image_files[self.current_index]} "
                f"[{self.current_index + 1}/{len(self.image_files)}]")
            self._load_existing_annotation()
            # Only overwrite the status bar if no annotation was loaded
            # (otherwise the "Loaded from …" message would be lost).
            if not self.canvas.history:
                self.statusBar().showMessage(self.canvas._next_step_hint())
            self._refresh_kp_panel()
            # Keep keyboard focus on the canvas so A/D/Space keep working
            # without an extra click after each navigation.
            self.canvas.setFocus()

    # ── annotation loading ───────────────────────────────────────────────

    def _load_existing_annotation(self):
        """Auto-load when an image is selected: search the label directories
        for a matching JSON (absolute coords) or TXT (normalised → denormalised
        before rendering)."""
        if not self.image_files or self.current_index < 0:
            return
        base  = os.path.splitext(self.image_files[self.current_index])[0]
        found = self._find_label_for(base)
        if found is None:
            return
        kind, path = found
        if kind == 'json':
            self._apply_json_file(path)
        else:
            self._apply_txt_file(path)

    # ── dialog-triggered loaders (kept for programmatic / scripted use) ──────

    def _load_from_json_dialog(self):
        """Browse for any LabelMe JSON and apply it to the current image."""
        if self.current_index < 0:
            QMessageBox.warning(self, "No image",
                                "Open a folder and select an image first.")
            return
        start = self.user_labels_dir or self.json_dir or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open LabelMe JSON annotation", start,
            "JSON files (*.json);;All files (*.*)")
        if path:
            self._apply_json_file(path)

    def _load_from_txt_dialog(self):
        """Browse for any YOLO TXT file and apply it to the current image."""
        if self.current_index < 0:
            QMessageBox.warning(self, "No image",
                                "Open a folder and select an image first.")
            return
        start = self.user_labels_dir or self.labels_dir or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open YOLO TXT annotation", start,
            "Text files (*.txt);;All files (*.*)")
        if path:
            self._apply_txt_file(path)

    # ── shared file parsers ───────────────────────────────────────────────

    def _apply_json_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            # FIX: Clean keys/values to handle malformed JSON with trailing spaces
            data = self._clean_json(raw_data)
            
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Load error",
                                f"Cannot read JSON:\n{e}")
            return

        rects, pts, pts_vis = [], [], []
        rect_label = None
        for shape in data.get("shapes", []):
            stype = shape.get("shape_type", "")
            if stype == "rectangle":
                raw = shape["points"]              # [[x1,y1],[x2,y2]]
                xs  = [p[0] for p in raw]
                ys  = [p[1] for p in raw]
                rects.append([min(xs), min(ys), max(xs), max(ys)])
                if rect_label is None:
                    rect_label = shape.get("label")
            elif stype == "point":
                pts.append(list(shape["points"][0]))
                
                # Default to session default
                v = self.canvas.kp_visibility 
                
                # 1. PRIMARY: Check the 'flags' dictionary (frame_000051.json standard)
                flags = shape.get("flags", {})
                if isinstance(flags, dict):
                    if flags.get("invisible") is True:
                        v = 0
                    elif flags.get("occluded") is True:
                        v = 1
                    elif flags.get("visible") is True:
                        v = 2
                
                # 2. FALLBACK: Check the description string for both "v=" and "visibility="
                # This ensures compatibility with older files and the updated writer.
                desc = str(shape.get("description", "")).strip().lower()
                if "visibility=" in desc:
                    try: 
                        v = int(desc.split("visibility=")[1].split()[0])
                    except (ValueError, IndexError): 
                        pass
                elif "v=" in desc:
                    try: 
                        v = int(desc.split("v=")[1].split()[0])
                    except (ValueError, IndexError): 
                        pass
                        
                pts_vis.append(v)

        # Sync the export identity (class label / id) from the loaded file.
        base = os.path.splitext(os.path.basename(path))[0]
        self._auto_configure_tool(rects, pts, rect_label, base)

        actions = [('box', r) for r in rects]
        if len(pts) >= 2:
            self.n_spin.setValue(len(pts))      # resets inner fractions…
            actions.append(('bat_top', pts[0]))
            actions.append(('bat_tip', pts[-1]))
            self._set_bat_inner_from_points(pts)  # …then restore exact ones
            # Restore each keypoint's individual visibility flag.
            self.canvas.bat_vis = list(pts_vis)

        if not actions:
            self.statusBar().showMessage(
                f"{os.path.basename(path)} — no shapes found")
            return

        self.canvas.set_history(actions)
        self.statusBar().showMessage(
            f"Loaded JSON: {os.path.basename(path)}")

    def _set_bat_inner_from_points(self, pts):
        """Store the absolute positions of intermediate keypoints from loaded points."""
        self.canvas.bat_inner_pts = [list(p) for p in pts[1:-1]]

    def _apply_txt_file(self, path):
        """Parse a YOLO-pose TXT file and load the annotations into the canvas.

        YOLO stores all values normalised to [0, 1].  Denormalisation:

            x_abs = normalised_x × image_width
            y_abs = normalised_y × image_height

        BBox centre + size → corner form:
            x1 = (xc − w/2) × W,   y1 = (yc − h/2) × H
            x2 = (xc + w/2) × W,   y2 = (yc + h/2) × H
        """
        W, H = self.canvas.img_w, self.canvas.img_h
        if W == 0 or H == 0:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except OSError as e:
            QMessageBox.warning(self, "Load error",
                                f"Cannot read TXT:\n{e}")
            return

        actions = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            xc = float(parts[1]) * W
            yc = float(parts[2]) * H
            bw = float(parts[3]) * W
            bh = float(parts[4]) * H
            actions.append(('box', [xc - bw / 2, yc - bh / 2,
                                     xc + bw / 2, yc + bh / 2]))
            # Keypoints follow as triplets <px> <py> <vis> in YOLOv8-pose.
            # Fall back to pairs for older files that omit the visibility flag.
            kp = parts[5:]
            stride = 3 if (kp and len(kp) % 3 == 0) else 2
            pts, vis_list = [], []
            i = 0
            while i + 1 < len(kp):
                pts.append([float(kp[i]) * W, float(kp[i + 1]) * H])
                if stride == 3 and i + 2 < len(kp):
                    vis_list.append(int(float(kp[i + 2])))
                else:
                    vis_list.append(self.canvas.kp_visibility)
                i += stride
            if len(pts) >= 2:
                self.n_spin.setValue(len(pts))
                actions.append(('bat_top', pts[0]))
                actions.append(('bat_tip', pts[-1]))
                self._set_bat_inner_from_points(pts)
                # Restore each keypoint's individual visibility flag.
                self.canvas.bat_vis = list(vis_list)

        if not actions:
            self.statusBar().showMessage(
                f"{os.path.basename(path)} — no annotations found")
            return

        self.canvas.set_history(actions)
        self.statusBar().showMessage(
            f"Loaded TXT: {os.path.basename(path)}")

    # ── saving ───────────────────────────────────────────────────────────

    def _collect_annotation(self):
        """Return (boxes, keypoints) in image pixel coords, or raise.

        Bat is the only mode: a single bounding box plus the interpolated
        keypoints, each carrying its own visibility flag from self.bat_vis."""
        c = self.canvas
        kps = c.bat_keypoints()
        if kps is None:
            raise ValueError("Place both keypoint clicks (Top and Tip) "
                             "before saving.")
        if c.bat_box is None:
            raise ValueError("A bounding box is required — draw it first.")
        c._sync_bat_vis()
        kps3 = [[p[0], p[1], (c.bat_vis[i] if i < len(c.bat_vis) else 2)]
                for i, p in enumerate(kps)]
        return [c.bat_box], kps3

    def save_current(self, silent=False):
        """Write the current annotation to disk.

        silent=True (auto-save / skip) never raises a dialog: an incomplete
        annotation is simply skipped so navigation is never interrupted."""
        if self.current_index < 0:
            return False
        try:
            boxes, kps = self._collect_annotation()
        except ValueError as e:
            # The annotation is now empty/incomplete.  If the user actively
            # cleared it (dirty) AND a label file still exists on disk, this is
            # a deletion — remove the stale .txt/.json so it does not reload
            # when the user navigates back.  Otherwise (image untouched / never
            # annotated) just skip without raising.
            base = os.path.splitext(self.image_files[self.current_index])[0]
            if self.canvas.dirty and self._find_label_for(base) is not None:
                self._remove_label_files_for(base)
                self.canvas.dirty = False
                self._mark_current_unannotated()
                self.statusBar().showMessage(
                    f"Annotation deleted — removed {base} label files")
                return True
            if not silent:
                QMessageBox.warning(self, "Cannot save", str(e))
            else:
                self.statusBar().showMessage(f"Not saved — {e}")
            return False

        image_name = self.image_files[self.current_index]
        base       = os.path.splitext(image_name)[0]
        txt_path   = os.path.join(self.labels_dir, base + ".txt")
        json_path  = os.path.join(self.json_dir,   base + ".json")

        self._write_yolo_txt(txt_path, boxes, kps)
        self._write_labelme_json(json_path, image_name, boxes, kps)
        self.canvas.dirty = False
        self._mark_current_annotated()
        self.statusBar().showMessage(f"Saved {base}.txt + {base}.json")
        return True

    def _write_yolo_txt(self, path, boxes, kps):
        """YOLOv8-pose format (normalised to [0,1]):

            <class_id> <xc> <yc> <w> <h> <px1> <py1> <vis1> … <pxn> <pyn> <visn>

        The per-keypoint visibility flag (0/1/2) is appended after each
        coordinate pair.  If no manual bbox exists the tight extent of the
        keypoints is used, because the YOLO format always requires a bbox.
        """
        W   = self.canvas.img_w
        H   = self.canvas.img_h
        cls = self.class_id_spin.value()

        def norm(v, dim):
            return min(max(v / dim, 0.0), 1.0)

        def bbox_fields(box):
            x1, y1, x2, y2 = box
            return (norm((x1 + x2) / 2, W), norm((y1 + y2) / 2, H),
                    norm(x2 - x1, W),       norm(y2 - y1, H))

        lines = []
        if kps:
            if boxes:
                bb, extra = bbox_fields(boxes[0]), boxes[1:]
            else:
                xs = [p[0] for p in kps];  ys = [p[1] for p in kps]
                bb, extra = bbox_fields([min(xs), min(ys),
                                         max(xs), max(ys)]), []
            fields = [str(cls)] + [f"{v:.6f}" for v in bb]
            for kp in kps:
                px, py = kp[0], kp[1]
                vis    = int(kp[2]) if len(kp) > 2 else 2
                fields += [f"{norm(px, W):.6f}", f"{norm(py, H):.6f}", str(vis)]
            lines.append(" ".join(fields))
            for box in extra:
                lines.append(" ".join(
                    [str(cls)] + [f"{v:.6f}" for v in bbox_fields(box)]))
        else:
            for box in boxes:
                lines.append(" ".join(
                    [str(cls)] + [f"{v:.6f}" for v in bbox_fields(box)]))

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _write_labelme_json(self, path, image_name, boxes, kps):
        """LabelMe JSON formatted exactly like frame_000051.json.
        - Uses group_id to link bbox and keypoints.
        - Uses semantic keypoint names (bat_top, bat_middle, bat_tip).
        - Uses explicit boolean flags for visibility.
        """
        label  = self.class_name_edit.text().strip() or "bat"
        shapes = []
        
        # 1. Write the Bounding Box with group_id = 0
        for box in boxes:
            shapes.append({
                "label":       label,
                "points":      [[float(box[0]), float(box[1])],
                                [float(box[2]), float(box[3])]],
                "group_id":    0,  # <--- FIXED: Links to keypoints
                "description": "",
                "shape_type":  "rectangle",
                "flags":       {},
                "mask":        None,
            })
            
        # 2. Write Keypoints with semantic names, group_id, and explicit flags
        num_kps = len(kps)
        for i, kp in enumerate(kps):
            px, py = kp[0], kp[1]
            vis    = int(kp[2]) if len(kp) > 2 else 2
            
            # Determine semantic label based on position
            if i == 0:
                kp_label = "bat_top"
            elif i == num_kps - 1:
                kp_label = "bat_tip"
            else:
                # If there are multiple middle points, name them bat_middle_1, etc.
                kp_label = f"bat_middle_{i}" if num_kps > 3 else "bat_middle"
                
            # Map integer visibility (0,1,2) to explicit boolean flags
            if vis == 2:
                vis_flags = {"visible": True, "occluded": False, "invisible": False}
            elif vis == 1:
                vis_flags = {"visible": False, "occluded": True, "invisible": False}
            else:
                vis_flags = {"visible": False, "occluded": False, "invisible": True}

            shapes.append({
                "label":       kp_label,       # <--- FIXED: Semantic names
                "points":      [[float(px), float(py)]],
                "group_id":    0,              # <--- FIXED: Links to bbox
                "description": f"visibility={vis}", # <--- FIXED: Matches frame_000051
                "shape_type":  "point",
                "flags":       vis_flags,      # <--- FIXED: Explicit boolean flags
                "mask":        None,
            })

        # Absolute path to avoid LabelMe loading bugs
        img_abs = os.path.abspath(self.current_image_path()).replace("\\", "/")
        
        doc = {
            "version":     "5.5.0", # Updated to match frame_000051
            "flags":       {},
            "shapes":      shapes,
            "imagePath":   img_abs,
            "imageData":   None,
            "imageHeight": int(self.canvas.img_h),
            "imageWidth":  int(self.canvas.img_w),
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False) # indent=1 to match your example

    # ── theme + help ────────────────────────────────────────────────────

    def apply_theme(self, theme):
        """Apply the light/dark stylesheet app-wide and tint the canvas."""
        self.theme = theme
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(theme))
        self.canvas.setBackgroundBrush(
            QColor(28, 28, 30) if theme == 'dark' else QColor(229, 229, 234))
        # Keep the menu radio in sync (setChecked does not re-fire triggered).
        self.act_theme_dark.setChecked(theme == 'dark')
        self.act_theme_light.setChecked(theme == 'light')

    def show_shortcuts(self):
        ShortcutsDialog(self).exec()

    def closeEvent(self, event):
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()


# ---------------------------------------------------------------------------
# Theming — sleek, rounded, iOS / macOS-flavoured light & dark stylesheets
# ---------------------------------------------------------------------------
THEMES = {
    'dark': {
        'window':  '#1c1c1e', 'panel':   '#2c2c2e', 'border':  '#3a3a3c',
        'text':    '#f2f2f7', 'subtle':  '#98989f', 'accent':  '#0a84ff',
        'accent2': '#409cff', 'input':   '#3a3a3c', 'hover':   '#48484a',
        'sel_txt': '#ffffff', 'menubar': '#161618', 'shadow':  '#000000',
    },
    'light': {
        'window':  '#f2f2f7', 'panel':   '#ffffff', 'border':  '#d6d6db',
        'text':    '#1c1c1e', 'subtle':  '#6c6c70', 'accent':  '#0a84ff',
        'accent2': '#0060df', 'input':   '#ffffff', 'hover':   '#e5e5ea',
        'sel_txt': '#ffffff', 'menubar': '#ffffff', 'shadow':  '#c8c8cd',
    },
}


def build_stylesheet(theme):
    """Return a full Qt Style Sheet for the named theme ('dark' / 'light').

    Everything is generously rounded (8–14 px) with soft borders and an iOS
    blue accent, for a clean, modern, 'Mac-app' feel."""
    c = THEMES.get(theme, THEMES['dark'])
    return f"""
    * {{
        font-family: 'Segoe UI', '-apple-system', 'Helvetica Neue', Arial;
        font-size: 13px;
        color: {c['text']};
    }}
    QMainWindow, QDialog {{ background: {c['window']}; }}

    QWidget#sidePanel {{
        background: {c['window']};
        border-left: 1px solid {c['border']};
    }}

    QLabel {{ background: transparent; }}
    QFrame#legend {{
        background: {c['panel']};
        border: 1px solid {c['border']};
        border-radius: 12px;
    }}
    QFrame#legend QLabel {{ font-size: 13px; }}
    QLabel#legendTitle {{ font-weight: 700; font-size: 13px; }}
    QLabel#hint {{ color: {c['subtle']}; font-size: 12px; }}
    QLabel#kpEmpty {{
        color: {c['subtle']};
        font-size: 12px;
        padding: 10px 8px;
    }}
    QLabel#kpName {{ font-size: 12px; }}

    QGroupBox {{
        background: {c['panel']};
        border: 1px solid {c['border']};
        border-radius: 14px;
        margin-top: 14px;
        padding: 14px 14px 12px 14px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 14px;
        padding: 0 6px;
        color: {c['subtle']};
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    QComboBox, QSpinBox, QLineEdit {{
        background: {c['input']};
        border: 1px solid {c['border']};
        border-radius: 9px;
        padding: 6px 10px;
        min-height: 20px;
        selection-background-color: {c['accent']};
        selection-color: {c['sel_txt']};
    }}
    QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{
        border: 1px solid {c['accent']};
    }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {c['subtle']};
        margin-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background: {c['panel']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        padding: 4px;
        outline: none;
        selection-background-color: {c['accent']};
        selection-color: {c['sel_txt']};
    }}
    QSpinBox::up-button, QSpinBox::down-button {{
        width: 18px; border: none; background: transparent;
    }}

    QPushButton {{
        background: {c['input']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        padding: 8px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{ background: {c['hover']}; }}
    QPushButton:pressed {{ background: {c['accent']}; color: {c['sel_txt']}; }}

    QPushButton#editBtn:checked {{
        background: {c['accent']};
        border: 1px solid {c['accent']};
        color: {c['sel_txt']};
    }}

    /* Segmented 0 / 1 / 2 visibility toggle in the keypoint panel. */
    QPushButton#segBtn {{
        background: {c['input']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 0;
        font-weight: 700;
        font-size: 12px;
    }}
    QPushButton#segBtn:hover {{ background: {c['hover']}; }}
    QPushButton#segBtn:checked {{
        background: {c['accent']};
        border: 1px solid {c['accent']};
        color: {c['sel_txt']};
    }}

    /* Red trash control on each image row. */
    QPushButton#deleteBtn {{
        background: transparent;
        border: none;
        border-radius: 8px;
        color: #ff3b30;
        font-family: 'Segoe UI Symbol';
        font-size: 14px;
        padding: 0;
    }}
    QPushButton#deleteBtn:hover {{ background: rgba(255, 59, 48, 0.18); }}
    QPushButton#deleteBtn:pressed {{ background: rgba(255, 59, 48, 0.34); }}

    QListWidget {{
        background: {c['panel']};
        border: 1px solid {c['border']};
        border-radius: 12px;
        padding: 4px;
        outline: none;
    }}
    QListWidget::item {{
        border-radius: 8px;
        padding: 2px;
        margin: 1px 2px;
    }}
    QListWidget::item:selected {{
        background: {c['accent']};
        color: {c['sel_txt']};
    }}
    QListWidget::item:hover {{ background: {c['hover']}; }}

    QScrollArea {{ background: transparent; border: none; }}
    /* Keep the keypoint scroll viewport + holder transparent (no grey card). */
    QScrollArea#kpScroll, QScrollArea#kpScroll > QWidget,
    QWidget#kpHolder {{ background: transparent; }}

    QMenuBar {{
        background: {c['menubar']};
        border-bottom: 1px solid {c['border']};
        padding: 3px 6px;
    }}
    QMenuBar::item {{
        background: transparent; padding: 6px 12px; border-radius: 8px;
    }}
    QMenuBar::item:selected {{ background: {c['hover']}; }}
    QMenu {{
        background: {c['panel']};
        border: 1px solid {c['border']};
        border-radius: 12px;
        padding: 6px;
    }}
    QMenu::item {{ padding: 7px 26px 7px 18px; border-radius: 8px; }}
    QMenu::item:selected {{ background: {c['accent']}; color: {c['sel_txt']}; }}
    QMenu::item:disabled {{ color: {c['subtle']}; }}
    QMenu::separator {{ height: 1px; background: {c['border']}; margin: 5px 8px; }}

    QStatusBar {{
        background: {c['menubar']};
        border-top: 1px solid {c['border']};
        color: {c['subtle']};
        padding: 3px 8px;
    }}
    QStatusBar::item {{ border: none; }}

    QScrollBar:vertical {{
        background: transparent; width: 11px; margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {c['border']}; border-radius: 5px; min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {c['subtle']}; }}
    QScrollBar:horizontal {{
        background: transparent; height: 11px; margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c['border']}; border-radius: 5px; min-width: 28px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {c['subtle']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QToolTip {{
        background: {c['panel']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 6px 9px;
    }}

    QLabel#kbd {{
        background: {c['input']};
        border: 1px solid {c['border']};
        border-radius: 7px;
        padding: 4px 10px;
        font-weight: 700;
        font-family: 'Consolas', 'SF Mono', monospace;
    }}
    QLabel#shortcutDesc {{ color: {c['text']}; font-size: 13px; }}
    QLabel#dialogTitle {{ font-size: 18px; font-weight: 700; }}
    """


# ---------------------------------------------------------------------------
# Keyboard-shortcuts reference dialog
# ---------------------------------------------------------------------------
SHORTCUTS = [
    ("A",             "Previous image"),
    ("D",             "Next image"),
    ("Space",         "Skip image — no prompt (auto-save: save, else discard)"),
    ("Backspace",     "Delete the selected annotation"),
    ("Delete",        "Delete all annotations on the frame (Ctrl+Z to undo)"),
    ("Ctrl + C",      "Copy the current annotation"),
    ("Ctrl + V",      "Paste the copied annotation (same coordinates)"),
    ("Ctrl + E",      "Toggle Edit / Resize mode"),
    ("Ctrl + S",      "Save current annotation"),
    ("Ctrl + Z",      "Undo"),
    ("Ctrl + Y",      "Redo"),
    ("Ctrl + F",      "Fit image to screen"),
    ("B",             "Zoom to the bounding box"),
    ("Ctrl + O",      "Open image directory"),
    ("Ctrl + D",      "Delete current image file (+ its labels)"),
    ("Ctrl + T",      "Toggle dark / light theme"),
    ("0 / 1 / 2",     "Set visibility of the selected keypoint"),
    ("Esc",           "Cancel the current operation (never deletes)"),
    ("Ctrl + Scroll", "Zoom in / out (under the cursor)"),
    ("Shift + Drag",  "Pan the view (hand / grab) — any mode"),
    ("Shift + Scroll","Pan horizontally"),
    ("Scroll",        "Pan vertically"),
    ("Right-click KP","Set that keypoint's visibility"),
    ("F1",            "Show this shortcuts list"),
]


class ShortcutsDialog(QDialog):
    """A clean, rounded reference of every keyboard shortcut."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(460)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(26, 22, 26, 22)
        outer.setSpacing(16)

        title = QLabel("Keyboard Shortcuts")
        title.setObjectName("dialogTitle")
        outer.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        for row, (keys, desc) in enumerate(SHORTCUTS):
            kbd = QLabel(keys)
            kbd.setObjectName("kbd")
            kbd.setAlignment(Qt.AlignmentFlag.AlignCenter)
            kbd.setSizePolicy(QSizePolicy.Policy.Maximum,
                              QSizePolicy.Policy.Preferred)
            d = QLabel(desc)
            d.setObjectName("shortcutDesc")
            d.setWordWrap(True)
            grid.addWidget(kbd, row, 0)
            grid.addWidget(d,   row, 1)
        outer.addLayout(grid)

        btn = QPushButton("Got it")
        btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn)
        outer.addLayout(row)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
