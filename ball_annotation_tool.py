"""
Ball Annotation Desktop Application
PyQt6 + OpenCV tool for annotating balls with bounding boxes only.

This is a simplified ball version of your bat annotation tool:
- No keypoints
- No keypoint visibility
- No top/tip interpolation
- Only bounding boxes are annotated and saved

Outputs:
<parent>/labels/<name>.txt       YOLO format: class xc yc w h
<parent>/json_labels/<name>.json LabelMe format: rectangle shapes

Keys:
A/D prev/next
Space skip
Backspace delete selected
Delete delete all
Ctrl+C copy
Ctrl+V paste
Ctrl+E edit mode
Ctrl+S save
Ctrl+Z undo
Ctrl+Y redo
Ctrl+F fit
B zoom to bbox
ESC cancel
Ctrl+Scroll zoom
Shift+left-drag pan
Shift+Scroll horizontal pan
Scroll vertical pan
"""

import json
import os
import sys

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# A mouse move smaller than this, in viewport pixels, is treated as a click.
CLICK_TOLERANCE = 6

# Edit-mode hit radii, in viewport pixels.
CORNER_HIT_RADIUS = 10
EDGE_HIT_RADIUS = 7

# Zoom multiplier per Ctrl+Scroll notch.
ZOOM_FACTOR = 1.15

# Bounding-box pen width in screen pixels. Cosmetic pen keeps it constant.
BOX_PEN_WIDTH = 4

# Screen-pixel half-size of an edit handle square.
HANDLE_HALF = 4

# Selection colour.
SELECT_COLOR = QColor(10, 132, 255)

# If False, drawing a new box replaces the existing one.
# If True, the tool allows multiple ball boxes per image.
ALLOW_MULTIPLE_BOXES = True


class HandleItem(QGraphicsItem):
    """
    Small filled square at a box corner/edge, fixed size in screen pixels.

    This is only a visual affordance. Hit-testing is done independently in
    the view's mouse handlers using viewport coordinates.
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
        painter.drawRect(
            QRectF(
                -HANDLE_HALF,
                -HANDLE_HALF,
                2 * HANDLE_HALF,
                2 * HANDLE_HALF,
            )
        )
        painter.setBrush(QBrush())


class AnnotationView(QGraphicsView):
    """
    Image canvas with zoom/pan, crosshairs, box drawing, and box editing.

    Scene coordinates are image pixel coordinates.
    Annotation state is stored in self.history as:
        ('box', [x1, y1, x2, y2])
    """

    def __init__(self, status_cb, parent=None):
        super().__init__(parent)

        self.status_cb = status_cb

        # Navigation hooks wired by MainWindow.
        self.request_prev = None
        self.request_next = None
        self.request_skip = None

        # Copy/paste hooks wired by MainWindow.
        self.request_copy = None
        self.request_paste = None

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # View configuration.
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        # Zoom pivot is computed manually in wheelEvent (see ZOOM_FACTOR /
        # wheelEvent below) using viewport-local coordinates. AnchorUnderMouse
        # relies on QCursor::pos() (global desktop coords), which can drift
        # out of sync with the viewport on multi-monitor / HiDPI setups and
        # made the zoom pivot jump to the wrong spot.
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.NoAnchor
        )
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter
        )

        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.setBackgroundBrush(QColor(35, 35, 40))
        self.setMinimumSize(400, 300)

        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        # Persistent crosshair items.
        _ch_pen = QPen(QColor(0, 0, 0), 2.0)
        _ch_pen.setCosmetic(True)

        self._crosshair_h = QGraphicsLineItem()
        self._crosshair_v = QGraphicsLineItem()

        for _ch in (self._crosshair_h, self._crosshair_v):
            _ch.setPen(QPen(_ch_pen))
            _ch.setZValue(30)
            _ch.setVisible(False)
            self._scene.addItem(_ch)

        # Rubber-band rectangle used while drawing.
        _rb_pen = QPen(QColor(0, 220, 90), BOX_PEN_WIDTH, Qt.PenStyle.SolidLine)
        _rb_pen.setCosmetic(True)

        self._rubber_rect = QGraphicsRectItem()
        self._rubber_rect.setPen(_rb_pen)
        self._rubber_rect.setBrush(QBrush())
        self._rubber_rect.setZValue(25)
        self._rubber_rect.setVisible(False)
        self._scene.addItem(self._rubber_rect)

        self._pixmap_item = None
        self._annotation_items = []

        # Image size.
        self.img_w = 0
        self.img_h = 0

        # Annotation state.
        self.boxes = []
        self.selected_box = None

        self.history = []
        self.redo_stack = []

        # Snapshot for undoing Delete All.
        self._cleared_snapshot = None

        self.edit_mode = False

        # Two-click drawing.
        self.pending_corner = None

        # Mouse state.
        self._press_vp = None
        self._press_scene_pt = None
        self._press_hit_box = None

        # Pan state.
        self._panning = False
        self._pan_last_vp = None

        # Edit state.
        self._edit_target = None
        self._edit_last_scene = None
        self._edit_moved = False
        self._drag_snapshot = None
        self._drag_dirty_snapshot = False
        self._dragging = False

        self.dirty = False

    # ── Image loading ────────────────────────────────────────────────

    def load_image(self, path):
        """Load image via OpenCV. np.fromfile handles Unicode paths."""
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            self.status_cb(f"Failed to load {path}")
            return False

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        qimg = QImage(
            img.data,
            w,
            h,
            3 * w,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimg.copy())

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
        self.history = []
        self.redo_stack = []
        self.dirty = False
        self.selected_box = None
        self._cleared_snapshot = None
        self._rebuild_state()

    def set_history(self, actions):
        """Install a pre-built action list used when loading saved files."""
        self.history = list(actions)
        self.redo_stack = []
        self.dirty = False
        self.selected_box = None
        self._cleared_snapshot = None
        self._rebuild_state()

    def set_edit_mode(self, on):
        self.edit_mode = bool(on)

        self._edit_target = None
        self.selected_box = None
        self.pending_corner = None

        self._rubber_rect.setVisible(False)

        if self.edit_mode:
            self._crosshair_h.setVisible(False)
            self._crosshair_v.setVisible(False)

        self._update_idle_cursor()
        self._rebuild_scene()

    # ── State / scene management ─────────────────────────────────────

    def _rebuild_state(self):
        """Replay history and update data fields."""
        self.boxes = []

        for kind, data in self.history:
            if kind == "box":
                self.boxes.append(list(data))

        if not ALLOW_MULTIPLE_BOXES and len(self.boxes) > 1:
            # In single-ball mode, only the latest box is shown.
            self.boxes = self.boxes[-1:]

        if self.selected_box is not None:
            if not (0 <= self.selected_box < len(self.boxes)):
                self.selected_box = None

        self._rebuild_scene()

    def _rebuild_scene(self):
        """Remove annotation QGraphicsItems and recreate them from state."""
        for item in self._annotation_items:
            self._scene.removeItem(item)
        self._annotation_items.clear()

        if self._pixmap_item is None:
            return

        def add(item):
            self._scene.addItem(item)
            self._annotation_items.append(item)

        def cpen(color, style=Qt.PenStyle.SolidLine, width=1.0):
            p = QPen(color, width, style)
            p.setCosmetic(True)
            return p

        box_pen = cpen(QColor(0, 220, 90), width=BOX_PEN_WIDTH)
        sel_pen = cpen(SELECT_COLOR, width=BOX_PEN_WIDTH + 1)

        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)

            r = QGraphicsRectItem(rect)

            if i == self.selected_box:
                r.setPen(sel_pen)
                r.setBrush(QBrush(QColor(10, 132, 255, 28)))
                r.setZValue(6)
            else:
                r.setPen(box_pen)
                r.setBrush(QBrush())
                r.setZValue(5)

            add(r)

            if self.edit_mode:
                self._add_box_handles(box, add)

    def _add_box_handles(self, box, add):
        """Place eight handle squares: four corners and four edge midpoints."""
        x1, y1, x2, y2 = box
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        handles = (
            (x1, y1),
            (x2, y1),
            (x1, y2),
            (x2, y2),
            (mx, y1),
            (mx, y2),
            (x1, my),
            (x2, my),
        )

        for hx, hy in handles:
            h = HandleItem()
            h.setPos(hx, hy)
            add(h)

    def _canonical_history(self):
        """Build a minimal history equivalent to the current state."""
        return [("box", list(box)) for box in self.boxes]

    def _record_box(self, box):
        """Commit a newly drawn box."""
        x1, y1, x2, y2 = box
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

        if box[2] <= box[0] or box[3] <= box[1]:
            return

        if ALLOW_MULTIPLE_BOXES:
            self.history.append(("box", box))
        else:
            self.history = [("box", box)]

        self.redo_stack = []
        self._cleared_snapshot = None
        self.dirty = True
        self.selected_box = None

        self._rebuild_state()

    # ── Undo / redo / cancel ─────────────────────────────────────────

    def undo(self):
        if not self.history:
            if self._cleared_snapshot is not None:
                self.history = list(self._cleared_snapshot)
                self._cleared_snapshot = None
                self.redo_stack = []
                self.dirty = True
                self.selected_box = None
                self._rebuild_state()
                self.status_cb("Restored all annotations")
                return

            self.status_cb("Nothing to undo")
            return

        self.redo_stack.append(self.history.pop())
        self.dirty = True
        self.selected_box = None
        self._rebuild_state()
        self.status_cb("Undo")

    def redo(self):
        if not self.redo_stack:
            self.status_cb("Nothing to redo")
            return

        self.history.append(self.redo_stack.pop())
        self.dirty = True
        self._rebuild_state()
        self.status_cb("Redo")

    def cancel_operation(self):
        """ESC: cancel current operation without deleting committed boxes."""
        # Cancel edit drag.
        if (
            self.edit_mode
            and self._edit_target is not None
            and self._drag_snapshot is not None
        ):
            self.boxes = [list(b) for b in self._drag_snapshot]
            self._edit_target = None
            self._drag_snapshot = None
            self._dragging = False
            self.dirty = self._drag_dirty_snapshot
            self._rebuild_scene()
            self.status_cb("Drag cancelled")
            return

        # Cancel two-click box.
        if self.pending_corner is not None:
            self.pending_corner = None
            self._rubber_rect.setVisible(False)
            self.status_cb("Bounding-box creation cancelled")
            return

        # Cancel rubber-band drag.
        if self._press_scene_pt is not None or self._rubber_rect.isVisible():
            self._press_vp = None
            self._press_scene_pt = None
            self._press_hit_box = None
            self._rubber_rect.setVisible(False)
            self.status_cb("Drawing cancelled")
            return

        # Clear selection.
        if self.selected_box is not None:
            self.selected_box = None
            self._rebuild_scene()
            self.status_cb("Selection cleared")
            return

        self.status_cb("Nothing to cancel")

    # ── Selection / deletion ─────────────────────────────────────────

    def _set_selection(self, index):
        if index == self.selected_box:
            return

        self.selected_box = index
        self._rebuild_scene()

    def delete_selected(self):
        if self.selected_box is None:
            self.status_cb(
                "Nothing selected — click a box in Edit mode, "
                "or right-click a box, then press Backspace"
            )
            return

        if 0 <= self.selected_box < len(self.boxes):
            self.boxes.pop(self.selected_box)

        self.selected_box = None
        self.history = self._canonical_history()
        self.redo_stack = []
        self._cleared_snapshot = None
        self.dirty = True

        self._rebuild_state()
        self.status_cb("Box deleted")

    def delete_all(self):
        if not self.boxes:
            self.status_cb("Nothing to delete")
            return

        self._cleared_snapshot = self._canonical_history()
        self.history = []
        self.redo_stack = []
        self.selected_box = None
        self.dirty = True

        self._rebuild_state()
        self.status_cb("All boxes deleted — Ctrl+Z to undo")

    # ── Zoom / fit ───────────────────────────────────────────────────

    def wheelEvent(self, event):
        mods = event.modifiers()
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
        if self.img_w > 0 and self.img_h > 0:
            self.fitInView(
                QRectF(0, 0, self.img_w, self.img_h),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def zoom_to_bbox(self):
        if not self.boxes:
            self.status_cb("No bbox yet — draw a bounding box first")
            return

        if self.selected_box is not None and 0 <= self.selected_box < len(self.boxes):
            box = self.boxes[self.selected_box]
        else:
            box = self.boxes[0]

        x1, y1, x2, y2 = box
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)

        pad_x = w * 0.15
        pad_y = h * 0.15

        self.fitInView(
            QRectF(x1 - pad_x, y1 - pad_y, w + 2 * pad_x, h + 2 * pad_y),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.status_cb("Zoomed to bbox")

    # ── Coordinate helpers ───────────────────────────────────────────

    def _scene_pt(self, viewport_qpoint):
        """Viewport QPoint to image-pixel [x, y], clamped to image bounds."""
        sp = self.mapToScene(viewport_qpoint)
        return [
            min(max(sp.x(), 0.0), max(self.img_w - 1.0, 0.0)),
            min(max(sp.y(), 0.0), max(self.img_h - 1.0, 0.0)),
        ]

    def _vp_dist(self, viewport_qpoint, scene_xy):
        screen = self.mapFromScene(QPointF(scene_xy[0], scene_xy[1]))
        return (
            QPointF(viewport_qpoint) - QPointF(screen)
        ).manhattanLength()

    # ── Hit testing ──────────────────────────────────────────────────

    def _hit_test_box_interior(self, viewport_qpoint):
        """Return index of the box whose interior contains the cursor."""
        vx, vy = viewport_qpoint.x(), viewport_qpoint.y()

        for i in reversed(range(len(self.boxes))):
            box = self.boxes[i]
            tl = self.mapFromScene(QPointF(box[0], box[1]))
            br = self.mapFromScene(QPointF(box[2], box[3]))

            left, right = min(tl.x(), br.x()), max(tl.x(), br.x())
            top, bottom = min(tl.y(), br.y()), max(tl.y(), br.y())

            if left < vx < right and top < vy < bottom:
                return i

        return None

    def _hit_test_edit(self, viewport_qpoint):
        """
        Return edit grab descriptor under cursor, or None.

        Returned form:
            ('box', box_index, handle)

        handle is one of:
            'move', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'
        """
        vx, vy = viewport_qpoint.x(), viewport_qpoint.y()

        for i in reversed(range(len(self.boxes))):
            box = self.boxes[i]

            tl = self.mapFromScene(QPointF(box[0], box[1]))
            br = self.mapFromScene(QPointF(box[2], box[3]))

            left, right = min(tl.x(), br.x()), max(tl.x(), br.x())
            top, bottom = min(tl.y(), br.y()), max(tl.y(), br.y())

            corners = (
                ("nw", left, top),
                ("ne", right, top),
                ("sw", left, bottom),
                ("se", right, bottom),
            )

            for handle, cx, cy in corners:
                if (
                    abs(vx - cx) <= CORNER_HIT_RADIUS
                    and abs(vy - cy) <= CORNER_HIT_RADIUS
                ):
                    return ("box", i, handle)

            in_x = left - EDGE_HIT_RADIUS <= vx <= right + EDGE_HIT_RADIUS
            in_y = top - EDGE_HIT_RADIUS <= vy <= bottom + EDGE_HIT_RADIUS

            if in_y and abs(vx - left) <= EDGE_HIT_RADIUS:
                return ("box", i, "w")

            if in_y and abs(vx - right) <= EDGE_HIT_RADIUS:
                return ("box", i, "e")

            if in_x and abs(vy - top) <= EDGE_HIT_RADIUS:
                return ("box", i, "n")

            if in_x and abs(vy - bottom) <= EDGE_HIT_RADIUS:
                return ("box", i, "s")

            if left < vx < right and top < vy < bottom:
                return ("box", i, "move")

        return None

    _CURSORS = {
        "nw": Qt.CursorShape.SizeFDiagCursor,
        "se": Qt.CursorShape.SizeFDiagCursor,
        "ne": Qt.CursorShape.SizeBDiagCursor,
        "sw": Qt.CursorShape.SizeBDiagCursor,
        "n": Qt.CursorShape.SizeVerCursor,
        "s": Qt.CursorShape.SizeVerCursor,
        "e": Qt.CursorShape.SizeHorCursor,
        "w": Qt.CursorShape.SizeHorCursor,
        "move": Qt.CursorShape.SizeAllCursor,
    }

    def _cursor_for_target(self, target):
        if target is None:
            return Qt.CursorShape.ArrowCursor

        return self._CURSORS.get(target[2], Qt.CursorShape.ArrowCursor)

    def _update_idle_cursor(self):
        if self._pixmap_item is None:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        elif self.edit_mode:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    # ── Edit mutation ────────────────────────────────────────────────

    def _apply_edit(self, scene_pt):
        if self._edit_target is None:
            return

        kind = self._edit_target[0]

        if kind == "box":
            i = self._edit_target[1]
            handle = self._edit_target[2]

            if 0 <= i < len(self.boxes):
                self._modify_box(self.boxes[i], handle, scene_pt)

        self._edit_last_scene = list(scene_pt)

    def _modify_box(self, box, handle, sp):
        """Resize or move a box in place, clamped to image bounds."""
        x, y = sp[0], sp[1]

        if handle == "move":
            dx = x - self._edit_last_scene[0]
            dy = y - self._edit_last_scene[1]

            w = box[2] - box[0]
            h = box[3] - box[1]

            nx1 = min(max(box[0] + dx, 0.0), max(0.0, self.img_w - w))
            ny1 = min(max(box[1] + dy, 0.0), max(0.0, self.img_h - h))

            box[0], box[1] = nx1, ny1
            box[2], box[3] = nx1 + w, ny1 + h
            return

        if "w" in handle:
            box[0] = min(max(x, 0.0), box[2] - 1.0)

        if "e" in handle:
            box[2] = max(
                min(x, float(max(self.img_w - 1, 0))),
                box[0] + 1.0,
            )

        if "n" in handle:
            box[1] = min(max(y, 0.0), box[3] - 1.0)

        if "s" in handle:
            box[3] = max(
                min(y, float(max(self.img_h - 1, 0))),
                box[1] + 1.0,
            )

    # ── Context menu ─────────────────────────────────────────────────

    def contextMenuEvent(self, event):
        if self._pixmap_item is None:
            return

        target = self._hit_test_edit(event.pos())

        if target is None or target[0] != "box":
            return

        i = target[1]
        self._set_selection(i)

        menu = QMenu(self)

        header = menu.addAction(f"Box {i + 1}")
        header.setEnabled(False)

        menu.addSeparator()

        delete_act = menu.addAction("Delete box    (Backspace)")
        delete_act.triggered.connect(self.delete_selected)

        menu.exec(event.globalPos())

    # ── Mouse events ─────────────────────────────────────────────────

    def leaveEvent(self, event):
        self._crosshair_h.setVisible(False)
        self._crosshair_v.setVisible(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if self._pixmap_item is None:
            super().mousePressEvent(event)
            return

        # Shift + left-drag pans the view.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self._panning = True
            self._pan_last_vp = event.pos()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        # Edit mode: grab box handles or interior.
        if self.edit_mode:
            self._edit_target = self._hit_test_edit(event.pos())
            self._edit_moved = False

            if self._edit_target is not None and self._edit_target[0] == "box":
                self._set_selection(self._edit_target[1])
            else:
                self._set_selection(None)

            if self._edit_target is not None:
                self._edit_last_scene = self._scene_pt(event.pos())
                self._drag_snapshot = [list(b) for b in self.boxes]
                self._drag_dirty_snapshot = self.dirty
                self._dragging = True
                self.viewport().setCursor(
                    self._cursor_for_target(self._edit_target)
                )

            return

        # Draw mode.
        self._press_vp = event.pos()
        self._press_scene_pt = self._scene_pt(event.pos())
        self._press_hit_box = None

        if self.pending_corner is None:
            self._press_hit_box = self._hit_test_box_interior(event.pos())

    def mouseMoveEvent(self, event):
        if self._pixmap_item is None:
            super().mouseMoveEvent(event)
            return

        # Pan.
        if self._panning and (event.buttons() & Qt.MouseButton.LeftButton):
            delta = event.pos() - self._pan_last_vp
            self._pan_last_vp = event.pos()

            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()

            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())

            event.accept()
            return

        sp = self.mapToScene(event.pos())
        lmb = bool(event.buttons() & Qt.MouseButton.LeftButton)

        # Edit mode.
        if self.edit_mode:
            if self._edit_target is not None and lmb:
                self._apply_edit(self._scene_pt(event.pos()))
                self._edit_moved = True
                self.dirty = True
                self._rebuild_scene()
            elif not lmb:
                target = self._hit_test_edit(event.pos())
                self.viewport().setCursor(self._cursor_for_target(target))

            return

        # Crosshair.
        in_img = (
            0 <= sp.x() <= self.img_w
            and 0 <= sp.y() <= self.img_h
        )

        if in_img:
            self._crosshair_h.setLine(0.0, sp.y(), float(self.img_w), sp.y())
            self._crosshair_v.setLine(sp.x(), 0.0, sp.x(), float(self.img_h))
            self._crosshair_h.setVisible(True)
            self._crosshair_v.setVisible(True)
        else:
            self._crosshair_h.setVisible(False)
            self._crosshair_v.setVisible(False)

        # Drag-draw rubber band.
        if self._press_scene_pt is not None and lmb:
            p1 = self._press_scene_pt
            p2 = self._scene_pt(event.pos())

            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])

            self._rubber_rect.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            self._rubber_rect.setVisible(True)

        # Two-click corner preview.
        elif self.pending_corner is not None and not lmb:
            c = self.pending_corner

            cx = min(max(sp.x(), 0.0), float(self.img_w))
            cy = min(max(sp.y(), 0.0), float(self.img_h))

            x1, y1 = min(c[0], cx), min(c[1], cy)
            x2, y2 = max(c[0], cx), max(c[1], cy)

            self._rubber_rect.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            self._rubber_rect.setVisible(True)

        else:
            self._rubber_rect.setVisible(False)

    def mouseReleaseEvent(self, event):
        if (
            self._pixmap_item is None
            or event.button() != Qt.MouseButton.LeftButton
        ):
            super().mouseReleaseEvent(event)
            return

        # End pan.
        if self._panning:
            self._panning = False
            self._pan_last_vp = None
            self.viewport().unsetCursor()
            event.accept()
            return

        # End edit drag.
        if self.edit_mode:
            self._dragging = False
            self._drag_snapshot = None

            if self._edit_target is not None and self._edit_moved:
                # Normalise any inverted boxes.
                for box in self.boxes:
                    x1, y1, x2, y2 = box
                    box[0], box[2] = min(x1, x2), max(x1, x2)
                    box[1], box[3] = min(y1, y2), max(y1, y2)

                self.history = self._canonical_history()
                self.redo_stack = []
                self._cleared_snapshot = None
                self.dirty = True

                self._rebuild_state()
                self.status_cb("Box edited")

            self._edit_target = None
            self._edit_moved = False

            self.viewport().setCursor(
                self._cursor_for_target(self._hit_test_edit(event.pos()))
            )

            return

        moved = 0.0
        if self._press_vp is not None:
            moved = (
                QPointF(event.pos()) - QPointF(self._press_vp)
            ).manhattanLength()

        # Click behaviour.
        if moved <= CLICK_TOLERANCE:
            if self.pending_corner is not None:
                self._handle_click(self._scene_pt(event.pos()))
            elif self._press_hit_box is not None:
                self._set_selection(self._press_hit_box)
                self.status_cb(
                    f"Box {self._press_hit_box + 1} selected — "
                    "Backspace deletes, Ctrl+E edits"
                )
            else:
                self._handle_click(self._scene_pt(event.pos()))

        # Drag-draw behaviour.
        elif self._press_scene_pt is not None:
            p1 = self._press_scene_pt
            p2 = self._scene_pt(event.pos())

            box = [
                min(p1[0], p2[0]),
                min(p1[1], p2[1]),
                max(p1[0], p2[0]),
                max(p1[1], p2[1]),
            ]

            if box[2] > box[0] and box[3] > box[1]:
                self.pending_corner = None
                self._record_box(box)
                self.status_cb("Bounding box created")

        self._press_vp = None
        self._press_scene_pt = None
        self._press_hit_box = None

        if self.pending_corner is None:
            self._rubber_rect.setVisible(False)

    # ── Keyboard ─────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
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
        else:
            super().keyPressEvent(event)
            return

        event.accept()

    # ── Click routing ────────────────────────────────────────────────

    def _handle_click(self, pt):
        if self.pending_corner is None:
            self.pending_corner = pt
            self._rubber_rect.setRect(QRectF(pt[0], pt[1], 0.0, 0.0))
            self._rubber_rect.setVisible(True)
            self.status_cb("BBox: click the opposite corner, or ESC to cancel")
            return

        c = self.pending_corner
        self.pending_corner = None

        box = [
            min(c[0], pt[0]),
            min(c[1], pt[1]),
            max(c[0], pt[0]),
            max(c[1], pt[1]),
        ]

        if box[2] > box[0] and box[3] > box[1]:
            self._record_box(box)
            self.status_cb("Bounding box created")
        else:
            self.status_cb("Box too small")

        self._rubber_rect.setVisible(False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ball Annotation Tool")
        self.resize(1400, 880)

        self.images_dir = None
        self.labels_dir = None
        self.json_dir = None
        self.user_labels_dir = None

        self.image_files = []
        self.current_index = -1

        self._clipboard = []

        self._build_menu()
        self._build_ui()
        self._build_shortcuts()
        self._sync_edit_controls()

    # ── Menu bar ─────────────────────────────────────────────────────

    def _build_menu(self):
        bar = self.menuBar()

        # File menu.
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

        self.autosave_check = QAction("Auto-save on navigate", self)
        self.autosave_check.setCheckable(True)
        self.autosave_check.setChecked(True)
        file_menu.addAction(self.autosave_check)

        file_menu.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Edit menu.
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

        act_cancel = QAction("Cancel Current Operation\tEsc", self)
        act_cancel.triggered.connect(lambda: self.canvas.cancel_operation())
        edit_menu.addAction(act_cancel)

        edit_menu.addSeparator()

        act_del_sel = QAction("Delete Selected Annotation\tBackspace", self)
        act_del_sel.triggered.connect(lambda: self.canvas.delete_selected())
        edit_menu.addAction(act_del_sel)

        act_del_all = QAction("Delete All Annotations\tDelete", self)
        act_del_all.triggered.connect(lambda: self.canvas.delete_all())
        edit_menu.addAction(act_del_all)

        act_clear = QAction("Clear Annotations", self)
        act_clear.triggered.connect(lambda: self.canvas.clear_annotations())
        edit_menu.addAction(act_clear)

        edit_menu.addSeparator()

        act_copy = QAction("Copy Annotation\tCtrl+C", self)
        act_copy.triggered.connect(self.copy_annotation)
        edit_menu.addAction(act_copy)

        act_paste = QAction("Paste Annotation\tCtrl+V", self)
        act_paste.triggered.connect(self.paste_annotation)
        edit_menu.addAction(act_paste)

        # View menu.
        view_menu = bar.addMenu("&View")

        act_fit = QAction("Fit to Screen", self)
        act_fit.setShortcut("Ctrl+F")
        act_fit.triggered.connect(lambda: self.canvas.fit_to_screen())
        view_menu.addAction(act_fit)

        act_zoom_box = QAction("Zoom to BBox\tB", self)
        act_zoom_box.triggered.connect(lambda: self.canvas.zoom_to_bbox())
        view_menu.addAction(act_zoom_box)

        view_menu.addSeparator()

        act_prev = QAction("Previous Image\tA", self)
        act_prev.triggered.connect(lambda: self.step(-1))
        view_menu.addAction(act_prev)

        act_next = QAction("Next Image\tD", self)
        act_next.triggered.connect(lambda: self.step(1))
        view_menu.addAction(act_next)

        act_skip = QAction("Skip Image\tSpace", self)
        act_skip.triggered.connect(self.skip_image)
        view_menu.addAction(act_skip)

        # Help menu.
        help_menu = bar.addMenu("&Help")

        act_keys = QAction("Keyboard Shortcuts…", self)
        act_keys.setShortcut(QKeySequence(Qt.Key.Key_F1))
        act_keys.triggered.connect(self.show_shortcuts)
        help_menu.addAction(act_keys)

    # ── UI ───────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = AnnotationView(self.statusBar().showMessage)

        self.canvas.request_prev = lambda: self.step(-1)
        self.canvas.request_next = lambda: self.step(1)
        self.canvas.request_skip = self.skip_image
        self.canvas.request_copy = self.copy_annotation
        self.canvas.request_paste = self.paste_annotation

        layout.addWidget(self.canvas, stretch=1)

        side = QVBoxLayout()
        side.setContentsMargins(14, 14, 14, 14)
        side.setSpacing(14)

        # Tool settings.
        tool_box = QGroupBox("Tool")
        tool_form = QFormLayout(tool_box)
        tool_form.setVerticalSpacing(10)

        self.class_id_spin = QSpinBox()
        self.class_id_spin.setRange(0, 999)
        self.class_id_spin.setValue(0)
        tool_form.addRow("Class ID:", self.class_id_spin)

        self.class_name_edit = QLineEdit("ball")
        tool_form.addRow("Class label:", self.class_name_edit)

        side.addWidget(tool_box)

        # Ball settings / hint.
        ball_box = QGroupBox("Ball Mode Settings")
        ball_form = QFormLayout(ball_box)
        ball_form.setVerticalSpacing(10)

        hint = QLabel(
            "Multiple balls are supported.\n"
            "Draw one bounding box for each ball.\n\n"
            "Ctrl+C copies all boxes from the frame.\n"
            "Ctrl+V pastes them at the same coordinates.\n\n"
            "Edit: press Ctrl+E, then drag box handles."
        )

        hint.setWordWrap(True)
        ball_form.addRow("", hint)

        side.addWidget(ball_box)

        # Edit toggle.
        self.edit_btn = QPushButton("Edit / Resize Annotations")
        self.edit_btn.setObjectName("editBtn")
        self.edit_btn.setCheckable(True)
        self.edit_btn.setMinimumHeight(38)
        self.edit_btn.setToolTip(
            "When ON: click a box to select it, drag handles to resize, "
            "or drag inside the box to move it.\n"
            "When OFF: click-drag draws a new box. (Ctrl+E)"
        )
        self.edit_btn.toggled.connect(self._on_edit_toggled)
        side.addWidget(self.edit_btn)

        # Image list.
        side.addWidget(QLabel("Images"))

        self.file_list = QListWidget()
        self._status_boxes = []
        self.file_list.currentRowChanged.connect(self._on_row_changed)
        side.addWidget(self.file_list, stretch=1)

        panel = QWidget()
        panel.setObjectName("sidePanel")
        panel.setLayout(side)
        panel.setFixedWidth(320)

        layout.addWidget(panel)

        self.setCentralWidget(central)

        self.statusBar().showMessage(
            "File ▸ Open Image Dir to begin · Ball bbox-only mode"
        )

    def _build_shortcuts(self):
        QShortcut(
            QKeySequence("Ctrl+D"),
            self,
            activated=self.delete_current_image,
        )

    # ── Edit mode plumbing ───────────────────────────────────────────

    def _on_edit_toggled(self, checked):
        if self.canvas.edit_mode == checked:
            return

        self.canvas.set_edit_mode(checked)
        self._sync_edit_controls()

        self.statusBar().showMessage(
            "Edit mode ON — drag box handles to adjust"
            if checked
            else "Edit mode OFF — draw new bounding boxes"
        )

    def _sync_edit_controls(self):
        on = self.canvas.edit_mode

        for w in (self.edit_btn, self.act_edit_mode):
            w.blockSignals(True)
            w.setChecked(on)
            w.blockSignals(False)

    # ── Folder management ────────────────────────────────────────────

    def open_image_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")

        if not folder:
            return

        self.images_dir = folder

        parent = os.path.dirname(os.path.abspath(folder))
        self.labels_dir = os.path.join(parent, "labels")
        self.json_dir = os.path.join(parent, "json_labels")

        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

        self.image_files = sorted(
            f
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )

        if not self.image_files:
            self._populate_list()
            QMessageBox.warning(
                self,
                "Empty folder",
                "No images found in the selected folder.",
            )
            return

        self._populate_list()

        self.current_index = -1
        self.file_list.setCurrentRow(0)

        self.statusBar().showMessage(f"{len(self.image_files)} images loaded")

    def open_labels_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select labels folder")

        if not folder:
            return

        self.user_labels_dir = folder
        self._populate_list()

        if self.current_index >= 0:
            self.load_current()

        n = sum(
            1
            for f in self.image_files
            if self._find_label_for(os.path.splitext(f)[0])
        )

        self.statusBar().showMessage(
            f"Labels dir set — {n}/{len(self.image_files)} images annotated"
        )

    # ── Label search helpers ─────────────────────────────────────────

    def _label_dirs(self):
        dirs = []

        if self.user_labels_dir:
            dirs.append(self.user_labels_dir)

        if self.json_dir:
            dirs.append(self.json_dir)

        if self.labels_dir:
            dirs.append(self.labels_dir)

        return dirs

    def _find_label_for(self, base):
        """Return (kind, path) for a base name, or None. JSON wins."""
        dirs = self._label_dirs()

        for d in dirs:
            p = os.path.join(d, base + ".json")
            if os.path.isfile(p):
                return ("json", p)

        for d in dirs:
            p = os.path.join(d, base + ".txt")
            if os.path.isfile(p):
                return ("txt", p)

        return None

    def _class_id_from_txt(self, base):
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

    # ── Image list ───────────────────────────────────────────────────

    @staticmethod
    def _set_status_tick(label, annotated):
        label.setText("✔" if annotated else "")
        label.setToolTip("Annotated" if annotated else "Not annotated")

    def _make_row_widget(self, name, annotated):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 1, 4, 1)
        lay.setSpacing(6)

        tick = QLabel()
        tick.setFixedWidth(16)
        tick.setStyleSheet("color: #34c759; font-weight: bold;")
        tick.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_status_tick(tick, annotated)

        lbl = QLabel(name)
        lbl.setToolTip(name)

        btn = QPushButton("✖")
        btn.setObjectName("deleteBtn")
        btn.setFixedSize(30, 26)
        btn.setToolTip(f"Delete {name} and its label files (Ctrl+D)")
        btn.clicked.connect(
            lambda _=False, n=name: self._delete_image_by_name(n)
        )

        lay.addWidget(tick)
        lay.addWidget(lbl, 1)
        lay.addWidget(btn)

        return w, tick

    def _populate_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        self._status_boxes = []

        for name in self.image_files:
            annotated = (
                self._find_label_for(os.path.splitext(name)[0]) is not None
            )

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

        if self.autosave_check.isChecked():
            self.save_current(silent=True)
            return True

        resp = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save annotation before leaving this image?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )

        if resp == QMessageBox.StandardButton.Save:
            return self.save_current()

        return resp == QMessageBox.StandardButton.Discard

    def step(self, delta):
        if not self.image_files:
            return

        row = min(
            max(self.current_index + delta, 0),
            len(self.image_files) - 1,
        )

        self.file_list.setCurrentRow(row)

    def skip_image(self):
        if not self.image_files:
            return

        if self.autosave_check.isChecked():
            self.save_current(silent=True)

        self.canvas.dirty = False

        if self.current_index >= len(self.image_files) - 1:
            self.statusBar().showMessage("Already at the last image")
            self.canvas.setFocus()
            return

        self.step(1)

    # ── Copy / paste ─────────────────────────────────────────────────

        # ── Copy / paste ─────────────────────────────────────────────────

    def copy_annotation(self):
        """
        Ctrl+C:
        Copy ALL bounding boxes from the current frame.

        The clipboard stores:
        - all boxes
        - class ID
        - class label
        - source image size, so we can warn if pasting into a different size
        """
        if not self.canvas.boxes:
            self.statusBar().showMessage(
                "Nothing to copy — draw at least one ball box first, "
                "then press Ctrl+C"
            )
            return

        self._clipboard = {
            "boxes": [list(b) for b in self.canvas.boxes],
            "class_id": self.class_id_spin.value(),
            "class_label": self.class_name_edit.text().strip() or "ball",
            "source_width": self.canvas.img_w,
            "source_height": self.canvas.img_h,
        }

        count = len(self._clipboard["boxes"])

        self.statusBar().showMessage(
            f"Copied {count} box(es) from this frame — "
            "go to another frame and press Ctrl+V"
        )

    @staticmethod
    def _box_iou(box_a, box_b):
        """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih

        if inter <= 0.0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter

        return inter / union if union > 0.0 else 0.0

    def _merge_pasted_boxes(self, existing_boxes, pasted_boxes, threshold=0.5):
        """
        Merge pasted boxes into the boxes already on the current frame.

        Each pasted box only overrides the existing box it overlaps most,
        and only when that overlap (IoU) exceeds `threshold`. Existing
        boxes with no sufficiently overlapping pasted box are left
        untouched; pasted boxes with no sufficiently overlapping existing
        box are added as new boxes.
        """
        result = [list(b) for b in existing_boxes]
        matched_existing = set()

        overridden = 0
        added = 0

        for pb in pasted_boxes:
            best_idx, best_iou = None, 0.0

            for i, eb in enumerate(result):
                if i in matched_existing:
                    continue

                iou = self._box_iou(pb, eb)
                if iou > best_iou:
                    best_idx, best_iou = i, iou

            if best_idx is not None and best_iou > threshold:
                result[best_idx] = list(pb)
                matched_existing.add(best_idx)
                overridden += 1
            else:
                result.append(list(pb))
                added += 1

        return result, overridden, added

    def paste_annotation(self):
        """
        Ctrl+V:
        Paste copied boxes onto the current frame. A pasted box only
        overrides an existing box on this frame when they overlap more
        than 50%; existing boxes that don't overlap any pasted box are
        left untouched.
        """
        if not self._clipboard:
            self.statusBar().showMessage(
                "Clipboard empty — copy annotations first with Ctrl+C"
            )
            return

        if self.current_index < 0:
            self.statusBar().showMessage("Open an image before pasting")
            return

        snap = self._clipboard

        # New dict-style clipboard.
        if isinstance(snap, dict):
            boxes = [list(b) for b in snap.get("boxes", [])]

            if snap.get("class_id") is not None:
                self.class_id_spin.setValue(int(snap["class_id"]))

            if snap.get("class_label"):
                self.class_name_edit.setText(str(snap["class_label"]))

            source_w = snap.get("source_width")
            source_h = snap.get("source_height")

        # Backward compatibility for old list-style clipboard.
        else:
            boxes = [list(b) for b in snap]
            source_w = None
            source_h = None

        if not boxes:
            self.statusBar().showMessage("Clipboard has no boxes to paste")
            return

        # Merge at the same image-pixel coordinates: only override existing
        # boxes that the pasted boxes overlap more than 50%.
        existing_boxes = [list(b) for b in self.canvas.boxes]
        merged_boxes, overridden, added = self._merge_pasted_boxes(
            existing_boxes, boxes
        )
        kept = len(existing_boxes) - overridden

        self.canvas.boxes = merged_boxes
        self.canvas.history = self.canvas._canonical_history()
        self.canvas.redo_stack = []
        self.canvas.selected_box = None
        self.canvas._cleared_snapshot = None
        self.canvas.dirty = True

        self.canvas._rebuild_state()
        self.canvas.setFocus()

        msg = (
            f"Pasted {len(boxes)} box(es): {overridden} overrode overlapping "
            f"box(es), {added} added, {kept} existing box(es) kept untouched"
        )

        if (
            source_w is not None
            and source_h is not None
            and (source_w != self.canvas.img_w or source_h != self.canvas.img_h)
        ):
            msg += " — warning: copied from a different image size"

        self.statusBar().showMessage(msg)

    # ── Image deletion ───────────────────────────────────────────────

    def delete_current_image(self):
        if 0 <= self.current_index < len(self.image_files):
            self._delete_image_by_name(self.image_files[self.current_index])

    def _delete_image_by_name(self, name):
        if name not in self.image_files:
            return

        resp = QMessageBox.question(
            self,
            "Delete image",
            f"Permanently delete '{name}' and its label files?\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if resp != QMessageBox.StandardButton.Yes:
            return

        idx = self.image_files.index(name)
        deleting_current = idx == self.current_index
        base = os.path.splitext(name)[0]

        self._remove_files_for(name, base)
        self.image_files.pop(idx)

        self._populate_list()

        if not self.image_files:
            self.current_index = -1
            self.canvas.clear_annotations()
            self.setWindowTitle("Ball Annotation Tool")
            self.statusBar().showMessage(f"Deleted {name} — no images left")
            return

        if deleting_current:
            self.canvas.dirty = False
            self.current_index = -1
            self.file_list.setCurrentRow(
                min(idx, len(self.image_files) - 1)
            )
        else:
            if idx < self.current_index:
                self.current_index -= 1

            self.file_list.blockSignals(True)
            self.file_list.setCurrentRow(self.current_index)
            self.file_list.blockSignals(False)

        self.statusBar().showMessage(f"Deleted {name}")

    def _remove_label_files_for(self, base):
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
                    f"Could not delete {os.path.basename(p)}: {e}"
                )

    def _remove_files_for(self, name, base):
        if self.images_dir:
            p = os.path.join(self.images_dir, name)

            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError as e:
                self.statusBar().showMessage(
                    f"Could not delete {os.path.basename(p)}: {e}"
                )

        self._remove_label_files_for(base)

    # ── Current image loading ────────────────────────────────────────

    def current_image_path(self):
        if self.current_index < 0:
            return None

        return os.path.join(
            self.images_dir,
            self.image_files[self.current_index],
        )

    def load_current(self):
        path = self.current_image_path()

        if path and self.canvas.load_image(path):
            self.setWindowTitle(
                f"Annotating {self.image_files[self.current_index]} "
                f"[{self.current_index + 1}/{len(self.image_files)}]"
            )

            self._load_existing_annotation()

            if not self.canvas.history:
                self.statusBar().showMessage(
                    "Draw one bounding box for each ball. "
                    "Click-drag or two-click."
                )

            self.canvas.setFocus()

    def _load_existing_annotation(self):
        if not self.image_files or self.current_index < 0:
            return

        base = os.path.splitext(self.image_files[self.current_index])[0]
        found = self._find_label_for(base)

        if found is None:
            return

        kind, path = found

        if kind == "json":
            self._apply_json_file(path)
        else:
            self._apply_txt_file(path)

    # ── File parsers ─────────────────────────────────────────────────

    def _apply_json_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Load error", f"Cannot read JSON:\n{e}")
            return

        rects = []
        rect_label = None

        for shape in data.get("shapes", []):
            if shape.get("shape_type", "") != "rectangle":
                continue

            raw = shape.get("points", [])
            if len(raw) < 2:
                continue

            xs = [p[0] for p in raw]
            ys = [p[1] for p in raw]

            rects.append([min(xs), min(ys), max(xs), max(ys)])

            if rect_label is None:
                rect_label = shape.get("label")

        if not ALLOW_MULTIPLE_BOXES and rects:
            rects = rects[:1]

        base = os.path.splitext(os.path.basename(path))[0]

        if rect_label:
            self.class_name_edit.setText(rect_label)

        cid = self._class_id_from_txt(base)
        if cid is not None:
            self.class_id_spin.setValue(cid)

        if not rects:
            self.statusBar().showMessage(
                f"{os.path.basename(path)} — no rectangles found"
            )
            return

        self.canvas.set_history([("box", r) for r in rects])
        self.statusBar().showMessage(f"Loaded JSON: {os.path.basename(path)}")

    def _apply_txt_file(self, path):
        W, H = self.canvas.img_w, self.canvas.img_h

        if W == 0 or H == 0:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except OSError as e:
            QMessageBox.warning(self, "Load error", f"Cannot read TXT:\n{e}")
            return

        rects = []
        first_cls = None

        for line in lines:
            parts = line.split()

            if len(parts) < 5:
                continue

            if first_cls is None:
                try:
                    first_cls = int(float(parts[0]))
                except ValueError:
                    first_cls = 0

            xc = float(parts[1]) * W
            yc = float(parts[2]) * H
            bw = float(parts[3]) * W
            bh = float(parts[4]) * H

            rects.append(
                [
                    xc - bw / 2.0,
                    yc - bh / 2.0,
                    xc + bw / 2.0,
                    yc + bh / 2.0,
                ]
            )

        if not ALLOW_MULTIPLE_BOXES and rects:
            rects = rects[:1]

        if first_cls is not None:
            self.class_id_spin.setValue(first_cls)

        if not rects:
            self.statusBar().showMessage(
                f"{os.path.basename(path)} — no annotations found"
            )
            return

        self.canvas.set_history([("box", r) for r in rects])
        self.statusBar().showMessage(f"Loaded TXT: {os.path.basename(path)}")

    # ── Saving ───────────────────────────────────────────────────────

    def _collect_annotation(self):
        boxes = [list(b) for b in self.canvas.boxes]

        if not boxes:
            raise ValueError("Draw a bounding box before saving.")

        return boxes

    def save_current(self, silent=False):
        if self.current_index < 0:
            return False

        try:
            boxes = self._collect_annotation()
        except ValueError as e:
            base = os.path.splitext(self.image_files[self.current_index])[0]

            if self.canvas.dirty and self._find_label_for(base) is not None:
                self._remove_label_files_for(base)
                self.canvas.dirty = False
                self._mark_current_unannotated()
                self.statusBar().showMessage(
                    f"Annotation deleted — removed {base} label files"
                )
                return True

            if not silent:
                QMessageBox.warning(self, "Cannot save", str(e))
            else:
                self.statusBar().showMessage(f"Not saved — {e}")

            return False

        image_name = self.image_files[self.current_index]
        base = os.path.splitext(image_name)[0]

        txt_path = os.path.join(self.labels_dir, base + ".txt")
        json_path = os.path.join(self.json_dir, base + ".json")

        self._write_yolo_txt(txt_path, boxes)
        self._write_labelme_json(json_path, image_name, boxes)

        self.canvas.dirty = False
        self._mark_current_annotated()

        self.statusBar().showMessage(f"Saved {base}.txt + {base}.json")
        return True

    def _write_yolo_txt(self, path, boxes):
        """
        YOLO format for detection:
            class_id xc yc w h

        All coordinates are normalised to [0, 1].
        """
        W = self.canvas.img_w
        H = self.canvas.img_h
        cls = self.class_id_spin.value()

        def norm(v, dim):
            return min(max(v / dim, 0.0), 1.0)

        lines = []

        for box in boxes:
            x1, y1, x2, y2 = box

            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            fields = [
                str(cls),
                f"{norm(xc, W):.6f}",
                f"{norm(yc, H):.6f}",
                f"{norm(w, W):.6f}",
                f"{norm(h, H):.6f}",
            ]

            lines.append(" ".join(fields))

        with open(path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
            else:
                f.write("")

    def _write_labelme_json(self, path, image_name, boxes):
        """
        LabelMe JSON with rectangle shapes.
        Coordinates are absolute pixel values.
        """
        label = self.class_name_edit.text().strip() or "ball"

        shapes = []

        for box in boxes:
            shapes.append(
                {
                    "label": label,
                    "points": [
                        [float(box[0]), float(box[1])],
                        [float(box[2]), float(box[3])],
                    ],
                    "group_id": 0,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None,
                }
            )

        img_abs = os.path.abspath(self.current_image_path()).replace("\\", "/")

        doc = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_abs,
            "imageData": None,
            "imageHeight": int(self.canvas.img_h),
            "imageWidth": int(self.canvas.img_w),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    # ── Help ─────────────────────────────────────────────────────────

    def show_shortcuts(self):
        text = "\n".join(
            [
                "A — Previous image",
                "D — Next image",
                "Space — Skip image",
                "Backspace — Delete selected box",
                "Delete — Delete all boxes",
                "Ctrl+C — Copy annotation",
                "Ctrl+V — Paste annotation",
                "Ctrl+E — Toggle Edit / Resize mode",
                "Ctrl+S — Save",
                "Ctrl+Z — Undo",
                "Ctrl+Y — Redo",
                "Ctrl+F — Fit image to screen",
                "B — Zoom to bounding box",
                "Ctrl+O — Open image directory",
                "Ctrl+D — Delete current image file and labels",
                "Esc — Cancel current operation",
                "Ctrl+Scroll — Zoom",
                "Shift+Drag — Pan",
                "Shift+Scroll — Horizontal pan",
                "Scroll — Vertical pan",
                "Right-click box — Delete menu",
            ]
        )

        QMessageBox.information(self, "Keyboard Shortcuts", text)

    # ── Close ────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()


DARK_STYLE = """
* {
    font-family: 'Segoe UI', '-apple-system', 'Helvetica Neue', Arial;
    font-size: 13px;
    color: #f2f2f7;
}

QMainWindow, QDialog {
    background: #1c1c1e;
}

QWidget#sidePanel {
    background: #1c1c1e;
    border-left: 1px solid #3a3a3c;
}

QLabel {
    background: transparent;
}

QGroupBox {
    background: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 14px;
    margin-top: 14px;
    padding: 14px 14px 12px 14px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: #98989f;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QSpinBox, QLineEdit {
    background: #3a3a3c;
    border: 1px solid #3a3a3c;
    border-radius: 9px;
    padding: 6px 10px;
    min-height: 20px;
    selection-background-color: #0a84ff;
    selection-color: #ffffff;
}

QSpinBox:focus, QLineEdit:focus {
    border: 1px solid #0a84ff;
}

QPushButton {
    background: #3a3a3c;
    border: 1px solid #3a3a3c;
    border-radius: 10px;
    padding: 8px 14px;
    font-weight: 600;
}

QPushButton:hover {
    background: #48484a;
}

QPushButton:pressed {
    background: #0a84ff;
    color: #ffffff;
}

QPushButton:checked {
    background: #0a84ff;
    border: 1px solid #0a84ff;
    color: #ffffff;
}

QPushButton#deleteBtn {
    background: transparent;
    border: none;
    border-radius: 8px;
    color: #ff3b30;
    font-size: 14px;
    padding: 0;
}

QPushButton#deleteBtn:hover {
    background: rgba(255, 59, 48, 0.18);
}

QPushButton#deleteBtn:pressed {
    background: rgba(255, 59, 48, 0.34);
}

QListWidget {
    background: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 12px;
    padding: 4px;
    outline: none;
}

QListWidget::item {
    border-radius: 8px;
    padding: 2px;
    margin: 1px 2px;
}

QListWidget::item:selected {
    background: #0a84ff;
    color: #ffffff;
}

QListWidget::item:hover {
    background: #48484a;
}

QMenuBar {
    background: #161618;
    border-bottom: 1px solid #3a3a3c;
    padding: 3px 6px;
}

QMenuBar::item {
    background: transparent;
    padding: 6px 12px;
    border-radius: 8px;
}

QMenuBar::item:selected {
    background: #48484a;
}

QMenu {
    background: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 12px;
    padding: 6px;
}

QMenu::item {
    padding: 7px 26px 7px 18px;
    border-radius: 8px;
}

QMenu::item:selected {
    background: #0a84ff;
    color: #ffffff;
}

QMenu::item:disabled {
    color: #98989f;
}

QMenu::separator {
    height: 1px;
    background: #3a3a3c;
    margin: 5px 8px;
}

QStatusBar {
    background: #161618;
    border-top: 1px solid #3a3a3c;
    color: #98989f;
    padding: 3px 8px;
}

QStatusBar::item {
    border: none;
}

QScrollBar:vertical {
    background: transparent;
    width: 11px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: #3a3a3c;
    border-radius: 5px;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover {
    background: #98989f;
}

QScrollBar:horizontal {
    background: transparent;
    height: 11px;
    margin: 2px;
}

QScrollBar::handle:horizontal {
    background: #3a3a3c;
    border-radius: 5px;
    min-width: 28px;
}

QScrollBar::handle:horizontal:hover {
    background: #98989f;
}

QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}

QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
}

QToolTip {
    background: #2c2c2e;
    color: #f2f2f7;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 6px 9px;
}
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()