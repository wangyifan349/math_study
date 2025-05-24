import sys, json, threading, time
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, QEvent
from PyQt5.QtGui import QPen, QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsEllipseItem,
    QGraphicsTextItem,
    QGraphicsLineItem,
    QGraphicsItem,
    QInputDialog,
    QColorDialog,
    QFileDialog,
    QMessageBox,
    QAction,
    QToolBar,
    QStyleFactory,
    QWidget,
    QVBoxLayout,
)

# 样式表
STYLE_SHEET = """
QMainWindow {
    background-color: #F0F0F0;
}

QToolBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #91C7F4, stop:1 #478EE0);
    spacing: 6px;
}

QToolButton {
    background-color: transparent;
    color: white;
    font: bold 12px;
    border: none;
    padding: 5px;
}

QToolButton:hover {
    background-color: rgba(255, 255, 255, 30);
}

QMenuBar {
    background: #478EE0;
    color: white;
}
QMenuBar::item {
    background: transparent;
}
QMenuBar::item:selected {
    background: #91C7F4;
}

QMenu {
    background-color: #F7F7F7;
    border: 1px solid #dcdcdc;
}

QMenu::item:selected {
    background-color: #91C7F4;
    color: white;
}

QMessageBox {
    background-color: #F0F0F0;
}
"""

class MindMapNode(QGraphicsEllipseItem):
    def __init__(self, x, y, text, node_id, shape="circle", color="#ADD8E6", radius=30, parent=None):
        self.radius = radius
        self.node_id = node_id
        self.text_str = text
        self.shape = shape
        self.color = color
        if shape == "circle":
            rect = QRectF(x - radius, y - radius, 2 * radius, 2 * radius)
        elif shape == "ellipse":
            rect = QRectF(x - radius * 1.3, y - radius, 2 * radius * 1.3, 2 * radius)
        else:
            rect = QRectF(x - radius, y - radius, 2 * radius, 2 * radius)
        super().__init__(rect, parent)
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#333333"), 2))
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.text_item = QGraphicsTextItem(text, self)
        self.text_item.setDefaultTextColor(QColor("#222222"))
        font = QFont("Arial", 12, QFont.Bold)
        self.text_item.setFont(font)
        self.adjust_text_position()

    def adjust_text_position(self):
        bounding = self.text_item.boundingRect()
        rect = self.rect()
        x = rect.x() + (rect.width() - bounding.width()) / 2
        y = rect.y() + (rect.height() - bounding.height()) / 2
        self.text_item.setPos(x, y)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self.scene().views()[0].parent().edges:
                if edge.node_from == self or edge.node_to == self:
                    edge.update_position()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        new_text, ok = QInputDialog.getText(None, "修改节点标签", "请输入新的标签：", text=self.text_str)
        if ok and new_text:
            self.text_str = new_text
            self.text_item.setPlainText(new_text)
            self.adjust_text_position()
        new_color = QColorDialog.getColor(QColor(self.color), None, "选择节点颜色")
        if new_color.isValid():
            self.color = new_color.name()
            self.setBrush(QBrush(new_color))
        shape_choice, ok = QInputDialog.getText(None, "选择节点形状", "请输入节点形状（circle/ellipse）：", text=self.shape)
        if ok and shape_choice in ["circle", "ellipse"] and shape_choice != self.shape:
            self.shape = shape_choice
            center = self.sceneBoundingRect().center()
            if self.shape == "circle":
                new_rect = QRectF(center.x() - self.radius, center.y() - self.radius,
                                  2 * self.radius, 2 * self.radius)
            else:
                new_rect = QRectF(center.x() - self.radius * 1.3, center.y() - self.radius,
                                  2 * self.radius * 1.3, 2 * self.radius)
            self.setRect(new_rect)
            self.adjust_text_position()
        super().mouseDoubleClickEvent(event)

    def to_dict(self):
        center = self.sceneBoundingRect().center()
        return {
            "node_id": self.node_id,
            "x": center.x(),
            "y": center.y(),
            "text": self.text_str,
            "shape": self.shape,
            "color": self.color
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["x"], data["y"], data["text"], data["node_id"],
                   shape=data.get("shape", "circle"), color=data.get("color", "#ADD8E6"))

class MindMapEdge(QGraphicsLineItem):
    def __init__(self, node_from: MindMapNode, node_to: MindMapNode, annotation="", color="black", parent=None):
        super().__init__(parent)
        self.node_from = node_from
        self.node_to = node_to
        self.annotation = annotation
        self.color = color
        self.setPen(QPen(QColor(color), 2))
        self.setZValue(-1)
        self.text_item = QGraphicsTextItem(annotation)
        self.text_item.setDefaultTextColor(QColor("red"))
        font = QFont("Arial", 10)
        self.text_item.setFont(font)
        self.update_position()
        self.setFlags(QGraphicsItem.ItemIsSelectable)

    def update_position(self):
        p1 = self.node_from.sceneBoundingRect().center()
        p2 = self.node_to.sceneBoundingRect().center()
        self.setLine(p1.x(), p1.y(), p2.x(), p2.y())
        self.update_label_position()

    def update_label_position(self):
        line = self.line()
        mid_x = (line.x1() + line.x2()) / 2
        mid_y = (line.y1() + line.y2()) / 2
        self.text_item.setPos(mid_x - self.text_item.boundingRect().width()/2, mid_y - 20)

    def mouseDoubleClickEvent(self, event):
        new_text, ok = QInputDialog.getText(None, "修改连线注释", "请输入新的注释：", text=self.annotation)
        if ok:
            self.annotation = new_text
            self.text_item.setPlainText(new_text)
            self.update_label_position()
        new_color = QColorDialog.getColor(QColor(self.color), None, "选择连线颜色")
        if new_color.isValid():
            self.color = new_color.name()
            self.setPen(QPen(new_color, 2))
        super().mouseDoubleClickEvent(event)

    def to_dict(self):
        return {
            "node_from": self.node_from.node_id,
            "node_to": self.node_to.node_id,
            "annotation": self.annotation,
            "color": self.color
        }

    @classmethod
    def from_dict(cls, nodes_dict, data):
        node_from = nodes_dict.get(data["node_from"])
        node_to = nodes_dict.get(data["node_to"])
        if node_from and node_to:
            return cls(node_from, node_to, annotation=data.get("annotation", ""), color=data.get("color", "black"))
        return None

class EdgeUpdaterThread(threading.Thread):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.running = True
        self.interval = 0.05

    def run(self):
        while self.running:
            time.sleep(self.interval)
            # 更新场景内所有边的位置
            self.window.update_edges()

    def stop(self):
        self.running = False

class MindMapMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("思维导图工具 - PyQt5 美观版")
        self.resize(1000, 800)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(self.view.renderHints() | Qt.Antialiasing)
        centralWidget = QWidget()
        layout = QVBoxLayout(centralWidget)
        layout.addWidget(self.view)
        self.setCentralWidget(centralWidget)

        self.statusBar().showMessage("就绪")

        self.nodes = {}
        self.edges = []
        self.node_counter = 1

        self.mode = None  # None, "add", "connect"
        self.connect_start_node = None

        self.create_actions()
        self.create_menus()
        self.create_toolbar()

        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        self.edge_updater = EdgeUpdaterThread(self)
        self.edge_updater.start()

    def closeEvent(self, event):
        self.edge_updater.stop()
        event.accept()

    def create_actions(self):
        self.new_map_act = QAction("新建", self)
        self.new_map_act.triggered.connect(self.new_map)
        self.open_map_act = QAction("导入", self)
        self.open_map_act.triggered.connect(self.import_map)
        self.save_map_act = QAction("导出", self)
        self.save_map_act.triggered.connect(self.export_map)
        self.exit_act = QAction("退出", self)
        self.exit_act.triggered.connect(self.close)
        self.add_node_act = QAction("添加节点", self)
        self.add_node_act.triggered.connect(self.start_add_node_mode)

    def create_menus(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("文件")
        fileMenu.addAction(self.new_map_act)
        fileMenu.addAction(self.open_map_act)
        fileMenu.addAction(self.save_map_act)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exit_act)

    def create_toolbar(self):
        toolbar = QToolBar("工具", self)
        self.addToolBar(toolbar)
        toolbar.addAction(self.add_node_act)

    def new_map(self):
        reply = QMessageBox.question(self, '新建', '新建会清空当前画布，是否继续？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.scene.clear()
            self.nodes.clear()
            self.edges.clear()
            self.node_counter = 1

    def start_add_node_mode(self):
        self.mode = "add"
        self.connect_start_node = None
        self.statusBar().showMessage("模式：添加节点，请左键点击场景添加。")

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and source == self.view.viewport():
            pos = event.pos()
            scene_pos = self.view.mapToScene(pos)
            if event.button() == Qt.LeftButton:
                if self.mode == "add":
                    self.add_node_at(scene_pos)
                    self.mode = None
                    self.statusBar().showMessage("就绪")
                else:
                    item = self.scene.itemAt(scene_pos, self.view.transform())
                    if isinstance(item, MindMapNode):
                        if self.mode == "connect":
                            self.finish_connection(item)
            elif event.button() == Qt.RightButton:
                item = self.scene.itemAt(scene_pos, self.view.transform())
                if isinstance(item, MindMapNode):
                    if not self.connect_start_node:
                        self.connect_start_node = item
                        self.mode = "connect"
                        self.statusBar().showMessage(f"连线起点：节点{item.node_id}。请右键点击目标节点。")
                    else:
                        if item != self.connect_start_node:
                            self.finish_connection(item)
            return False
        return super().eventFilter(source, event)

    def add_node_at(self, pos: QPointF):
        default_text = f"节点{self.node_counter}"
        text, ok = QInputDialog.getText(self, "节点标签", "请输入节点标签：", text=default_text)
        if not ok or not text:
            text = default_text
        shape, ok = QInputDialog.getText(self, "节点形状", "请输入节点形状（circle/ellipse）：", text="circle")
        if not ok or shape not in ["circle", "ellipse"]:
            shape = "circle"
        color = QColorDialog.getColor(QColor("#ADD8E6"), self, "选择节点颜色")
        if not color.isValid():
            color = QColor("#ADD8E6")
        node = MindMapNode(pos.x(), pos.y(), text, self.node_counter, shape=shape, color=color.name())
        self.scene.addItem(node)
        self.nodes[self.node_counter] = node
        self.node_counter += 1

    def finish_connection(self, target_node):
        if self.connect_start_node and target_node != self.connect_start_node:
            annotation, ok = QInputDialog.getText(self, "连线注释", "请输入连线注释：")
            if not ok:
                annotation = ""
            line_color = QColorDialog.getColor(QColor("black"), self, "选择连线颜色")
            if not line_color.isValid():
                line_color = QColor("black")
            edge = MindMapEdge(self.connect_start_node, target_node, annotation=annotation, color=line_color.name())
            self.scene.addItem(edge)
            self.scene.addItem(edge.text_item)
            self.edges.append(edge)
            self.statusBar().showMessage(f"连线：节点{self.connect_start_node.node_id} -> 节点{target_node.node_id}")
        self.connect_start_node = None
        self.mode = None

    def update_edges(self):
        for edge in self.edges:
            edge.update_position()

    def export_map(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "导出思维导图", filter="JSON files (*.json)")
        if file_path:
            data = {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "edges": [edge.to_dict() for edge in self.edges],
                "node_counter": self.node_counter
            }
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                QMessageBox.information(self, "导出成功", f"思维导图已保存到 {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))

    def import_map(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "导入思维导图", filter="JSON files (*.json)")
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.scene.clear()
                self.nodes.clear()
                self.edges.clear()
                for node_data in data.get("nodes", []):
                    node = MindMapNode.from_dict(node_data)
                    self.scene.addItem(node)
                    self.nodes[node.node_id] = node
                self.node_counter = data.get("node_counter", len(self.nodes) + 1)
                for edge_data in data.get("edges", []):
                    edge = MindMapEdge.from_dict(self.nodes, edge_data)
                    if edge:
                        self.scene.addItem(edge)
                        self.scene.addItem(edge.text_item)
                        self.edges.append(edge)
                QMessageBox.information(self, "导入成功", f"思维导图已从 {file_path} 导入。")
            except Exception as e:
                QMessageBox.critical(self, "导入失败", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setStyleSheet(STYLE_SHEET)
    window = MindMapMainWindow()
    window.show()
    sys.exit(app.exec_())
