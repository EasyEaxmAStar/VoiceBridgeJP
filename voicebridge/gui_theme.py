"""Explicit settings colors: main-window text styles must not inherit a dark base."""
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

def settings_theme(widget):
    dark=QApplication.palette().color(QPalette.Window).lightness()<128
    bg,base,fg,border,muted=('#202a36','#15202c','#edf3fb','#71849a','#b4c0ce') if dark else ('#f4f6fa','#ffffff','#203549','#75879c','#526578')
    widget.setStyleSheet(f'''
        QWidget {{background:{bg};color:{fg};}}
        QLineEdit,QPlainTextEdit,QTableWidget,QComboBox,QSpinBox,QDoubleSpinBox {{background:{base};color:{fg};border:1px solid {border};padding:5px;selection-background-color:#21649c;selection-color:white;}}
        QAbstractItemView,QHeaderView::section,QComboBox QAbstractItemView {{background:{base};color:{fg};}}
        QPushButton {{background:{base};color:{fg};border:1px solid {border};padding:9px;border-radius:4px;}}
        QPushButton:hover,QTabBar::tab:selected {{background:#246398;color:white;}}
        QPushButton:focus,QLineEdit:focus,QComboBox:focus {{border:2px solid #3889c7;}}
        QWidget:disabled {{color:{muted};}}
        QTabBar::tab {{background:{base};color:{fg};padding:8px;border:1px solid {border};}}
        QToolTip {{background:{base};color:{fg};border:1px solid {border};}}
        QCheckBox::indicator {{width:16px;height:16px;border:1px solid {border};background:{base};}}
        QCheckBox::indicator:checked {{background:#3889c7;}}
    ''')
    widget.setProperty('explicit_settings_theme','dark' if dark else 'light')
