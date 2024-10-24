import sys
from PyQt6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt


class DataGrid(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the data
        data = [
            ["Name", "Age", "City"],
            ["Alice", 30, "New York"],
            ["Bob", 25, "Los Angeles"],
            ["Charlie", 35, "Chicago"],
        ]

        # Create the table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(data))
        self.tableWidget.setColumnCount(len(data[0]))

        # Populate the table
        for i, row in enumerate(data):
            for j, item in enumerate(row):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(item)))

        # Apply dark theme
        self.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF;")
        self.tableWidget.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E; 
                color: #FFFFFF;
            }
            QTableWidget::item {
                padding: 10px;
            }
        """)

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)

        self.setWindowTitle('Data Grid Example')
        self.resize(400, 300)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataGrid()
    window.show()
    sys.exit(app.exec())