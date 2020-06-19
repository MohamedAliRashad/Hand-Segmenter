from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    qApp,
)
from pathlib import Path
from utils.transforms import cv2qimg
from utils.view import gray_masking
import cv2


class HandSegmenter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(False)

        # Variables Initilization
        self.title = "Hand Segmenter"
        self.directory = None
        self.index = None
        self.image = None
        self.masks = None
        self.save_dir = None
        self.offset = 0

        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle(self.title)
        self.resize(800, 600)

    def open(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Choose RGB Image",
            "",
            "Images (*.jpeg *.jpg *.bmp *.gif)",
            options=options,
        )

        if fileName:
            self.directory = sorted(
                list(Path(fileName).parent.glob("*" + Path(fileName).suffix))
            )
            self.index = self.directory.index(Path(fileName))

            self.open_image(fileName)

    def open_image(self, file_name):

        self.image = QImage(file_name)
        if self.image.isNull():
            QMessageBox.information(
                self, "Hand Segmenter", "Cannot load %s." % file_name
            )
            return

        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))
        self.scaleFactor = 1.0

        # Menu actions to be enabled after opening an image
        self.scrollArea.setVisible(True)
        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.movRightAct.setEnabled(True)
        self.movLeftAct.setEnabled(True)
        self.autoMaskAct.setEnabled(True)
        self.addMaskAct.setEnabled(True)
        self.saveMaskAct.setEnabled(True)

        self.setWindowTitle(self.title + " - {}".format(Path(file_name).name))

        self.updateActions()

        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def move_right(self):
        self.index = (self.index + 1) % len(self.directory)
        if self.masks:
            current_img = cv2.imread(str(self.directory[self.index]))
            mask = cv2.imread(
                str(self.masks[self.index - self.offset]), cv2.IMREAD_GRAYSCALE
            )
            output = gray_masking(current_img, mask)
            self.imageLabel.setPixmap(cv2qimg(output))
        else:
            self.open_image(str(self.directory[self.index]))

    def move_left(self):
        self.index = (self.index - 1) % len(self.directory)
        if self.masks:
            current_img = cv2.imread(str(self.directory[self.index]))
            mask = cv2.imread(
                str(self.masks[self.index - self.offset]), cv2.IMREAD_GRAYSCALE
            )
            output = gray_masking(current_img, mask)
            self.imageLabel.setPixmap(cv2qimg(output))
        else:
            self.open_image(str(self.directory[self.index]))

    def add_mask(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "Images (*.png)", options=options
        )

        if fileName:
            current_img = cv2.imread(str(self.directory[self.index]))
            mask = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

            output = gray_masking(current_img, mask)
            self.imageLabel.setPixmap(cv2qimg(output))

            if Path(fileName).parent != self.directory[0].parent:
                msg = QMessageBox()
                msg.setWindowTitle("Link Directories")
                msg.setText(
                    "We have noticed different directories for images and masks, Do you want to link them for you ?"
                )
                msg.setIcon(QMessageBox.Question)
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.Yes)
                if msg.exec_() == QMessageBox.Yes:
                    self.masks = sorted(list(Path(fileName).parent.glob("*" + Path(fileName).suffix)))
                    if len(self.masks) != len(self.directory):
                        msg = QMessageBox()
                        msg.setWindowTitle("Can't assosiate images to masks")
                        msg.setText(
                            "Number of masks doesn't equal the number of images, Do you want to continue ?"
                        )
                        msg.setIcon(QMessageBox.Question)
                        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Abort)
                        msg.setDefaultButton(QMessageBox.Abort)
                        if msg.exec_() == QMessageBox.Yes:
                            self.offset = len(self.directory) - len(self.masks)
                        else:
                            self.masks = None

    def auto_mask(self):
        try:
            heatmap = maskIt(self.seg, self.image)
            final_image = np.where(heatmap == 0, self.image, [0, 0, 255])
            self.show_image(final_image, self.name)
        except AttributeError:
            print("HandSegNet Not Configured yet")

    def save_mask(self):

        mask_name = self.directory[self.index].stem + ".png"
        if self.save_dir is None:
            msg = QMessageBox()
            msg.setWindowTitle("Save Directory ?")
            msg.setText(
                "Do you want to save your masks in the same directory as your images ?"
            )
            msg.setIcon(QMessageBox.Question)
            msg.setStandardButtons(
                QMessageBox.Yes | QMessageBox.Open | QMessageBox.Cancel
            )
            msg.setDefaultButton(QMessageBox.Yes)
            x = msg.exec_()  # this will show our messagebox
            if x == QMessageBox.Yes:
                self.save_dir = self.directory[0].parent
            elif x == QMessageBox.Open:
                self.save_dir = Path(
                    QFileDialog.getExistingDirectory(self, "Select Directory")
                )

        print("Mask will be saved in: {}".format(str(self.save_dir / mask_name)))

    def color_mask(self):
        # define range of white color in HSV
        # lower_white = np.array([0,0,255])
        # upper_white = np.array([255,255,255])

        # Create the mask
        # mask = cv2.inRange(image_mark, lower_white, upper_white)
        pass

    def about(self):
        QMessageBox.about(
            self,
            "About Hand Segmenter",
            "<p>The <b>Hand Segmenter</b> enables fast segmentation "
            "for the human hands with datasets desing in mind.</p>"
            "<p>We were able to handle different sceniores for segmentation "
            "with easy usage as our primer goal, and feature riching "
            "as our second. </p>"
        )

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.addMaskAct = QAction(
            "&Add...", self, shortcut="Ctrl+A", enabled=False, triggered=self.add_mask
        )
        self.autoMaskAct = QAction(
            "&Generate...",
            self,
            shortcut="Space",
            enabled=False,
            triggered=self.auto_mask,
        )
        self.saveMaskAct = QAction(
            "&Save...", self, shortcut="Ctrl+S", enabled=False, triggered=self.save_mask
        )
        self.printAct = QAction(
            "&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_
        )
        self.exitAct = QAction("&Exit", self, shortcut="ESC", triggered=self.close)
        self.movRightAct = QAction(
            "Move &Right",
            self,
            shortcut="Right",
            enabled=False,
            triggered=self.move_right,
        )
        self.movLeftAct = QAction(
            "Move &Left", self, shortcut="Left", enabled=False, triggered=self.move_left
        )
        self.zoomInAct = QAction(
            "Zoom &In (25%)",
            self,
            shortcut="Ctrl++",
            enabled=False,
            triggered=self.zoomIn,
        )
        self.zoomOutAct = QAction(
            "Zoom &Out (25%)",
            self,
            shortcut="Ctrl+-",
            enabled=False,
            triggered=self.zoomOut,
        )
        self.normalSizeAct = QAction(
            "&Normal Size",
            self,
            shortcut="Ctrl+N",
            enabled=False,
            triggered=self.normalSize,
        )
        self.fitToWindowAct = QAction(
            "&Fit to Window",
            self,
            enabled=False,
            checkable=True,
            shortcut="Ctrl+F",
            triggered=self.fitToWindow,
        )
        self.aboutAct = QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.maskMenu = QMenu("&Mask", self)
        self.maskMenu.addAction(self.addMaskAct)
        self.maskMenu.addAction(self.autoMaskAct)
        self.maskMenu.addSeparator()
        self.maskMenu.addAction(self.saveMaskAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.movRightAct)
        self.viewMenu.addAction(self.movLeftAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.maskMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(
            int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2))
        )


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    handSegmenter = HandSegmenter()
    handSegmenter.show()
    sys.exit(app.exec_())
