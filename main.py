import sys #system library

# Import another widget
from src.authors_window import Ui_Authors_Window
from src.helpCenter import Ui_HelpCenter
from src.references import Ui_References

# Import PyQt5 Widgets
from PyQt5.QtWidgets import (
    QApplication as APP,
    QMainWindow as MAIN,
    QLabel as LABEL,
    QPushButton as BUTTON,
    QFileDialog as DIALOG,
    QAction as ACTION,
    QMessageBox as MESSAGE,
    QSlider as SLIDER
)

from PyQt5.uic import loadUi #for load the ui file
from PyQt5.QtGui import QPixmap, QImage #for image accessibillity

#Import Python Image Processing Library
import cv2 #OpenCV
import matplotlib.pyplot as plt #Matplolib
import numpy as np #NumPy

THE_DEFAULT_UI = 'gui_repo/process_window.ui' #The folder that containing the UI
THE_DEFAULT_IMG = 'component' # default image directory

class Image_Processing_Tools(MAIN):
    def __init__(self):
        super(Image_Processing_Tools, self).__init__()
        loadUi(THE_DEFAULT_UI, self)

        #Define all widgets
        ###### LABEL
        self.image_before = self.findChild(LABEL, "imported_image")
        self.image_after = self.findChild(LABEL, "process_image")

        ############ MENU #################
        ######### File Action ###########
        self.newFile = self.findChild(ACTION, "actionNew")
        self.saveFile = self.findChild(ACTION, "actionSave")
        self.exitWindow = self.findChild(ACTION, "actionExit")

        #################### Geometry ##################33
        self.resizeImg = self.findChild(ACTION, "actionResize")
        self.zoomInImg = self.findChild(ACTION, "actionZoomIn")
        self.zoomOutImg = self.findChild(ACTION, "actionZoomOut")
        self.rotImg45clock = self.findChild(ACTION, "actionRotation_46")
        self.rotImg45 = self.findChild(ACTION, "actionRotation_47")
        self.rotImg90clock = self.findChild(ACTION, "actionRotation_90")
        self.rotImg90 = self.findChild(ACTION, "actionRotation_91")

        ################# Enhancement ####################
        self.powerLaw = self.findChild(ACTION, "actionPower_Law_Transform")
        self.bitPlane = self.findChild(ACTION, "actionBit_Plane_8")

        ################  Help and About #####################
        self.helpCent = self.findChild(ACTION, "actionEditor_Help")
        self.reference = self.findChild(ACTION, "actionReferences")
        self.author_app = self.findChild(ACTION, "actionAuthor")

        #### Button and Slider
        self.gsb = self.findChild(BUTTON, "grayscale_button")
        self.ngv = self.findChild(BUTTON, "negative_button")
        self.gma = self.findChild(BUTTON, "gamma_button")
        self.edges_detect = self.findChild(BUTTON, "edges_button")
        self.sharpen = self.findChild(BUTTON, "sharpen_button")
        self.reduce_noise = self.findChild(BUTTON, "reduce_noise_button")
        self.hist = self.findChild(BUTTON, "histogram_button")
        self.medthre = self.findChild(BUTTON, "median_threshold_button")

        self.resetImg = self.findChild(BUTTON, "reset_button")
        self.saveImg = self.findChild(BUTTON, "save_button")

        self.blurSlid = self.findChild(SLIDER, "blurry_slider")
        self.brightSlid = self.findChild(SLIDER, "bright_slider")

        #File Menu Action
        self.newFile.triggered.connect(self.goNew)
        self.saveFile.triggered.connect(self.goSave)
        self.exitWindow.triggered.connect(self.goExit)

        #Geometry Menu Action
        self.resizeImg.triggered.connect(self.goResize)
        self.zoomInImg.triggered.connect(self.goZoomIn)
        self.zoomOutImg.triggered.connect(self.goZoomOut)
        self.rotImg45clock.triggered.connect(self.goRotation45clock)
        self.rotImg45.triggered.connect(self.goRotation45)
        self.rotImg90clock.triggered.connect(self.goRotation90clock)
        self.rotImg90.triggered.connect(self.goRotation90)

        #Enhancement
        self.powerLaw.triggered.connect(self.goPowerLaw)
        self.bitPlane.triggered.connect(self.goBitPlane)

        #Help Menu Action
        self.helpCent.triggered.connect(self.goHelp)
        self.reference.triggered.connect(self.goReference)

        #About Menu Action
        self.author_app.triggered.connect(self.goAuthor)

        ###### Button and Slider Clicked
        self.gsb.clicked.connect(self.goGrayScale)
        self.ngv.clicked.connect(self.goNegative)
        self.gma.clicked.connect(self.goGamma)

        self.edges_detect.clicked.connect(self.goEdges)
        self.hist.clicked.connect(self.goHistogram)
        self.medthre.clicked.connect(self.goMedianThreshold)
        self.reduce_noise.clicked.connect(self.goReduceNoise)
        self.sharpen.clicked.connect(self.goSharpen)
        self.saveImg.clicked.connect(self.goSave)
        self.resetImg.clicked.connect(self.goReset)

        self.brightSlid.setMinimum(1)
        self.blurSlid.valueChanged.connect(self.goBlur) #The slider are int value
        self.brightSlid.valueChanged.connect(self.goBrightness) #The slider are int value

        self.image = None
        self.disableAll()# While image are not imported, Button and Action are Disabled

        self.show()# Show the windows

    ############################# FILE HANDLING ACTION FUNCTION ##############################

    def goNew(self): # New File 
        fname, getImg = DIALOG.getOpenFileName(self, 'Open File', THE_DEFAULT_IMG, "Image Files (*)")
        if fname:
            self.image = cv2.imread(fname)
            self.tmp = self.image
            self.displayImage()
            self.enableAll()
        else:
            print("Invalid Image")

   
    def displayImage(self, window = 1): # Displaying Image to Frame (QPixmap)
        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if(self.image.shape) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat) # np.strides(self.image[0]) , self.image.strides[0]
        img = img.rgbSwapped()
        if window == 1:
            self.image_before.setPixmap(QPixmap.fromImage(img))
        if window == 2:
            self.image_after.setPixmap(QPixmap.fromImage(img))
    
    def goSave(self): # Save the Proceed Image
        fname, saveImg = DIALOG.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")
        if fname :
            cv2.imwrite(fname, self.image)
            print('Image saved as:', fname)
            

    def goExit(self): # Exit Program
        message = MESSAGE.question(self, "Exit", "Are you sure to exit?", MESSAGE.Yes | MESSAGE.No, MESSAGE.No)
        if message == MESSAGE.Yes:
            print("Yes")
            self.close()
        else:
            print("No")

    def enableAll(self): # Enabled all the buttons and actions
        self.saveFile.setEnabled(True)
        self.resizeImg.setEnabled(True)

        self.zoomInImg.setEnabled(True)
        self.zoomOutImg.setEnabled(True)
        self.rotImg45clock.setEnabled(True)
        self.rotImg45.setEnabled(True)
        self.rotImg90clock.setEnabled(True)
        self.rotImg90.setEnabled(True)

        self.powerLaw.setEnabled(True)
        self.bitPlane.setEnabled(True)

        self.gsb.setEnabled(True)
        self.ngv.setEnabled(True)
        self.gma.setEnabled(True)

        self.blurSlid.setEnabled(True)
        self.brightSlid.setEnabled(True)

        self.edges_detect.setEnabled(True)
        self.hist.setEnabled(True)
        self.medthre.setEnabled(True)
        self.sharpen.setEnabled(True)
        self.reduce_noise.setEnabled(True)

        self.saveImg.setEnabled(True)
        self.resetImg.setEnabled(True)

    def disableAll(self): # Disabled all the buttons and actions
        self.saveFile.setEnabled(False)

        self.resizeImg.setEnabled(False)
        self.zoomInImg.setEnabled(False)
        self.zoomOutImg.setEnabled(False)
        self.rotImg45clock.setEnabled(False)
        self.rotImg45.setEnabled(False)
        self.rotImg90clock.setEnabled(False)
        self.rotImg90.setEnabled(False)

        self.powerLaw.setEnabled(False)
        self.bitPlane.setEnabled(False)

        self.gsb.setEnabled(False)
        self.ngv.setEnabled(False)
        self.gma.setEnabled(False)

        self.blurSlid.setEnabled(False)
        self.brightSlid.setEnabled(False)

        self.edges_detect.setEnabled(False)
        self.hist.setEnabled(False)
        self.medthre.setEnabled(False)
        self.sharpen.setEnabled(False)
        self.reduce_noise.setEnabled(False)

        self.saveImg.setEnabled(False)
        self.resetImg.setEnabled(False)
        

    ########################## TRANSFORM ACTION FUNCTION ########################################

    def goResize(self): # Resize Image
        height = int(self.image.shape[0] * 200 / 100) # np.shape ==> (height, width, depth) ==> acessing like array
        width = int(self.image.shape[1] * 200 / 100)
        new_size = (height, width)

        self.image = cv2.resize(self.image, new_size, interpolation = cv2.INTER_CUBIC)
        self.displayImage(2)

    def goZoomIn(self): # Zoom In
        self.image = cv2.resize(self.image, 
                                None, 
                                fx=1.75, 
                                fy=1.75, 
                                interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def goZoomOut(self): # Zoom Out
        self.image = cv2.resize(self.image, 
                                None, 
                                fx=0.50, 
                                fy=0.50, 
                                interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def goRotation45clock(self): # Rotation 45 degree clockwise
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -45, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)


    def goRotation45(self): # Rotation 45 Degree
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)


    def goRotation90clock(self): # Rotation 90 degree clockwise
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)


    def goRotation90(self): # Rotation 90 degree
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)

    def goPowerLaw(self): # Power Law Transform
        self.image = self.tmp
        process = self.image / 255.0
        self.image = cv2.pow(process, 1.8)
        self.displayImage(2)

    def goBitPlane(self): # 8 Bit Plane
        self.image = self.tmp
        self.image = self.image.copy()
        bit_no = 8
        for i in range(0, self.image.shape[0]):
            for j in range(0, self.image.shape[1]):
                self.image[i, j] = (self.image[i, j] & 2**(bit_no - 1))
        
        self.displayImage(2)

    ########################### HELP ACTION FUNCTION ##############################
    def goHelp(self): # Help Action
        self.myWindow = MAIN()
        self.UI_window = Ui_HelpCenter()
        self.UI_window.setupUi(self.myWindow)
        self.myWindow.show()

    def goReference(self): # References Action
        self.myWindow = MAIN()
        self.UI_window = Ui_References()
        self.UI_window.setupUi(self.myWindow)
        self.myWindow.show()

    ############################# ABOUT ACTION FUNCTION ##################################
    def goAuthor(self): # Author Action
        self.myWindow = MAIN()
        self.UI_window = Ui_Authors_Window()
        self.UI_window.setupUi(self.myWindow)
        self.myWindow.show()

    ################################# BUTTON AND SLIDER FUNCTION #########################


    def goGrayScale(self): # GRAY SCALE
        self.image = self.tmp
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.displayImage(2)


    def goNegative(self): # NEGATIVE
        self.image = self.tmp
        self.image = 255 - self.image
        self.displayImage(2)


    def goGamma(self, gamma): # GAMMA
        self.image = self.tmp
        gamma = 3.0
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)


    def goEdges(self): # EDGES DETECTION
        self.image = self.tmp
        kernel = np.array([
            [-1, -1, -1],  
            [-1, 8, -1],    
            [-1, -1, -1]   
        ])
        self.image = cv2.filter2D(self.image, ddepth = -1, kernel = kernel)
        self.displayImage(2)

  
    def goHistogram(self): # MATPLOTLIB HISTOGRAM
        self.image = self.tmp
        histg = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        self.image = histg
        plt.plot(self.image)
        plt.show()
        self.displayImage(2)


    def goMedianThreshold(self): # MEDIAN THRESHOLD
        self.image = self.tmp
        grayscaled = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.medianBlur(self.image,5)
        retval, threshold = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.image = threshold
        self.displayImage(2)
    
    def goReduceNoise(self): # REDUCE NOISE
        self.image = self.tmp
        self.image = cv2.fastNlMeansDenoisingColored(self.image,None,10,10,7,21)
        self.displayImage(2)
        # src = self.image
        # dst = None
        # h = luminance component filter strength. bigger h perfectly ruce noise, but image details too
        # templateWindowsSize = should odd. 7 px is recommend
        # searchWindowSize = compute weighted avg given px. should px. 21 px is recommend

    def goSharpen(self):
        self.image = self.tmp
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        self.image = cv2.filter2D(self.image, ddepth = -1, kernel = kernel)
        self.displayImage(2)

    def goBrightness(self, gamma): # Brightness
        self.image = self.tmp
        gamma = gamma*0.1
        invGamma = 1.0 /gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)

    def goBlur(self, g): # Blur
        self.image = self.tmp
        self.image = cv2.GaussianBlur(self.image, (5, 5), g)
        self.displayImage(2)

    def goReset(self): # Reset all the changes
        self.image = self.tmp
        self.displayImage(2)
        
if __name__ == '__main__':
    app = APP(sys.argv)
    win = Image_Processing_Tools()
    win.show()
    sys.exit(app.exec())
