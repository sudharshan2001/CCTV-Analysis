import sys,PIL, os
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import  QApplication, QFileDialog, QMainWindow
from PyQt5.QtGui import QPixmap, QImage, QIcon
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication
from req_packages.ESRGAN.get_esrgan import enhance_clarity_esrgan, get_enhance_full, get_model_for_enhance
from vehicle_package.tracker import *
from general_package.general_file import *
import time
from person_package.person_detection import get_person_detection, get_person_detection_model, compare_faces
model = get_model()
feature_extractor = build_feature_extractor()

class WelcomeScreen(QMainWindow):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi(os.path.join("UIFiles","home.ui"),self)
        
        self.speedtrack.clicked.connect(self.gotospeedtrack) 
        self.person_track.clicked.connect(self.gotopersontrack)
        self.brightness_enhance_home.clicked.connect(self.gotobrightness_enhance_home) 
        self.clarity_enhance_home.clicked.connect(self.gotobrightness_clarity_home)
        self.brightness_general_tracking.clicked.connect(self.gotobrightness_general_tracking)

    def gotobrightness_general_tracking(self):
        two_dim_window = CreateGeneralTrackingSelection()
        widget.addWidget(two_dim_window)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def gotobrightness_enhance_home(self):
        two_dim_window = CreateSpeedTrack()
        widget.addWidget(two_dim_window)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def gotobrightness_clarity_home(self):
        two_dim_window = CreateClarityWindowSelection()
        widget.addWidget(two_dim_window)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def gotospeedtrack(self):
        two_dim_window = CreateSpeedTrack()
        widget.addWidget(two_dim_window)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def gotopersontrack(self):
        person_track_windows = CreatePersonSelection()
        widget.addWidget(person_track_windows)
        widget.setCurrentIndex(widget.currentIndex()+1)

class CreateClarityWindowSelection(QMainWindow):
    def __init__(self):
        super(CreateClarityWindowSelection, self).__init__()
        loadUi(os.path.join("UIFiles","tracking_person2.ui"), self)
        # self.start_video_person.clicked.connect(self.select_person_from_folder) 
        self.pathFrontglobal=''

        self.cancel_trackingtwo.clicked.connect(self.go_to_backhome2)
        self.dimselector.clicked.connect(self.check_for_error)
        self.folderselect.clicked.connect(self.open_front_box)
        self.keep_running = True
        self.brightness = 1.0
        self.contrast = 1.0

    def go_to_backhome2(self):
        back_to_home2 = WelcomeScreen()
        widget.addWidget(back_to_home2)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        
    def check_for_error(self):
        if len(self.pathFrontglobal) < 1:
            self.error.setText("Please Select the Path to Video")
        else:
            try:
                self.frame_count = 0
                # enhance_model = get_model_for_enhance()
                self.frame1 = cv2.imread(self.pathFrontglobal)
                self.brightness_scale.valueChanged.connect(self.brightness_update)
                self.clarity_scale.valueChanged.connect(self.clarity_update)
                QApplication.processEvents()
                
                self.frame1=cv2.resize(self.frame1, (380,380), interpolation = cv2.INTER_AREA)
                # prediction = get_enhance_full(self.frame1, enhance_model)
                prediction = cv2.detailEnhance(self.frame1, sigma_s=self.brightness, sigma_r=self.contrast)
                # prediction= cv2.addWeighted(self.frame1, self.contrast, np.zeros(self.frame1.shape, self.frame1.dtype), 0, self.brightness)
                # kernel1 = np.array([[-1, -1, -1],
                #         [-1, 8, -1],
                #         [-1, -1, -1]])
                # prediction = cv2.filter2D(prediction, ddepth=-1, kernel=kernel1)
                prediction=cv2.resize(prediction, (380,380), interpolation = cv2.INTER_AREA)
                # cv2.imshow('dummy',)
                prediction = np.array(prediction, dtype=np.uint8)
                
                print(prediction.shape)
                # cv2.imshow('dummy', prediction)
                self.frame1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
                self.frame1 = QImage(self.frame1.data.tobytes(), self.frame1.shape[1],self.frame1.shape[0],QImage.Format_RGB888)
                self.original_photo.setPixmap(QPixmap.fromImage(self.frame1))
                self.original_photo.setScaledContents(True)

                prediction = QImage(prediction.data.tobytes(), prediction.shape[1],prediction.shape[0],QImage.Format_RGB888)
                self.enhance_photo.setPixmap(QPixmap.fromImage(prediction))
                self.enhance_photo.setScaledContents(True)
                self.error.setText("done")
            except:
                self.check_for_error()
    def open_front_box(self):
        filename = QFileDialog.getOpenFileName(filter="Video Files (*.png *.jpg *.jpeg)")
        self.pathFrontglobal = filename[0]

        if len(self.pathFrontglobal) > 1:
            self.pathlabel.setText(self.pathFrontglobal )

    def brightness_update(self,value):
        self.brightness = value
        print(value, self.brightness)
        self.check_for_error()

    def clarity_update(self, value):
        self.contrast = value/100.0
        print(value, self.contrast)
        self.check_for_error()

class CreateGeneralTrackingSelection(QMainWindow):
    def __init__(self):
        super(CreateGeneralTrackingSelection, self).__init__()
        loadUi(os.path.join("UIFiles","tracking_person.ui"), self)
        # self.start_video_person.clicked.connect(self.select_person_from_folder) 
        self.pathFrontglobal=''

        self.cancel_trackingtwo.clicked.connect(self.go_to_backhome2)
        self.dimselector.clicked.connect(self.check_for_error)
        self.folderselect.clicked.connect(self.open_front_box)

    def go_to_backhome2(self):
        back_to_home2 = WelcomeScreen()
        widget.addWidget(back_to_home2)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        
    def check_for_error(self):
        if len(self.pathFrontglobal) < 1:
            self.error.setText("Please Select the Path to Video")
        else:
            popup_window =  CreateGeneralTracking(self.pathFrontglobal)
            widget.addWidget(popup_window)
            widget.setCurrentIndex(widget.currentIndex() + 1)

    def open_front_box(self):
        filename = QFileDialog.getOpenFileName(filter="Video Files (*.avi *.mp4)")
        self.pathFrontglobal = filename[0]

        if len(self.pathFrontglobal) > 1:
            self.pathlabel.setText(self.pathFrontglobal )

class CreateGeneralTracking(QMainWindow):
    def __init__(self, path_to_track):
        super(CreateGeneralTracking, self).__init__()
        loadUi(os.path.join("UIFiles","generaltracking.ui"), self)
        self.path_to_track= path_to_track
        self.start_generaltracking.clicked.connect(self.start_general_track) 
        self.frame_count = 0
        self.keep_running = True

        self.cancel_generaltracking.clicked.connect(self.cancel_genetal_track)

    def cancel_genetal_track(self):
        self.keep_running = False
        back_to_home2 = WelcomeScreen()
        widget.addWidget(back_to_home2)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def start_general_track(self):
        QApplication.processEvents()
        cam1 = cv2.VideoCapture(self.path_to_track)
        total_frames = int(cam1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(cam1.get(cv2.CAP_PROP_FPS))
        temp_framelist = []
        self.temp_embeds = []
        array_count = 0
        
        while self.frame_count<total_frames and self.keep_running:
            cam1.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
            ret1,self.frame1 = cam1.read()
            self.frame_count += frame_interval
            QApplication.processEvents()
            try:
                self.frame1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
                self.frame1=cv2.resize(self.frame1, (224,224), interpolation = cv2.INTER_AREA)
                normalized_frame = self.frame1 / 255.0
                self.temp_embeds.append(feature_extractor.predict( np.expand_dims(normalized_frame,axis=0), verbose=False)[0])           
                # print(np.asarray(self.temp_embeds).shape)

                if np.asarray(self.temp_embeds).shape[0]>SEQUENCE_LENGTH:
                    self.temp_embeds.pop(0)
                    # print(np.asarray(self.temp_embeds).shape)
                    
                    prediction = model.predict(np.expand_dims(np.asarray(self.temp_embeds),axis=0))
                    print(prediction)
                    self.ok_generaltracking.setText(CLASSES_LIST[np.argmax(prediction)])
                self.frame1=cv2.resize(self.frame1, (608,608), interpolation = cv2.INTER_AREA)
                self.frame1 = QImage(self.frame1.data.tobytes(), self.frame1.shape[1],self.frame1.shape[0],QImage.Format_RGB888)
                self.embedvideo_generaltracking.setPixmap(QPixmap.fromImage(self.frame1))
                self.embedvideo_generaltracking.setScaledContents(True)
            except:
                self.ok_generaltracking.setText(CLASSES_LIST[0])

class CreatePersonSelection(QMainWindow):
    def __init__(self):
        super(CreatePersonSelection, self).__init__()
        loadUi(os.path.join("UIFiles","tracking_person.ui"), self)
        # self.start_video_person.clicked.connect(self.select_person_from_folder) 
        self.pathFrontglobal=''

        self.cancel_trackingtwo.clicked.connect(self.go_to_backhome2)
        self.dimselector.clicked.connect(self.check_for_error)
        self.folderselect.clicked.connect(self.open_front_box)

    def go_to_backhome2(self):
        back_to_home2 = WelcomeScreen()
        widget.addWidget(back_to_home2)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        
    def check_for_error(self):
        if len(self.pathFrontglobal) < 1:
            self.error.setText("Please Select the Path to Video")
        else:
            popup_window =  CreatePersonTrack(self.pathFrontglobal)
            widget.addWidget(popup_window)
            widget.setCurrentIndex(widget.currentIndex() + 1)

    def open_front_box(self):
        filename = QFileDialog.getOpenFileName(filter="Video Files (*.avi *.mp4)")
        self.pathFrontglobal = filename[0]

        if len(self.pathFrontglobal) > 1:
            self.pathlabel.setText(self.pathFrontglobal )

class CreatePersonTrack(QMainWindow):
    def __init__(self, path_to_person):
        super(CreatePersonTrack, self).__init__()
        loadUi(os.path.join("UIFiles","trackingperson.ui"), self)

        self.start_video_person.clicked.connect(self.start_person_track) 
        
        self.path_to_search_person = None
        self.path_to_person= path_to_person
        self.frame_count = 0
        self.keep_running = True

        self.video_cancel_person.clicked.connect(self.cancel_person_track)
        
        
    def cancel_person_track(self):
        self.keep_running = False
        back_to_home2 = WelcomeScreen()
        widget.addWidget(back_to_home2)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def start_person_track(self):

        QApplication.processEvents()
        self.folderselect_person.clicked.connect(self.select_person_from_folder)
        
        cam1 = cv2.VideoCapture(self.path_to_person)
        total_frames = int(cam1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(cam1.get(cv2.CAP_PROP_FPS))
        
        while self.frame_count<total_frames and self.keep_running:
            cam1.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
            ret1,self.frame1 = cam1.read()
            self.frame_count += frame_interval
            QApplication.processEvents()
            person_id_model = get_person_detection_model()
            if ret1:
                self.frame1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
                self.frame1=cv2.resize(self.frame1, (608,608), interpolation = cv2.INTER_AREA)
                
                if self.path_to_search_person is not None:
                    self.enhance_clarity_person.clicked.connect(self.enhance_clarity)
                    QApplication.processEvents()
                    # self.frame1 = compare_faces(self.frame1)
                    # self.frame1,frame_cropped = get_person_detection(self.frame1,person_id_model)
                    self.frame1 = get_person_detection(self.frame1,person_id_model,self.image_to_search_person)
                    self.frame1 = QImage(self.frame1.data.tobytes(), self.frame1.shape[1],self.frame1.shape[0],QImage.Format_RGB888)
                    self.video_embed_person.setPixmap(QPixmap.fromImage(self.frame1))
                    self.video_embed_person.setScaledContents(True)
                    
                    self.frame2 = QImage(self.image_to_search_person.data.tobytes(), self.image_to_search_person.shape[1],self.image_to_search_person.shape[0],QImage.Format_RGB888)
                    self.video_embed_person_to_check.setPixmap(QPixmap.fromImage(self.frame2))
                    self.video_embed_person_to_check.setScaledContents(True)
                    
                else:  
                    try:
                        QApplication.processEvents()
                        
                        # cv2.imshow('ddd', frame_cropped)
                        self.frame1 = QImage(self.frame1.data.tobytes(), self.frame1.shape[1],self.frame1.shape[0],QImage.Format_RGB888)
                        self.video_embed_person.setPixmap(QPixmap.fromImage(self.frame1))
                        self.video_embed_person.setScaledContents(True)
                    except:
                        print(self.path_to_search_person)
        
                 

    def select_person_from_folder(self):
        if self.keep_running:
            QApplication.processEvents()
            filename = QFileDialog.getOpenFileName(filter="Video Files (*.png *.jpeg *.jpg)")
            self.path_to_search_person = filename[0]
            self.image_to_search_person = cv2.imread(self.path_to_search_person)

            self.image_to_search_person = cv2.cvtColor(self.image_to_search_person, cv2.COLOR_BGR2RGB)
            self.image_to_search_person=cv2.resize(self.image_to_search_person, (224,224), interpolation = cv2.INTER_AREA)
            
            self.start_person_track()

    def enhance_clarity(self):
        
        self.image_to_search_person=enhance_clarity_esrgan(self.image_to_search_person)
        self.image_to_search_person = np.array(self.image_to_search_person, dtype=np.uint8)
        self.image_to_search_person=cv2.resize(self.image_to_search_person, (224,224), interpolation = cv2.INTER_AREA)

        self.start_person_track()

# Vehicle Track
class CreateSpeedTrack(QMainWindow):
    def __init__(self):
        super(CreateSpeedTrack, self).__init__()
        loadUi(os.path.join("UIFiles","tracking2.ui"), self)
        self.pathTopglobal=''
        self.pathFrontglobal=''

        self.cancel_trackingtwo.clicked.connect(self.go_to_backhome2)
        self.dimselector.clicked.connect(self.check_for_error)
        self.folderselect.clicked.connect(self.open_front_box)
          
        
    def go_to_backhome2(self):
        back_to_home2 = WelcomeScreen()
        widget.addWidget(back_to_home2)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        
    def check_for_error(self):
        if len(self.pathFrontglobal) < 1:
            self.error.setText("Please Select the Path to Video")

        else:
            self.error.setText('')

            self.pathTopglobal = self.pathFrontglobal
            cam1 = cv2.VideoCapture(self.pathTopglobal)
            while(True):
                ret1,frame1 = cam1.read()
                if ret1:

                    path_to_img_filetop = './Temporary Files/' + str(self.pathTopglobal.split('/')[-1][:-4]) + '.jpg'
                    print(str(self.pathTopglobal.split('/')[-1][:-4]))
            
                    cv2.imwrite(path_to_img_filetop, frame1)
                    break
                break

            cam2 = cv2.VideoCapture(self.pathFrontglobal)
            while(True):
                ret2,frame2 = cam2.read()
                if ret2:

                    path_to_img_filefront = './Temporary Files/' + str(self.pathFrontglobal.split('/')[-1][:-4]) + '.jpg'
                    print(str(self.pathFrontglobal.split('/')[-1][:-4]))
            
                    cv2.imwrite(path_to_img_filefront, frame2)
                    break
                break
        popup_window =  createpopupforthreedimimage(path_to_img_filetop, path_to_img_filefront,self.pathTopglobal,self.pathFrontglobal)
        widget.addWidget(popup_window)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def open_front_box(self):
        filename = QFileDialog.getOpenFileName(filter="Video Files (*.avi *.mp4)")
        print(filename)
        self.pathFrontglobal = filename[0]
        pathtofront = filename[0]
        if len(self.pathFrontglobal) > 1:
            self.pathlabel.setText(pathtofront)

# Three Dimension Top View Cropping Window
class createpopupforthreedimimage(QMainWindow):
    def __init__(self, path_to_img_filetop, path_to_img_filefront,pathTopglobal,pathFrontglobal):
        super(createpopupforthreedimimage, self).__init__()
        loadUi(os.path.join("UIFiles","dimensionselect.ui"), self)
        self.path_to_img_filetop = path_to_img_filetop
        self.path_to_img_filefront = path_to_img_filefront

        self.pathTopglobal=pathTopglobal
        self.pathFrontglobal = pathFrontglobal

        cropping = False
        img = Image.open(path_to_img_filetop)
        img=img.resize((608, 608), PIL.Image.ANTIALIAS)
        file_cropped_save_name = path_to_img_filetop[:-4]+'resized_image.jpg'
        img.save(file_cropped_save_name)

        self.image = cv2.imread(file_cropped_save_name)
        oriImage =self.image.copy()
        self.oriImage2 =self.image.copy()

        x_start, y_start, x_end, y_end = 0, 0, 0, 0
        def mouse_crop(event, x, y, flags, param):
            global x_start, y_start, x_end, y_end, cropping

            if event == cv2.EVENT_LBUTTONDOWN:
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            elif event == cv2.EVENT_MOUSEMOVE:
                try:
                    if cropping == True:
                        x_end, y_end = x, y
                except:
                    pass

            elif event == cv2.EVENT_LBUTTONUP:
                
                x_end, y_end = x, y
                cropping = False  

                refPoint = [(x_start, y_start), (x_end, y_end)]
                self.toprefPoint = refPoint


                if len(refPoint) == 2:  


                    cv2.cvtColor(self.oriImage2, cv2.COLOR_BGR2RGB, self.oriImage2)
                    self.roi=cv2.resize(self.oriImage2, (608,608), interpolation = cv2.INTER_AREA)
                    cv2.line(self.roi, refPoint[0], refPoint[1], (0, 255, 0) , thickness=2)
                    self.roi = QImage(self.roi.data.tobytes(), self.roi.shape[1],self.roi.shape[0],QImage.Format_RGB888)
                    self.front_line.setPixmap(QPixmap.fromImage(self.roi))
                    self.front_line.setScaledContents(True)
                    cv2.destroyAllWindows()
                    self.retrybuttondim2.clicked.connect(self.crop_again)
                    self.dimcancel2.clicked.connect(self.go_back_to_3Dtrack)
                    self.dimok2.clicked.connect(self.gotofrontview)
                    
                    
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)
        
        while True:
            self.i = self.image.copy()

            if not cropping:
                cv2.imshow("image", self.image)
                
            elif cropping:
                cv2.rectangle(self.i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("image", self.i)
            break

    def gotofrontview(self):
        three_dim_window_front = Createthreedimefront(self.path_to_img_filefront,self.path_to_img_filetop,self.toprefPoint,self.pathTopglobal,self.pathFrontglobal)
        widget.addWidget(three_dim_window_front)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def go_back_to_3Dtrack(self):
        three_dim_window = CreateSpeedTrack()
        widget.addWidget(three_dim_window)
        widget.setCurrentIndex(widget.currentIndex() + 1)
          
    def crop_again(self):
        x_start, y_start, x_end, y_end = 0, 0, 0, 0
        def mouse_crop(event, x, y, flags, param):
            global x_start, y_start, x_end, y_end, cropping

            if event == cv2.EVENT_LBUTTONDOWN:
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            elif event == cv2.EVENT_MOUSEMOVE:
                try:
                    if cropping == True:
                        x_end, y_end = x, y
                except:
                    pass

            elif event == cv2.EVENT_LBUTTONUP:
                
                x_end, y_end = x, y
                cropping = False  

                refPoint = [(x_start, y_start), (x_end, y_end)]
                self.toprefPoint = refPoint

                if len(refPoint) == 2:  
                    
                    
                    cv2.cvtColor(self.oriImage2 , cv2.COLOR_BGR2RGB, self.oriImage2 )
                    self.roi=cv2.resize(self.oriImage2, (608,608), interpolation = cv2.INTER_AREA)
                    cv2.line(self.roi, refPoint[0], refPoint[1], (0, 255, 0) , thickness=2)
                    self.roi = QImage(self.roi.data.tobytes(), self.roi.shape[1],self.roi.shape[0],QImage.Format_RGB888)
                    self.front_line.setPixmap(QPixmap.fromImage(self.roi))
                    self.front_line.setScaledContents(True)
                    cv2.destroyAllWindows()
                    self.retrybuttondim2.clicked.connect(self.crop_again)
                    self.dimcancel2.clicked.connect(self.go_back_to_3Dtrack)
                    self.dimok2.clicked.connect(self.gotofrontview)
                    
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)
        
        while True:
            self.i = self.image.copy()

            if not cropping:
                cv2.imshow("image", self.image)

            elif cropping:
                cv2.rectangle(self.i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("image", self.i)
            break

# Three Dimension Front View Cropping Window
class Createthreedimefront(QMainWindow):
    def __init__(self,path_to_img_filefront,path_to_img_filetop,toprefPoint,pathTopglobal,pathFrontglobal):
        super(Createthreedimefront, self).__init__()
        loadUi(os.path.join("UIFiles","dimensionselectend.ui"), self)

        self.path_to_img_filefront = path_to_img_filefront
        self.toprefPoint = toprefPoint
        self.path_to_img_filetop =path_to_img_filetop
        self.pathTopglobal=pathTopglobal
        self.pathFrontglobal = pathFrontglobal

        cropping = False
        img = Image.open(path_to_img_filefront)
        img=img.resize((608, 608), PIL.Image.ANTIALIAS)
        file_cropped_save_name = path_to_img_filefront[:-4]+'resized_image.jpg'
        img.save(file_cropped_save_name)

        self.image = cv2.imread(file_cropped_save_name)
        oriImage =self.image.copy()
        self.oriImage2 =self.image.copy()

        x_start, y_start, x_end, y_end = 0, 0, 0, 0
        def mouse_crop(event, x, y, flags, param):
            global x_start, y_start, x_end, y_end, cropping

            if event == cv2.EVENT_LBUTTONDOWN:
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            elif event == cv2.EVENT_MOUSEMOVE:
                try:
                    if cropping == True:
                        x_end, y_end = x, y
                except:
                    pass

            elif event == cv2.EVENT_LBUTTONUP:
                
                x_end, y_end = x, y
                cropping = False  

                refPoint = [(x_start, y_start), (x_end, y_end)]
                self.frontrefPoint = refPoint
                if len(refPoint) == 2:  
                    self.roi = self.oriImage2
                    # cv2.imshow("Cropped", self.roi)
                    cv2.cvtColor(self.roi, cv2.COLOR_BGR2RGB, self.roi)
                    self.roi=cv2.resize(self.roi, (608,608), interpolation = cv2.INTER_AREA)
                    cv2.line(self.roi, refPoint[0], refPoint[1], (0, 255, 0) , thickness=2)

                    self.roi = QImage(self.roi.data.tobytes(), self.roi.shape[1],self.roi.shape[0],QImage.Format_RGB888)
                    self.endline.setPixmap(QPixmap.fromImage(self.roi))
                    self.endline.setScaledContents(True)
                    cv2.destroyAllWindows()
                    self.retrybuttondim.clicked.connect(self.crop_again)
                    self.dimcancel.clicked.connect(self.go_back_to_3Dtrack)
                    self.dimok.clicked.connect(self.goto3DPredicting)
                    
                    
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)
        
        while True:
            self.i = self.image.copy()

            if not cropping:
                cv2.imshow("image", self.image)
                
            elif cropping:
                cv2.rectangle(self.i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("image", self.i)
            break

    def go_back_to_3Dtrack(self):
        three_dim_window = CreateSpeedTrack()
        widget.addWidget(three_dim_window)
        widget.setCurrentIndex(widget.currentIndex() + 1)
          
    def crop_again(self):
        x_start, y_start, x_end, y_end = 0, 0, 0, 0
        def mouse_crop(event, x, y, flags, param):
            global x_start, y_start, x_end, y_end, cropping

            if event == cv2.EVENT_LBUTTONDOWN:
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            elif event == cv2.EVENT_MOUSEMOVE:
                try:
                    if cropping == True:
                        x_end, y_end = x, y
                except:
                    pass

            elif event == cv2.EVENT_LBUTTONUP:
                
                x_end, y_end = x, y
                cropping = False  

                refPoint = [(x_start, y_start), (x_end, y_end)]
                self.frontrefPoint = refPoint
                
                print(refPoint)
                if len(refPoint) == 2:  
                    self.roi = self.oriImage2
                    
                    cv2.cvtColor(self.roi, cv2.COLOR_BGR2RGB, self.roi)
                    self.roi=cv2.resize(self.roi, (608,608), interpolation = cv2.INTER_AREA)
                    cv2.line(self.roi, refPoint[0], refPoint[1], (0, 255, 0) , thickness=2)

                    self.roi = QImage(self.roi.data.tobytes(), self.roi.shape[1],self.roi.shape[0],QImage.Format_RGB888)
                    self.endline.setPixmap(QPixmap.fromImage(self.roi))
                    self.endline.setScaledContents(True)

                    cv2.destroyAllWindows()
                    self.retrybuttondim.clicked.connect(self.crop_again)
                    self.dimcancel.clicked.connect(self.go_back_to_3Dtrack)
                    self.dimok.clicked.connect(self.goto3DPredicting)
                    
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)
        
        while True:
            self.i = self.image.copy()

            if not cropping:
                cv2.imshow("image", self.image)
                
            elif cropping:
                cv2.rectangle(self.i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("image", self.i)
            break

    def goto3DPredicting(self):
        twodimloader = Createthreedimloader(self.pathTopglobal,self.pathFrontglobal,self.toprefPoint,self.frontrefPoint)
        widget.addWidget(twodimloader)
        widget.setCurrentIndex(widget.currentIndex() + 1)

# Three Dimension Completion Window
class Createthreedimloader(QMainWindow):
    def __init__(self,pathTopglobal,pathFrontglobal,firstredpoint,endrefpoint):
        super(Createthreedimloader, self).__init__()
        loadUi(os.path.join("UIFiles","runspeedtrack.ui"), self)

        self.pathTopglobal = pathTopglobal
        self.pathFrontglobal = pathFrontglobal
        self.toprefPoint = firstredpoint
        self.frontrefPoint = endrefpoint
        print(self.toprefPoint ,self.frontrefPoint )
        self.frame_count = 0
        self.keep_running = True
        print(self.toprefPoint[0][1],self.toprefPoint[1][1] ,self.frontrefPoint[0][1],self.frontrefPoint[1][1] )
        self.start_video.clicked.connect(self.proceed)
        # self.cancelthreed.clicked.connect(self.go_to_backhome)
        
    def proceed(self):    

        QApplication.processEvents()
        end = 0

        tracker = EuclideanDistTracker()
        cap = cv2.VideoCapture(self.pathTopglobal)
        f = 25.0
        w = int(1000/(f-1))

        object_detector = cv2.createBackgroundSubtractorMOG2(history=None,varThreshold=None)

        kernalOp = np.ones((3,3),np.uint8)
        kernalOp2 = np.ones((5,5),np.uint8)
        kernalCl = np.ones((11,11),np.uint8)

        fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        kernal_e = np.ones((5,5),np.uint8)

        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS))
        
        while True:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
            ret,frame  = cap.read()
            # self.frame_count += frame_interval
            QApplication.processEvents()
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            # self.roi=cv2.resize(self.roi, (608,608), interpolation = cv2.INTER_AREA)
            if not ret:
                break
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            height,width,_ = frame.shape
            print(height, width)
            roi = frame
            
            mask = object_detector.apply(roi)
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

            fgmask = fgbg.apply(roi)
            ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
            mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
            e_img = cv2.erode(mask2, kernal_e)


            _,contours,_ = cv2.findContours(e_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            detections = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                #THRESHOLD
                if area > 1000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
                    
                    detections.append([x,y,w,h])

            #Object Tracking
            boxes_ids = tracker.update(detections)

            for box_id in boxes_ids:
                x,y,w,h,id = box_id

                if(tracker.getsp(id)<tracker.limit()):
                    cv2.putText(roi,str(id)+" "+str(tracker.getsp(id)),(x,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                else:
                    cv2.putText(roi,str(id)+ " "+str(tracker.getsp(id)),(x, y-15),cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255),2)
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3)

                s = tracker.getsp(id)
                print(x, y, h, w, s, id)
                if (tracker.f[id] == 1 and s != 0):
                    tracker.capture(roi, x, y, h, w, s, id)

            # DRAW LINES

            cv2.line(roi, (0, 410), (960, 410), (0, 0, 255), 2)
            # cv2.line(roi, (0, 430), (960, 430), (0, 0, 255), 2)

            cv2.line(roi, (0, 235), (960, 235), (0, 0, 255), 2)
            # cv2.line(roi, (0, 255), (960, 255), (0, 0, 255), 2)
            QApplication.processEvents()
            print(roi.shape)
            roi = QImage(roi.data.tobytes(), roi.shape[1],roi.shape[0],QImage.Format_RGB888)
            self.video_embed.setPixmap(QPixmap.fromImage(roi))
            self.video_embed.setScaledContents(True)

    def go_to_backhome(self):
        
        back_to_home1 = WelcomeScreen()
        widget.addWidget(back_to_home1)
        widget.setCurrentIndex(widget.currentIndex() + 1)


app = QApplication(sys.argv)
welcome = WelcomeScreen()

widget = QtWidgets.QStackedWidget()
widget.addWidget(welcome)
widget.setFixedHeight(900)
widget.setFixedWidth(1200)
widget.setWindowTitle('Chennaipol')
widget.setWindowIcon(QIcon(os.path.join('icons','icon.png')))
widget.show()

try:
    sys.exit(app.exec_())
except:
    print("Exiting")