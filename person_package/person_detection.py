import torch, torchvision, cv2, time
from utils.cfg import CFG
from utils.detection_utils import detect_utils
import cv2
import numpy as np
import math
import torchvision.transforms as transforms
from utils.detection_utils.coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
from person_package.presonreid import  build_person_reid_model,predict_person_reid

def compare_faces(img):
    face_detector=cv2.CascadeClassifier("./req_packages/Haar/haarcascade_frontalface_alt_tree.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = face_detector.detectMultiScale(gray, 1.3, 5)
    
    cv2.imshow('dummy', gray[y - 5:y + h + 5, x - 5:x + w + 5])
    for (x,y,w,h) in results:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    

    return img
def get_person_detection_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =  torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    # model = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)

    model = model.eval().to(device)
    return model
def get_person_detection(frame, model, frame2):
    # reid_model = build_person_reid_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    count = 0
    center_points_prev_frame = []

    tracking_objects = {}
    track_id = 0
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    count += 1

    # Point current frame
    center_points_cur_frame = []

    image = transform(frame).to(device)
    # add a batch dimension
    image = image.unsqueeze(0) 

    with torch.no_grad():# Detect objects on frame
        outputs = model(image) # (class_ids, scores, boxes) 

    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]

    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes = pred_bboxes[pred_scores >= CFG.THRESHOLD].astype(np.int32)

    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
        # print(predict_person_reid(frame2,frame[y - 5:y + h + 5, x - 5:x + w + 5], reid_model))
    # print("Tracking objects")
    # print(tracking_objects)


    # print("CUR FRAME LEFT PTS")
    # print(center_points_cur_frame)

    # cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    # key = cv2.waitKey(1)
    # if key == 27:
    #     break
    return frame

    # cv2.destroyAllWindows()

# cap = cv2.VideoCapture('./test/lot.avi')

# if (cap.isOpened() == False):
#     print('Error reading files')

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# frame_count = 0 
# total_fps = 0 

# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# frame_interval = int(cap.get(cv2.CAP_PROP_FPS))
# frame_count = 0

# while frame_count<total_frames:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
#     ret, frame = cap.read()
#     if ret == True:

#         start_time = time.time()
#         with torch.no_grad():

#             boxes, classes, labels = detect_utils.predict(frame, model, device, CFG.THRESHOLD)

#         image = detect_utils.draw_boxes(boxes, classes, labels, frame)

#         end_time = time.time()
#         frame_count +=  frame_interval

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         cv2.imshow('image', image)
#         # press `q` to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()