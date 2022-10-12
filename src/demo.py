import argparse
import cv2
import time
import math
import onnxruntime
import numpy as np
from math import cos, sin
import mediapipe as mp
import os

# HEADPOSE DRAW FUNC
def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50, img_size=50):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    if math.isnan(yaw) or math.isnan(pitch):
        return img
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    if math.isnan(roll):
        print('roll is nan')
    else:
        roll = roll * np.pi / 180
        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
    # Z-Axis (out of the screen) drawn in blue
    # x3 = size * (sin(yaw)) + tdx
    # y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    x3 = img_size * (sin(yaw)) + tdx
    y3 = img_size * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

#
def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]
    if math.isnan(yaw) or math.isnan(pitch):
        return img

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

# BBOX, HEADPOSE DRAW
def draw_bbox_axis(frame, face_pos, add_face, yaw, pitch, roll, draw_bbox=0, draw_cube=1, draw_line=0):

    (x, y, w, h) = face_pos
    (x2, y2) = add_face
    w = x2-x
    h = y2-y

    # Draw bbox
    if draw_bbox:
        deg_norm = 1.0 - abs(yaw / 180)
        blue = int(255 * deg_norm)
        cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), color=(blue, 0, 255 - blue), thickness=2)

    # Draw pose cube
    if draw_cube:
        frame = plot_pose_cube(frame, yaw, pitch, roll, tdx=x + w / 2, tdy=y + h / 2, size=w)

    # Draw pose axis
    if draw_line:
        frame = draw_axis(frame, yaw, pitch, roll, tdx=x + w / 2, tdy=y + h / 2, size=w // 2)

    return frame

# ONNX LOAD
def load_onnx_model(path, name):
    onnx_model = onnxruntime.InferenceSession(path_or_bytes=os.path.join((os.getcwd() + os.path.sep).split('src')[0], 'models', path))
    globals()['onnx_input_{}'.format(name)] = onnx_model.get_inputs()[0].name
    print(">>> onnx model load : {}".format(name))
    print(">>> input name : {}".format(onnx_model.get_inputs()[0].name))
    print(">>> input shape : {}".format(onnx_model.get_inputs()[0].shape))
    print(">>> done.\n")
    return onnx_model

# 6DREPNET
def headpose_6drepnet2(rgb_img, x, y, x2, y2, onnx_input_sixdrepnet, sixdrepnet_model):

    face_img = rgb_img[y:y2, x:x2, :]

    face_img = cv2.resize(face_img, (256, 256))
    face_img = face_img[16:240,16:240,0:3]

    # 공식 깃헙 노말라이즈 구현
    face_img = np.array(face_img, dtype=np.uint8)
    face_img = face_img / 255
    face_img[:,:,0] = (face_img[:,:,0] - 0.485) / 0.229
    face_img[:,:,1] = (face_img[:,:,1] - 0.456) / 0.224
    face_img[:,:,2] = (face_img[:,:,2] - 0.406) / 0.225

    face_img = face_img.transpose(2, 0, 1)

    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.array(face_img, dtype=np.float32)

    st_time = time.time()
    outputs = sixdrepnet_model.run(None, input_feed={onnx_input_sixdrepnet: face_img})[0]

    R = outputs
    sy = np.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6

    x = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    y = np.arctan2(-R[:, 2, 0], sy)
    z = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    xs = np.arctan2(-R[:,1,2], R[:,1,1])
    ys = np.arctan2(-R[:,2,0], sy)
    zs = R[:, 1, 0] * 0

    pitch = (x * (1 - singular) + xs * singular)[0] * 180 / np.pi
    yaw = (y * (1 - singular) + ys * singular)[0] * 180 / np.pi
    roll = (z * (1 - singular) + zs * singular)[0] * 180 / np.pi

    print(">>> 6DREPNET Use Time : {}".format(time.time() - st_time))
    print(yaw, pitch, roll)

    return yaw, pitch, roll

# MAIN
def main(draw_bbox, draw_cube, draw_line):

    # Load 6DPRepNet
    sixdrepnet_model = load_onnx_model(path='sixdrepnet.onnx', name='sixdrepnet')

    # Load Mediapipe to predict Face
    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.9)

    # Capture
    cap = cv2.VideoCapture(0)

    # Start Loop
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        output_frame = frame.copy()
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start_time, st_time = time.time(), time.time()

        # Face Detect (using Mediapipe)
        detected = face_detection.process(rgb_img)
        print(">>> BlazeFace Use Time : {}".format(time.time() - st_time))

        if detected.detections:

            face_pos = detected.detections[0].location_data.relative_bounding_box
            x = int(rgb_img.shape[1] * max(face_pos.xmin, 0))
            y = int(rgb_img.shape[0] * max(face_pos.ymin, 0))
            w = int(rgb_img.shape[1] * min(face_pos.width, 1))
            h = int(rgb_img.shape[0] * min(face_pos.height, 1))

            # bbox
            face_plus_scalar = 5
            x2 = min(x + w + face_plus_scalar, rgb_img.shape[1])
            y2 = min(y + h + face_plus_scalar, rgb_img.shape[0])
            x = max(0, x - face_plus_scalar)
            y = max(0, y - face_plus_scalar)
            face_pos = (x, y, w, h)

            # headpose
            yaw, pitch, roll = headpose_6drepnet2(rgb_img, x, y, x2, y2, onnx_input_sixdrepnet, sixdrepnet_model)

            # draw bbox, axis
            draw_bbox_axis(output_frame, face_pos, (x2, y2), yaw, pitch, roll,
                           draw_bbox=draw_bbox, draw_cube=draw_cube, draw_line=draw_line)

        print(">>> Total Loop Time : {}\n".format(time.time() - start_time))
        cv2.imshow('demo', output_frame)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='6DRepNet to ONNX')
    parser.add_argument('--draw_bbox', default=1, type=int)
    parser.add_argument('--draw_cube', default=0, type=int)
    parser.add_argument('--draw_line', default=1, type=int)
    args = parser.parse_args()

    main(args.draw_bbox, args.draw_cube, args.draw_line)