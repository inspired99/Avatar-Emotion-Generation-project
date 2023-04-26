import dlib
from PIL import Image
import time
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import multiprocessing as mp
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class FaceExtractor:
    def __init__(self, save_name, video_path, output_path, predictor_path, frame_count, black_width=89):
        self.save_name = save_name
        self.video_path = video_path
        self.output_path = output_path
        self.predictor_path = predictor_path
        self.black_width = black_width
        self.frame_count = frame_count
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.detected_faces = 0

    @staticmethod
    def align_face(img, lm, output_size=512, transform_size=512, enable_padding=True):
        # landmarks
        lm_eye_left = lm[36: 42]
        lm_eye_right = lm[42: 48]
        lm_mouth_outer = lm[48: 60]

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left

        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # shrink
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # crop
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # padding
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # transform
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        return img

    def extract_face(self, quality=95):

        video_capture = cv2.VideoCapture(self.video_path)
        success = True
        i = 0

        while success:
            i = i + 1
            success, frame = video_capture.read()
            if i % self.frame_count > 0:
                continue

            if self.black_width > 0:
                frame = cv2.cvtColor(frame[self.black_width:-self.black_width], cv2.COLOR_RGB2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            dets = self.detector(frame, 1)
            findlm = False
            for k, d in enumerate(dets):
                if d.width() >= 256 and d.height() >= 256:
                    shape = self.predictor(frame, d)
                    findlm = True
                    break
            if findlm == False:
                continue

            t = list(shape.parts())
            a = []
            for tt in t:
                a.append([tt.x, tt.y])
            lm = np.array(a)
            img = Image.fromarray(frame)
            result = self.align_face(img, lm)
            result.save(os.path.join(self.output_path, f'{self.save_name}' + str(self.detected_faces) + '.jpg'),
                        quality=quality)
            self.detected_faces += 1

        video_capture.release()

    def get_stats(self):
        print(f'Detected {self.detected_faces} face images in video {self.video_path}')

    def run(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        print(f'Started looking for face images...')
        self.extract_face()
        print(f'Finished looking for face images...')
        self.get_stats()
