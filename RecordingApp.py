# import the necessary packages
from __future__ import print_function

from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import cv2

import pyrealsense2 as rs
import numpy as np
import imageio


class RecordingApp:
    def __init__(self):
        self.thread = None
        self.stopEvent = None

        # initialize the root window and image panels
        self.root = tki.Tk()
        self.panel_color = None
        self.panel_depth = None
        self.panel_infrared = None

        self.first = True
        self.recordingState = False

        self.colorVideo = cv2.VideoWriter()
        self.depthVideo = cv2.VideoWriter()
        self.binDepthVideo = cv2.VideoWriter()
        self.infraredVideo = cv2.VideoWriter()

        # Create label
        self.lLabel = tki.Label(self.root, text="left_color_rsD415", width=20, height=1)
        self.mLabel = tki.Label(self.root, text="mid_depth_rsD415", width=20, height=1)
        self.rLabel = tki.Label(self.root, text="right_infrared_rsD415", width=20, height=1)
        self.lLabel.grid(row=0, column=0)
        self.mLabel.grid(row=0, column=1)
        self.rLabel.grid(row=0, column=2)

        # create a button, that when pressed, will take the current frame and save it to file
        btn_snapshot = tki.Button(self.root, text="Snapshot", width=25, command=self.takeSnapshot)
        btn_snapshot.grid(row=2, column=0)

        # Button that lets the user record video
        btn_start_record = tki.Button(self.root, text="Start Record", width=25, command=self.start_record)
        btn_start_record.grid(row=2, column=1)
        btn_end_record = tki.Button(self.root, text="End Record", width=25, command=self.end_record)
        btn_end_record.grid(row=2, column=2)

        self.init_pipeline()

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        # self.thread.daemon = True
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("3D camera recorder")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)


    def __del__(self):
        self.pipeline.stop()


    def init_pipeline(self, setRecordBag = False):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 1280, 720
        self.config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

        if setRecordBag == True:
            ts = datetime.datetime.now()
            bagFileName = "{}.bag".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            self.config.enable_record_to_file(bagFileName)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = 1  # 1 meter
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)


    def update_imgs(self):
        self.frames = self.pipeline.wait_for_frames()
        # Align the depth frame to color frame
        self.aligned_frames = self.align.process(self.frames)

        # Get frames
        self.aligned_depth_frame = self.aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        self.color_frame = self.aligned_frames.get_color_frame()
        self.infrared_frame = self.frames.get_infrared_frame()

        self.depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.infrared_image = np.asanyarray(self.infrared_frame.get_data())

        # Render images
        self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)


    def update_panel(self, panel, img, col_num):
        # if the panel is not None, we need to initialize it
        if panel is None:
            panel = tki.Label(image=img)
            panel.image = img
            panel.grid(row=1, column=col_num)

        # otherwise, simply update the panel
        else:
            panel.configure(image=img)
            panel.image = img


    def videoLoop(self):
        try:
            while not self.stopEvent.is_set():
                self.update_imgs()
                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                color_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                color_img = Image.fromarray(color_img)
                color_img = ImageTk.PhotoImage(color_img)

                depth_img = cv2.cvtColor(self.depth_colormap, cv2.COLOR_BGR2RGB)
                depth_img = Image.fromarray(depth_img)
                depth_img = ImageTk.PhotoImage(depth_img)

                # infrared_img = cv2.cvtColor(self.infrared_image, cv2.COLOR_BGR2RGB)
                infrared_img = Image.fromarray(self.infrared_image)
                infrared_img = ImageTk.PhotoImage(infrared_img)

                if self.first == True:
                    self.colorVideo = cv2.VideoWriter()
                    self.depthVideo = cv2.VideoWriter()
                    self.binDepthVideo = cv2.VideoWriter()
                    self.infraredVideo = cv2.VideoWriter()

                else:
                    if self.recordingState == True:
                        self.colorVideo.write(self.color_image)
                        self.depthVideo.write(self.depth_colormap)
                        self.binDepthVideo.write(self.depth_colormap)
                        self.infraredVideo.write(self.infrared_image)

                self.update_panel(self.panel_color, color_img, 0)
                self.update_panel(self.panel_depth, depth_img, 1)
                self.update_panel(self.panel_infrared, infrared_img, 2)

        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

        # finally:
        #     self.pipeline.stop()


    def takeSnapshot(self):
        # grab the current timestamp and use it to construct the
        # output path
        ts = datetime.datetime.now()
        colorImgFileName = "{}{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"), "_color")
        depthImgFileName = "{}{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"), "_depth")
        infraredImgFileName = "{}{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"), "_infrared")

        imageio.imwrite(colorImgFileName, cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR))
        imageio.imwrite(depthImgFileName, cv2.cvtColor(self.depth_colormap, cv2.COLOR_RGB2BGR))
        imageio.imwrite(infraredImgFileName, self.infrared_image)

        print("[INFO] saved {}".format(colorImgFileName))
        print("[INFO] saved {}".format(depthImgFileName))
        print("[INFO] saved {}".format(infraredImgFileName))

    def start_record(self):
        self.init_pipeline(True)
        self.update_imgs()

        self.recordingState = True
        if self.first == True:
            ts = datetime.datetime.now()
            colorVideoFileName = "{}{}.avi".format(ts.strftime("%Y-%m-%d_%H-%M-%S"), "_color")
            depthVideoFileName = "{}{}.avi".format(ts.strftime("%Y-%m-%d_%H-%M-%S"), "_depth")
            infraredVideoFileName = "{}{}.avi".format(ts.strftime("%Y-%m-%d_%H-%M-%S"), "_infrared")

            self.colorVideo = cv2.VideoWriter(colorVideoFileName, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480), 1)
            self.depthVideo = cv2.VideoWriter(depthVideoFileName, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480), 1)
            self.infraredVideo = cv2.VideoWriter(infraredVideoFileName, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480), 1)

            self.first = False

            print("[INFO] start recording {}".format(colorVideoFileName))
            print("[INFO] start recording {}".format(depthVideoFileName))
            print("[INFO] start recording {}".format(infraredVideoFileName))

    def end_record(self):
        if self.recordingState == True:
            self.recordingState = False
            self.first = True

            print("[INFO] end recording")
            ts = datetime.datetime.now()
            videoFileName = "{}{}.avi".format(ts.strftime("%Y-%m-%d_%H-%M-%S"), "_xxx")
            print("[INFO] saved video files {}".format(videoFileName))

            self.pipeline.stop()
            self.init_pipeline()


    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.root.quit()

if __name__ == '__main__':
    ra = RecordingApp("image/", "video/")
    ra.root.mainloop()