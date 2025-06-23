# --------------------------------------------------------
# Based on:
# https://github.com/lasr-lab/lasr-robot/blob/main/robotenv/deploy/deploy.py
# --------------------------------------------------------
import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import torch

from robotenv.robots.allegro import Allegro
from robotenv.robots.xarm.wrapper import XArmAPI
from robotenv.sensors.digit import Digit


def _obs_allegro(obses):
    obs_index = obses[0:4]
    obs_middle = obses[4:8]
    obs_ring = obses[8:12]
    obs_thumb = obses[12:16]
    print("index:", np.round(obs_index, 3))
    print("middle:", np.round(obs_middle, 3))
    print("ring:", np.round(obs_ring, 3))
    print("thumb:", np.round(obs_thumb, 3))
    print("########################")
    obses = np.concatenate([obs_index, obs_middle, obs_ring, obs_thumb]).astype(
        np.float32
    )
    return obses


class RobotEnvironment(object):
    def __init__(self, cfg):
        # hand setting
        self.allegro = Allegro(hand_topic_prefix="allegroHand_0")
        self.control_freq = cfg.control.control_freq

        # arm setting
        self.arm = XArmAPI(cfg.robots.xarm_ip)

        self.with_digit = True  # Use DIGIT?
        if self.with_digit:
            self.serial_digit_thumb = cfg.robots.digit_thumb
            self.serial_digit_middle = cfg.robots.digit_middle

        self.with_realsense = True  # Use RealSense?
        if self.with_realsense:
            self.depth_alpha = cfg.robots.depth_alpha

    def observation(self):
        # try to set up rospy
        rospy.init_node("robot_obs")

        if self.with_digit:
            digit_thumb = Digit(self.serial_digit_thumb)
            digit_middle = Digit(self.serial_digit_middle)
            digit_thumb.connect()
            digit_middle.connect()

        if self.with_realsense:
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline = rs.pipeline()
            pipeline.start(config)

        while True:
            # Observe xArm positions
            print("cartesian position:", np.round(self.arm.get_position()[1], 3))
            print("joint angles:", np.round(self.arm.get_servo_angle()[1], 3))
            print("------------------------")

            # Observe Allegro Hand positions
            obses, _ = self.allegro.poll_joint_position(wait=True)
            obses = _obs_allegro(obses)
            obses = torch.from_numpy(obses)

            # Observe images from Digits
            if self.with_digit:
                frame_thumb = digit_thumb.get_frame()
                frame_middle = digit_middle.get_frame()
                cv2.imshow("thumb", frame_thumb)
                cv2.imshow("middle", frame_middle)

            # Observe RGB and depth images from RealSense
            if self.with_realsense:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                # translation to OpenCV Format
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=self.depth_alpha),
                    cv2.COLORMAP_JET,
                )
                # Show color and depth images
                cv2.namedWindow("color_image", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("color_image", color_image)
                cv2.namedWindow("depth_image", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("depth_image", depth_colormap)

            # ESC key to exit
            if cv2.waitKey(1) == 27:
                break

        self.allegro.disconnect()
        self.arm.disconnect()
        if self.with_digit:
            digit_thumb.disconnect()
            digit_middle.disconnect()
        cv2.destroyAllWindows()
