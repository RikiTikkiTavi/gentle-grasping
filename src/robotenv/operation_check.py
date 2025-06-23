import pickle
import random

import numpy as np
import pyaudio
import pyrealsense2 as rs
import rospy
import torch
from torchvision import transforms

from models.models import TouchDetectionModel
from robotenv.robots.allegro import Allegro
from robotenv.robots.xarm.wrapper import XArmAPI
from robotenv.sensors.audio import Audio
from robotenv.sensors.digit import Digit


def _obs_allegro(obses):
    obs_index = obses[0:4]
    obs_middle = obses[4:8]
    obs_ring = obses[8:12]
    obs_thumb = obses[12:16]
    obses = np.concatenate([obs_index, obs_middle, obs_ring, obs_thumb]).astype(
        np.float32
    )
    return obses


# for visualiztion
def _kinematics_middle(angles, l1=55, l2=40, l3=35):
    # Calculate fingertip position using trigonometry
    x = (
        l1 * np.cos(angles[0])
        + l2 * np.cos(angles[0] + angles[1])
        + l3 * np.cos(angles[0] + angles[1] + angles[2])
    )
    y = (
        l1 * np.sin(angles[0])
        + l2 * np.sin(angles[0] + angles[1])
        + l3 * np.sin(angles[0] + angles[1] + angles[2])
    )
    return x, y


def _kinematics_thumb(angles, l1=50, l2=50):
    # Calculate fingertip position using trigonometry
    x = l1 * np.cos(angles[0]) + l2 * np.cos(angles[0] + angles[1])
    y = l1 * np.sin(angles[0]) + l2 * np.sin(angles[0] + angles[1])
    return x, y


class OperationCheck(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # hand setting
        self.allegro = Allegro(hand_topic_prefix="allegroHand_0")
        self.control_freq = cfg.control.control_freq

        # random sampling for a_1
        hand_random = cfg.control.hand_random
        hand_random[5] = random.uniform(1.45, 1.75)  # from hand_min to max
        hand_random[6] = random.uniform(0, 0.3)
        hand_random[7] = random.uniform(0, 0.3)
        hand_random[14] = random.uniform(0.2, 0.5)
        hand_random[15] = random.uniform(0, 0.3)

        self.hand_before = torch.from_numpy(np.array(cfg.control.hand_before)).to(
            cfg.device
        )
        self.hand_min = torch.from_numpy(np.array(cfg.control.hand_min)).to(cfg.device)
        self.hand_init = torch.from_numpy(np.array(cfg.control.hand_init)).to(
            cfg.device
        )
        self.hand_random = torch.from_numpy(np.array(hand_random)).to(cfg.device)
        # self.hand_random = torch.from_numpy(np.array(cfg.control.hand_max)).to(
        #     cfg.device
        # )
        self.hand_max = torch.from_numpy(np.array(cfg.control.hand_max)).to(cfg.device)
        self.hand_lower = torch.from_numpy(np.array(cfg.control.hand_lower)).to(
            cfg.device
        )
        self.hand_upper = torch.from_numpy(np.array(cfg.control.hand_upper)).to(
            cfg.device
        )

        # arm setting
        self.arm = XArmAPI(cfg.robots.xarm_ip)
        self.arm.motion_enable(enable=True)
        self.arm.clean_error()
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_tcp_load(
            weight=cfg.robots.xarm_weight,
            center_of_gravity=cfg.robots.xarm_tcp,
            wait=True,
        )
        self.arm.set_tcp_offset(cfg.robots.xarm_tcp_offset)
        self.arm.set_collision_sensitivity(cfg.robots.xarm_sensitivity)
        # self.arm.set_allow_approx_motion(True)
        self.arm.set_allow_approx_motion(True)
        self.xarm_pre = cfg.control.xarm_pre
        self.xarm_before = cfg.control.xarm_before
        self.xarm_init = cfg.control.xarm_init
        self.xarm_grasp = cfg.control.xarm_grasp

        # DIGIT setting
        self.serial_digit_thumb = cfg.robots.digit_thumb
        self.serial_digit_middle = cfg.robots.digit_middle

        # RealSense setting
        self.depth_alpha = cfg.robots.depth_alpha

        # audio setting
        self.sound_threshold = cfg.robots.sound_threshold
        self.format = pyaudio.paInt16
        self.channels = cfg.robots.sound_channels
        self.sample_rate = cfg.robots.sound_sample_rate
        self.chunk_size = cfg.robots.sound_chunk_size

    def operation(self):
        # try to set up rospy
        rospy.init_node("robot_data_collection")
        rospy.sleep(0.5)  # Wait for connections
        hz = self.control_freq
        ros_rate = rospy.Rate(hz)

        # set the poses of the hand
        hand_before = torch.clip(self.hand_before, self.hand_lower, self.hand_upper)
        hand_min = torch.clip(self.hand_min, self.hand_lower, self.hand_upper)
        hand_init = torch.clip(self.hand_init, self.hand_lower, self.hand_upper)
        hand_random = torch.clip(self.hand_random, self.hand_lower, self.hand_upper)

        # set up DIGIT
        digit_thumb = Digit(self.serial_digit_thumb)
        digit_middle = Digit(self.serial_digit_middle)
        digit_thumb.connect()
        digit_middle.connect()

        # set up RealSense
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        pipeline = rs.pipeline()
        pipeline.start(config)

        # move the arm to initial position
        self.arm.set_servo_angle(angle=self.xarm_before, speed=25, wait=True)

        # s_0: before touch
        for t in range(hz * 2):
            if t == hz * 0.1:
                self.allegro.command_joint_position(hand_before)
            ros_rate.sleep()

        # a_0: initial grasp
        action0_hand = hand_init
        [grasp_x, grasp_y, grasp_z, grasp_rot] = [525, 0, 5, 0]  # set grasp position
        print("grasp [x, y, z]: ", [grasp_x, grasp_y, grasp_z])
        print("grasp rotation: ", grasp_rot)
        self.arm.set_position(yaw=grasp_rot, relative=True, speed=50, wait=True)
        self.arm.set_position(x=grasp_x, y=grasp_y, z=300, speed=50, wait=True)
        self.arm.set_position(
            z=grasp_z + self.cfg.control.offset_z, speed=50, wait=True
        )

        # s_1: initial grasp
        for t in range(hz * 4):
            if t > hz * 0.5:
                self.allegro.command_joint_position(hand_min)
            if t > hz * 1.5:
                self.allegro.command_joint_position(action0_hand)
            if t > hz * 3:
                self.allegro.command_joint_position(hand_min)
            ros_rate.sleep()

        # a_1: regrasping (to be optimized)
        action1_hand = hand_random  # to be optimized
        self.allegro.command_joint_position(hand_before)
        self.arm.set_position(z=100, speed=50, relative=True, wait=True)

        [regrasp_x, regrasp_y, regrasp_z, regrasp_rot] = [
            -10,
            -10,
            0,
            -15,
        ]  # set regrasp position
        regrasp_pose = [regrasp_x, regrasp_y, regrasp_z, regrasp_rot]
        print("regrasp [x, y, z]:", regrasp_pose[:3])
        print("regrasp rotation:", regrasp_pose[3])
        self.arm.set_position(yaw=regrasp_rot, relative=True, speed=50, wait=True)
        self.arm.set_position(
            x=regrasp_x, y=regrasp_y, relative=True, speed=50, wait=True
        )
        self.arm.set_position(
            z=grasp_z + regrasp_z + self.cfg.control.offset_z, speed=50, wait=True
        )

        # observe the actual hand poses (before regrasping)
        hand_obs_before, _ = self.allegro.poll_joint_position(wait=True)
        hand_obs_before = _obs_allegro(hand_obs_before)
        middletip0_obs = _kinematics_middle(hand_obs_before[5:8])
        thumbtip0_obs = _kinematics_thumb(hand_obs_before[14:])

        # s_2: regrasping (gentleness labeling)
        self.arm.set_pause_time(0.5, wait=True)

        # set up audio
        audio = Audio()
        frames = []
        duration = 3
        record_loop_count = audio.record_loop_count(duration)
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        # gentleness labeling
        label_gentle = 1
        print("start record", 0)
        for i in range(int(record_loop_count)):
            if i >= int(record_loop_count / 4):
                if i == int(record_loop_count / 4):
                    print("regrasp", i)
                self.allegro.command_joint_position(action1_hand)
            data = stream.read(self.chunk_size)
            frames.append(data)
            label_temp = audio.detect_sound(self.sound_threshold)
            if i > int(record_loop_count / 10):
                if label_temp:
                    label_gentle = 0
                    print("make sounds", i)
        print("stop record", i)
        audio.disconnect()
        stream.stop_stream()
        stream.close()
        p.terminate()

        # a_2: lift (fixed)
        self.arm.set_position(z=100, speed=50, relative=True, wait=True)

        # s_3: in the air (stability labeling)
        model = TouchDetectionModel()
        checkpoint = torch.load(
            "../../../models/trained_models/touchwalrus_20240713113417.pth"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.cfg.device).eval()
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        touch_count = 0
        all_count = 0
        for t in range(hz * 3):
            frame_thumb = digit_thumb.get_frame()
            if t > hz * 2:
                transformed_image = transform(frame_thumb)
                transformed_image = transformed_image.unsqueeze(0)
                transformed_image = transformed_image.to("cuda")
                outputs = model(transformed_image)
                all_count += 1
                if not (outputs.argmax(-1)[0] == 0):
                    print("Touch")
                    touch_count += 1
            ros_rate.sleep()
        print("[touch_count, all_count]:", [touch_count, all_count])
        if touch_count / all_count > self.cfg.robots.touch_detection_threshold:
            label_success = 1
        else:
            label_success = 0

        # calculate the finger displacements of the inputs
        middletip0 = _kinematics_middle(hand_before[5:8].cpu())
        thumbtip0 = _kinematics_thumb(hand_before[14:].cpu())
        middletip1 = _kinematics_middle(action1_hand[5:8].cpu())
        thumbtip1 = _kinematics_thumb(action1_hand[14:].cpu())
        disp_middle = round(abs(middletip1[0] - middletip0[0]).item(), 2)
        disp_thumb = round(abs(thumbtip1[1] - thumbtip0[1]).item(), 2)
        print("Inputs [disp_middle, disp_thumb]:", [disp_middle, disp_thumb])
        print("Total Disp:", disp_middle + disp_thumb)

        # observe the actual hand poses (after regrasping)
        hand_obs_after, _ = self.allegro.poll_joint_position(wait=True)
        hand_obs_after = _obs_allegro(hand_obs_after)
        middletip1_obs = _kinematics_middle(hand_obs_after[5:8])
        thumbtip1_obs = _kinematics_thumb(hand_obs_after[14:])

        # show the actural fingertip displacements
        disp_middle_obs = round(abs(middletip1_obs[0] - middletip0_obs[0]), 2)
        disp_thumb_obs = round(abs(thumbtip1_obs[1] - thumbtip0_obs[1]), 2)
        print("Obs [disp_middle, disp_thumb]: ", [disp_middle_obs, disp_thumb_obs])

        # # consider the fingertip displacement for the success label
        # if disp_thumb_obs > self.cfg.control.threshold_disp_thumb:
        #     print("displacement over threshold")
        #     label_success = 0

        # consider the fingertip displacement for the success label
        disp_regr_path = "../../../models/trained_models/disp_regr_model.pkl"
        with open(disp_regr_path, "rb") as f:
            model = pickle.load(f)
        label_success = model.predict(np.array([[disp_middle_obs, disp_thumb_obs]]))[0]
        print("label_success (by disp):", label_success)

        # a_3: put down (randomized point if success)
        if label_success:
            self.arm.set_position(z=-100, speed=50, relative=True, wait=True)
        self.arm.set_pause_time(0.5, wait=True)
        self.allegro.disconnect()

        # save stability and gentleness labels
        print("label [success, gentle]:\n", [int(label_success), int(label_gentle)])

        # return the arm to the initial position
        self.arm.set_pause_time(0.5, wait=True)
        self.arm.set_position(z=500, speed=50, wait=True)
        self.arm.set_servo_angle(angle=self.xarm_pre, speed=25, wait=True)
        self.arm.disconnect()
