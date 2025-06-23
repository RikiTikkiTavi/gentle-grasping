import os
import pickle
import random
import time

import cv2
import numpy as np
import pyaudio
import pyrealsense2 as rs
import rospy
import torch
from torchvision import transforms

from models.data.make_dataset import make_obsset
from models.models import DenseNet, TouchDetectionModel
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


def _kinematics_middle(angles, l1=55, l2=40, l3=35):
    # calculate fingertip position using trigonometry
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
    # calculate fingertip position using trigonometry
    x = l1 * np.cos(angles[0]) + l2 * np.cos(angles[0] + angles[1])
    y = l1 * np.sin(angles[0]) + l2 * np.sin(angles[0] + angles[1])
    return x, y


class OptimalControl(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # hand setting
        self.allegro = Allegro(hand_topic_prefix="allegroHand_0")
        self.control_freq = cfg.control.control_freq
        self.hand_before = torch.from_numpy(np.array(cfg.control.hand_before)).to(
            cfg.device
        )
        self.hand_min = torch.from_numpy(np.array(cfg.control.hand_min)).to(cfg.device)
        self.hand_init = torch.from_numpy(np.array(cfg.control.hand_init)).to(
            cfg.device
        )
        # self.hand_opt = torch.from_numpy(np.array(cfg.control.hand_opt)).to(cfg.device)
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

    def optimal_control(self):
        # path setting for observed data
        dirname = "optctrl_" + time.strftime("%Y_%m_%d_%H%M%S")
        data_dir_path = os.path.abspath("../../../data/optimal_control") + "/" + dirname
        os.mkdir(data_dir_path)

        # try to set up rospy
        rospy.init_node("robot_optimal")
        rospy.sleep(0.5)  # Wait for connections
        hz = self.control_freq
        ros_rate = rospy.Rate(hz)

        # set the poses of the hand
        hand_before = torch.clip(self.hand_before, self.hand_lower, self.hand_upper)
        hand_min = torch.clip(self.hand_min, self.hand_lower, self.hand_upper)
        hand_init = torch.clip(self.hand_init, self.hand_lower, self.hand_upper)
        # hand_opt = torch.clip(self.hand_opt, self.hand_lower, self.hand_upper)

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

        def obs_camera():
            # observe RGB and depth images from RealSense
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=self.depth_alpha),
                cv2.COLORMAP_JET,
            )
            return color_image, depth_image

        def save_obs(color_image, depth_image, frame_thumb, frame_middle, num):
            # store all images to the specified path
            cv2.imwrite(data_dir_path + "/camera_rgb_{}.png".format(num), color_image)
            cv2.imwrite(data_dir_path + "/camera_depth_{}.png".format(num), depth_image)
            cv2.imwrite(data_dir_path + "/touch_thumb_{}.png".format(num), frame_thumb)
            cv2.imwrite(
                data_dir_path + "/touch_middle_{}.png".format(num), frame_middle
            )

        def random_sample(range):
            return round(random.uniform(range[0], range[1]), 2)

        def limit_sample(sample, range):
            return min(max(sample, range[0]), range[1])

        def deg2rad(deg):
            return deg * np.pi / 180

        def get_rotation_matrix(roll, pitch, yaw):
            roll_rad = deg2rad(roll)
            pitch_rad = deg2rad(pitch)
            yaw_rad = deg2rad(yaw)
            R_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)],
                ]
            )
            R_y = np.array(
                [
                    [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
                ]
            )
            R_z = np.array(
                [
                    [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1],
                ]
            )
            R = R_z @ R_y @ R_x
            return R

        def image_to_world(point, intrinsics, camera_pos, roll, pitch, yaw):
            # image to camera
            x, y, depth = point
            X_c = (x - intrinsics.ppx) * depth / intrinsics.fx
            Y_c = (y - intrinsics.ppy) * depth / intrinsics.fy
            Z_c = depth
            camera_coords = np.array([X_c, Y_c, Z_c])
            camera_coords = camera_coords * 1000

            # camera to world
            R = get_rotation_matrix(roll, pitch, yaw)
            world_coords = R @ camera_coords + camera_pos

            return world_coords

        # obtain the object position in world coordinate
        self.arm.set_servo_angle(angle=self.xarm_pre, speed=25, wait=True)
        camera_pos = np.array(self.cfg.control.camera_pos)
        roll, pitch, yaw = self.cfg.control.camera_rot
        background = cv2.imread(os.path.abspath("../../../robotenv/sensors/depth.png"))
        start_time = time.time()
        processed = False
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise Exception("Depth frame not available")
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=self.depth_alpha),
                cv2.COLORMAP_JET,
            )

            # extract region of interest
            src_pts = np.float32(self.cfg.control.roi)
            dst_pts = np.float32(
                [
                    [0, 0],
                    [depth_colormap.shape[1], 0],
                    [depth_colormap.shape[1], depth_colormap.shape[0]],
                    [0, depth_colormap.shape[0]],
                ]
            )
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            depth_colormap_trapezoid = cv2.warpPerspective(
                depth_colormap,
                matrix,
                (depth_colormap.shape[1], depth_colormap.shape[0]),
            )
            background_trapezoid = cv2.warpPerspective(
                background, matrix, (background.shape[1], background.shape[0])
            )
            depth_colormap_roi = depth_colormap_trapezoid[
                0 : depth_colormap.shape[0], 0 : depth_colormap.shape[1]
            ]
            background_roi = background_trapezoid[
                0 : background.shape[0], 0 : background.shape[1]
            ]

            # subtract the background
            diff = cv2.absdiff(background_roi, depth_colormap_roi)

            # grayscale and resize
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            full_diff = np.zeros_like(depth_colormap)
            full_diff = cv2.warpPerspective(
                diff_gray,
                np.linalg.inv(matrix),
                (full_diff.shape[1], full_diff.shape[0]),
            )

            # denoising
            _, binary_mask = cv2.threshold(full_diff, 150, 255, cv2.THRESH_BINARY_INV)
            cleaned_diff = cv2.bitwise_and(full_diff, full_diff, mask=binary_mask)
            kernel = np.ones((5, 5), np.uint8)
            cleaned_diff = cv2.morphologyEx(cleaned_diff, cv2.MORPH_OPEN, kernel)

            # find the max difference point
            _, _, _, max_loc = cv2.minMaxLoc(cleaned_diff)

            if time.time() - start_time > 1.8 and not processed:
                actual_depth = (
                    depth_image[max_loc[1], max_loc[0]] * depth_frame.get_units()
                )
                max_diff_point_3d = [max_loc[0], max_loc[1], actual_depth]

                # get intrinsics parameters
                profile = pipeline.get_active_profile()
                depth_stream = profile.get_stream(rs.stream.depth)
                intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
                # print(f"Intrinsics: fx={intrinsics.fx}, fy={intrinsics.fy}, cx={intrinsics.ppx}, cy={intrinsics.ppy}")

                # transform to world coordinates
                world_coords = image_to_world(
                    max_diff_point_3d, intrinsics, camera_pos, roll, pitch, yaw
                )
                world_coords[1] += self.cfg.control.offset_y  # adjustment
                print("World Coordinates:", world_coords)
                processed = True
            if time.time() - start_time > 2:
                break

        # move the arm to initial position
        self.arm.set_servo_angle(angle=self.xarm_before, speed=25, wait=True)

        # s_0: before touch
        for t in range(hz * 2):
            frame_thumb = digit_thumb.get_frame()
            frame_middle = digit_middle.get_frame()
            color_image, depth_image = obs_camera()
            if t == hz * 0.1:
                self.allegro.command_joint_position(hand_before)
            if t == hz * 1:
                save_obs(color_image, depth_image, frame_thumb, frame_middle, 0)
            ros_rate.sleep()

        # a_0: initial grasp
        action0_hand = hand_init
        object_pos = world_coords
        # clip and randomize the grasping pose
        grasp_x = limit_sample(
            object_pos[0] + random_sample(self.cfg.control.turb_xy),
            self.cfg.control.range_x,
        )
        grasp_y = limit_sample(
            object_pos[1] + random_sample(self.cfg.control.turb_xy),
            self.cfg.control.range_y,
        )
        grasp_z = random_sample(self.cfg.control.range_z)
        grasp_rot = random_sample(self.cfg.control.range_rot)
        print("grasp [x, y, z]: ", [grasp_x, grasp_y, grasp_z])
        print("grasp rotation: ", grasp_rot)
        self.arm.set_position(yaw=grasp_rot, relative=True, speed=50, wait=True)
        self.arm.set_position(x=grasp_x, y=grasp_y, z=300, speed=50, wait=True)
        self.arm.set_position(
            z=grasp_z + self.cfg.control.offset_z, speed=50, wait=True
        )
        action0_arm = self.arm.get_servo_angle()[1]
        np.save(data_dir_path + "/action0_hand", action0_hand.cpu())
        np.save(data_dir_path + "/action0_arm", action0_arm)

        # load the trained model
        model_filename = "Densenet-RgbTouch-20241004145158.pth"
        # model_filename = "Densenet-RbgTouch-shared-20240920225846.pth"
        # model_filename = "Densenet-Rgb-20240920210429.pth"
        model_path = os.path.join("../../../models/trained_models", model_filename)

        model = DenseNet(self.cfg)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.cfg.device).eval()

        # s_1: initial grasp
        for t in range(hz * 3):
            frame_thumb = digit_thumb.get_frame()
            frame_middle = digit_middle.get_frame()
            color_image, depth_image = obs_camera()
            if t > hz * 1.5:
                self.allegro.command_joint_position(action0_hand)
            if t == hz * 2:
                save_obs(color_image, depth_image, frame_thumb, frame_middle, 1)
            ros_rate.sleep()

        # regrasp until satisfying criteria
        num_max_regrasp = self.cfg.control.opt.num_max_regrasp
        for i in range(num_max_regrasp):
            # load observation at s_1
            crop = self.cfg.dataset.crop
            touchdiff = self.cfg.dataset.touchdiff
            vision_imgs, touch_imgs = make_obsset(data_dir_path, crop, touchdiff)
            vision_imgs = [img.to(self.cfg.device) for img in vision_imgs]
            touch_imgs = [img.to(self.cfg.device) for img in touch_imgs]

            # random search
            num_samples = self.cfg.control.opt.num_samples
            num_nonrelpose = self.cfg.control.opt.num_nonrelpose
            # for finger joints
            hand_opt = hand_init.repeat(num_samples, 1)
            hand_opt[:, 5] = torch.FloatTensor(num_samples).uniform_(1.45, 1.75)
            hand_opt[:, 6] = torch.FloatTensor(num_samples).uniform_(0, 0.3)
            hand_opt[:, 7] = torch.FloatTensor(num_samples).uniform_(0, 0.3)
            hand_opt[:, 14] = torch.FloatTensor(num_samples).uniform_(0.2, 0.5)
            hand_opt[:, 15] = torch.FloatTensor(num_samples).uniform_(0, 0.3)
            hand_opt = torch.clip(hand_opt, self.hand_lower, self.hand_upper)
            hand_opt = hand_opt.to(self.cfg.device).float()

            # for regrasping pose
            regrasp_pose_opt = torch.tensor(
                np.array([0, 0, 0, 0]).astype(np.float32)
            ).repeat(num_samples, 1)
            regrasp_pose_opt[num_nonrelpose:, 0] = torch.FloatTensor(
                num_samples - num_nonrelpose
            ).uniform_(self.cfg.control.regrasp_xy[0], self.cfg.control.regrasp_xy[1])
            regrasp_pose_opt[num_nonrelpose:, 1] = torch.FloatTensor(
                num_samples - num_nonrelpose
            ).uniform_(self.cfg.control.regrasp_xy[0], self.cfg.control.regrasp_xy[1])
            regrasp_pose_opt[num_nonrelpose:, 2] = torch.FloatTensor(
                num_samples - num_nonrelpose
            ).uniform_(self.cfg.control.regrasp_z[0], self.cfg.control.regrasp_z[1])
            regrasp_pose_opt[num_nonrelpose:, 3] = torch.FloatTensor(
                num_samples - num_nonrelpose
            ).uniform_(self.cfg.control.regrasp_rot[0], self.cfg.control.regrasp_rot[1])
            # clip z values
            grasp_z_tensor = torch.tensor(grasp_z)
            min_z = self.cfg.control.range_z[0] - grasp_z_tensor
            max_z = self.cfg.control.range_z[1] - grasp_z_tensor
            regrasp_pose_opt[:, 2] = regrasp_pose_opt[:, 2].clamp(min_z, max_z)

            actions = hand_opt, regrasp_pose_opt
            actions = [act.to(self.cfg.device) for act in actions]
            torch.cuda.synchronize()

            with torch.no_grad():
                outputs = model(vision_imgs, touch_imgs, actions)

            outputs = outputs.cpu().numpy()
            actions[0] = actions[0].cpu().numpy()  # hand
            actions[1] = actions[1].cpu().tolist()  # regrasp pose

            # criteria for best actions
            valid_indices = (outputs[:, 0] > 0.95) & (outputs[:, 1] > 0.95)
            print("optimal regrasp?:", valid_indices.any())

            # select the best action
            if valid_indices.any():
                best_idx_opt = np.argmax(outputs[valid_indices, 0])
                best_idx_opt = np.where(valid_indices)[0][best_idx_opt]
                action1_hand = actions[0][best_idx_opt]
                regrasp_pose = actions[1][best_idx_opt]
                outputs_opt = outputs[best_idx_opt]
                print()
                print("outputs_opt:\n", outputs_opt, end="\n")
                print("hand_opt (middle):\n", action1_hand[5:8], end="\n")
                print("hand_opt (thumb):\n", action1_hand[14:], end="\n")
                print("regrasp [x, y, z]:", regrasp_pose[:3])
                print("regrasp rotation:", regrasp_pose[3])

                self.allegro.command_joint_position(hand_before)
                if not best_idx_opt < num_nonrelpose:
                    self.arm.set_position(z=100, speed=50, relative=True, wait=True)
                    self.arm.set_position(
                        yaw=regrasp_pose[3], relative=True, speed=50, wait=True
                    )
                    self.arm.set_position(
                        x=regrasp_pose[0],
                        y=regrasp_pose[1],
                        relative=True,
                        speed=50,
                        wait=True,
                    )
                    self.arm.set_position(
                        z=grasp_z + regrasp_pose[2] + self.cfg.control.offset_z,
                        speed=50,
                        wait=True,
                    )
                break
            # choose the sub-optimal action considering the model outputs with relpose
            else:
                best_idx_subopt = np.argmax(outputs[num_nonrelpose:, 0])
                best_idx_subopt = best_idx_subopt + num_nonrelpose
                outputs_opt = outputs[best_idx_subopt]
                regrasp_pose = actions[1][best_idx_subopt]
                print("outputs_opt:\n", outputs_opt, end="\n")
                print("regrasp [x, y, z]:", regrasp_pose[:3])
                print("regrasp rotation:", regrasp_pose[3])

                self.allegro.command_joint_position(hand_before)
                self.arm.set_position(z=100, speed=50, relative=True, wait=True)
                self.arm.set_position(
                    yaw=regrasp_pose[3], relative=True, speed=50, wait=True
                )
                self.arm.set_position(
                    x=regrasp_pose[0],
                    y=regrasp_pose[1],
                    relative=True,
                    speed=50,
                    wait=True,
                )
                self.arm.set_position(
                    z=grasp_z + regrasp_pose[2] + self.cfg.control.offset_z,
                    speed=50,
                    wait=True,
                )
                if i == num_max_regrasp - 1:
                    action1_hand = actions[0][best_idx_subopt]
                    print("hand_opt (middle):\n", action1_hand[5:8], end="\n")
                    print("hand_opt (thumb):\n", action1_hand[14:], end="\n")
                    print("[Causion]: execute the sub-optimal action")
                    break
                else:
                    for t in range(hz * 3):
                        frame_thumb = digit_thumb.get_frame()
                        frame_middle = digit_middle.get_frame()
                        color_image, depth_image = obs_camera()
                        if t > hz * 1.5:
                            self.allegro.command_joint_position(action0_hand)
                        if t == hz * 2:
                            save_obs(
                                color_image, depth_image, frame_thumb, frame_middle, 1
                            )
                        ros_rate.sleep()

        action1_arm = self.arm.get_servo_angle()[1]
        np.save(data_dir_path + "/action1_hand", action1_hand)
        np.save(data_dir_path + "/action1_arm", action1_arm)
        np.save(data_dir_path + "/action1_regrasp_pose", regrasp_pose)

        # # show delta of fingers
        # action_delta_hand = action1_hand - hand_min
        # print()
        # print("Delta (middle):\n", action_delta_hand[5:8], end="\n")
        # print("Delta (thumb):\n", action_delta_hand[14:], end="\n")

        # observe the actual hand poses (before regrasping)
        hand_obs_before, _ = self.allegro.poll_joint_position(wait=True)
        hand_obs_before = _obs_allegro(hand_obs_before)
        middletip0_obs = _kinematics_middle(hand_obs_before[5:8])
        thumbtip0_obs = _kinematics_thumb(hand_obs_before[14:])
        np.save(data_dir_path + "/hand_obs_before", hand_obs_before)

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
            frame_thumb = digit_thumb.get_frame()
            frame_middle = digit_middle.get_frame()
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
            if i == int(record_loop_count / 2):
                color_image, depth_image = obs_camera()
                save_obs(color_image, depth_image, frame_thumb, frame_middle, 2)
        print("stop record", i)
        audio.save_record(data_dir_path + "/record_s2.wav", frames)  # save record
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
            frame_middle = digit_middle.get_frame()
            color_image, depth_image = obs_camera()
            if t > hz * 2:
                transformed_image = transform(frame_thumb)
                transformed_image = transformed_image.unsqueeze(0)
                transformed_image = transformed_image.to("cuda")
                outputs = model(transformed_image)
                all_count += 1
                if not (outputs.argmax(-1)[0] == 0):
                    print("Touch")
                    touch_count += 1
                save_obs(color_image, depth_image, frame_thumb, frame_middle, 3)
            ros_rate.sleep()
        print("[touch_count, all_count]:", [touch_count, all_count])
        if touch_count / all_count > self.cfg.robots.touch_detection_threshold:
            label_success = 1
        else:
            label_success = 0

        # calculate the finger displacements of the inputs
        middletip0 = _kinematics_middle(hand_before[5:8].cpu())
        thumbtip0 = _kinematics_thumb(hand_before[14:].cpu())
        middletip1 = _kinematics_middle(action1_hand[5:8])
        thumbtip1 = _kinematics_thumb(action1_hand[14:])
        disp_middle = round(abs(middletip1[0] - middletip0[0]).item(), 2)
        disp_thumb = round(abs(thumbtip1[1] - thumbtip0[1]).item(), 2)
        print("Inputs [disp_middle, disp_thumb]:", [disp_middle, disp_thumb])
        print("Total Disp:", disp_middle + disp_thumb)

        # observe the actual hand poses (after regrasping)
        hand_obs_after, _ = self.allegro.poll_joint_position(wait=True)
        hand_obs_after = _obs_allegro(hand_obs_after)
        middletip1_obs = _kinematics_middle(hand_obs_after[5:8])
        thumbtip1_obs = _kinematics_thumb(hand_obs_after[14:])
        np.save(data_dir_path + "/hand_obs_after", hand_obs_after)

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
        print("label_success (from disps):", label_success)

        # a_3: put down (randomized point if success)
        if label_success:
            # randomize the position to put down
            back_x = random_sample(self.cfg.control.range_x)
            back_y = random_sample(self.cfg.control.range_y)
            print("next position [x, y]:", [back_x, back_y])
            back_rot = round(random.uniform(-5, 5), 2)
            self.arm.set_position(yaw=back_rot, relative=True, speed=50, wait=True)
            self.arm.set_position(x=back_x, y=back_y, speed=50, wait=True)
            self.arm.set_position(z=-100, speed=50, relative=True, wait=True)
        self.arm.set_pause_time(0.5, wait=True)
        self.allegro.disconnect()

        # save stability and gentleness labels
        np.save(data_dir_path + "/labels", [int(label_success), int(label_gentle)])
        print("label [success, gentle]:\n", np.load(data_dir_path + "/labels.npy"))

        # return the arm to the initial position
        self.arm.set_pause_time(0.5, wait=True)
        self.arm.set_position(z=500, speed=50, wait=True)
        self.arm.set_servo_angle(angle=self.xarm_pre, speed=25, wait=True)
        self.arm.disconnect()
