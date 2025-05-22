# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
from pathlib import Path
import time
import logging

import cv2
import numpy as np
import torch
import zmq

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import So1002RobotConfig
from lerobot.common.robot_devices.robots.feetech_calibration import run_arm_manual_calibration
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError

def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    if not torch.allclose(goal_pos, safe_goal_pos):
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"  requested relative goal position target: {diff}\n"
            f"    clamped relative goal position target: {safe_diff}"
        )

    return safe_goal_pos

class MobileManipulator:
    """
    MobileManipulator is a class for connecting to and controlling a remote mobile manipulator robot.
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    def __init__(self, config: So1002RobotConfig):
        """
        Expected keys in config:
          - ip, port, video_port for the remote connection.
          - calibration_dir, leader_arms, follower_arms, max_relative_target, etc.
        """
        self.robot_type = config.type
        self.config = config
        self.remote_ip = config.ip
        self.remote_port = config.port
        self.remote_port_video = config.video_port
        self.calibration_dir = Path(self.config.calibration_dir)
        self.logs = {}

        # For teleoperation, the leader arm (local) is used to record the desired arm pose.
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)

        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)

        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.is_connected = False

        self.last_frames = {}
        self.last_remote_arm_state = torch.zeros(6, dtype=torch.float32)

        # ZeroMQ context and sockets.
        self.context = None
        self.cmd_socket = None
        self.video_socket = None

    def get_motor_names(self, arms: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arms.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        follower_arm_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(follower_arm_names),),
                "names": follower_arm_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(follower_arm_names),),
                "names": follower_arm_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available = []
        for name in self.leader_arms:
            available.append(get_arm_id(name, "leader"))
        for name in self.follower_arms:
            available.append(get_arm_id(name, "follower"))
        return available

    def connect(self):

        # Set up ZeroMQ sockets to communicate with the remote mobile robot.
        self.context = zmq.Context()
        self.cmd_socket = self.context.socket(zmq.PUSH)
        connection_string = f"tcp://{self.remote_ip}:{self.remote_port}"
        self.cmd_socket.connect(connection_string)
        self.cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.video_socket = self.context.socket(zmq.PULL)
        video_connection = f"tcp://{self.remote_ip}:{self.remote_port_video}"
        self.video_socket.connect(video_connection)
        self.video_socket.setsockopt(zmq.CONFLATE, 1)
        print(
            f"[INFO] Connected to remote robot at {connection_string} and video stream at {video_connection}."
        )
        self.is_connected = True

    def load_calibration(self, name, arm_type):
        arm_id = get_arm_id(name, arm_type)
        arm_calib_path = self.calibration_dir / f"{arm_id}.json"

        if not arm_calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {arm_calib_path}")

        with open(arm_calib_path) as f:
            calibration = json.load(f)

        return calibration

#################### 추후 calibration도 가능하게 만들 예정 ##########################

    # def load_or_run_calibration_(self, name, arm, arm_type):
    #     arm_id = get_arm_id(name, arm_type)
    #     arm_calib_path = self.calibration_dir / f"{arm_id}.json"

    #     if arm_calib_path.exists():
    #         with open(arm_calib_path) as f:
    #             calibration = json.load(f)
    #     else:
    #         print(f"Missing calibration file '{arm_calib_path}'")
    #         calibration = run_arm_manual_calibration(arm, self.robot_type, name, arm_type)
    #         print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
    #         arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
    #         with open(arm_calib_path, "w") as f:
    #             json.dump(calibration, f)

    #     return calibration

    # def calibrate_leader(self):
    #     for name, arm in self.leader_arms.items():
    #         # Connect the bus
    #         arm.connect()

    #         # Disable torque on all motors
    #         for motor_id in arm.motors:
    #             arm.write("Torque_Enable", TorqueMode.DISABLED.value, motor_id)

    #         # Now run calibration
    #         calibration = self.load_or_run_calibration_(name, arm, "leader")
    #         arm.set_calibration(calibration)

    # def calibrate_follower(self):
    #     for name, bus in self.follower_arms.items():
    #         bus.connect()

    #         # Disable torque on all motors
    #         for motor_id in bus.motors:
    #             bus.write("Torque_Enable", 0, motor_id)

    #         # Then filter out wheels
    #         arm_only_dict = {k: v for k, v in bus.motors.items() if not k.startswith("wheel_")}
    #         if not arm_only_dict:
    #             continue

    #         original_motors = bus.motors
    #         bus.motors = arm_only_dict

    #         calibration = self.load_or_run_calibration_(name, bus, "follower")
    #         bus.set_calibration(calibration)

    #         bus.motors = original_motors

    def _get_data(self):
        """
        Polls the video socket for up to 15 ms. If data arrives, decode only
        the *latest* message, returning frames, speed, and arm state. If
        nothing arrives for any field, use the last known values.
        """
        frames = {}
        remote_arm_state_tensor = torch.zeros(6, dtype=torch.float32)

        # Poll up to 15 ms
        poller = zmq.Poller()
        poller.register(self.video_socket, zmq.POLLIN)
        socks = dict(poller.poll(15))
        if self.video_socket not in socks or socks[self.video_socket] != zmq.POLLIN:
            # No new data arrived → reuse ALL old data
            return (self.last_frames, self.last_remote_arm_state)

        # Drain all messages, keep only the last
        last_msg = None
        while True:
            try:
                obs_string = self.video_socket.recv_string(zmq.NOBLOCK)
                last_msg = obs_string
            except zmq.Again:
                break

        if not last_msg:
            # No new message → also reuse old
            return (self.last_frames, self.last_present_speed, self.last_remote_arm_state)

        # Decode only the final message
        try:
            observation = json.loads(last_msg)
            # print(f"[DEBUG] Received observation: {observation}") #####################################
            images_dict = observation.get("images", {})
            new_arm_state = observation.get("follower_arm_state", None)

            # Convert images
            for cam_name, image_b64 in images_dict.items():
                if image_b64:
                    jpg_data = base64.b64decode(image_b64)
                    np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
                    frame_candidate = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame_candidate is not None:
                        frames[cam_name] = frame_candidate

            # If remote_arm_state is None and frames is None there is no message then use the previous message
            if new_arm_state is not None and frames is not None:
                self.last_frames = frames

                remote_arm_state_tensor = torch.tensor(new_arm_state, dtype=torch.float32)
                self.last_remote_arm_state = remote_arm_state_tensor

            else:
                frames = self.last_frames

                remote_arm_state_tensor = self.last_remote_arm_state

            # _get_data 안에서
            # print(f"[Client] Received remote_arm_state: {remote_arm_state_tensor},frames: {frames}") #########################

        except Exception as e:
            print(f"[DEBUG] Error decoding video message: {e}")
            # If decode fails, fall back to old data
            return (self.last_frames, self.last_remote_arm_state)

        return frames, remote_arm_state_tensor
    
    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("MobileManipulator is not connected. Run `connect()` first.")

        # Read position of leader arm
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Send goal to follower arm
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]

            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            follower_goal_pos[name] = goal_pos
            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        if not record_data:
            return

        # Read follower state
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        state = torch.cat([follower_pos[name] for name in self.follower_arms])
        action = torch.cat([follower_goal_pos[name] for name in self.follower_arms])

        # Read images
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict = {"observation.state": state}
        action_dict = {"action": action}
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self) -> dict:
        """
        Capture observations from the remote robot: follower arm positions and camera frames.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")

        frames, remote_arm_state_tensor = self._get_data()

        obs_dict = {"observation.state": remote_arm_state_tensor}

        # Loop over each configured camera
        for cam_name, cam in self.cameras.items():
            frame = frames.get(cam_name, None)
            if frame is None:
                # Create a black image using the camera's configured width, height, and channels
                frame = np.zeros((cam.height, cam.width, cam.channels), dtype=np.uint8)
            obs_dict[f"observation.images.{cam_name}"] = torch.from_numpy(frame)

        return obs_dict


    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")

        # Ensure the action tensor has at least 9 elements:
        #   - First 6: arm positions.
        #   - Last 3: base commands.
        if action.numel() < 6:
            raise ValueError("Action tensor must have at least 6 elements for arm positions.")

        # Extract arm and base actions.
        arm_actions = action[:6].flatten()

        arm_positions_list = arm_actions.tolist()

        # send_action 안에서
        # print(f"[Client] Sending arm_positions: {arm_positions_list}") ################################

        message = {"arm_positions": arm_positions_list}
        # print(f"[Client] Sending message: {message}") #########################
        self.cmd_socket.send_string(json.dumps(message))

        return action

    def print_logs(self):
        pass

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected.")
        if self.cmd_socket:
            stop_cmd = {"arm_positions": {}}
            self.cmd_socket.send_string(json.dumps(stop_cmd))
            self.cmd_socket.close()
        if self.video_socket:
            self.video_socket.close()
        if self.context:
            self.context.term()
        self.is_connected = False
        print("[INFO] Disconnected from remote robot.")


    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

