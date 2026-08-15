from __future__ import annotations

import time
import pickle
import socket
import struct
import traceback
from pathlib import Path

import numpy as np

from robolab.eval.base_client import InferenceClient

################################ CLIENT ############################################

_HEADER = struct.Struct("!Q")

def _send_message(sock: socket.socket, payload: dict) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)) + data)

def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("VALPA policy socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def _recv_message(sock: socket.socket) -> dict:
    size = _HEADER.unpack(_recv_exact(sock, _HEADER.size))[0]
    return pickle.loads(_recv_exact(sock, size))

class VALPADroidEEClient(InferenceClient):
    """Local RoboLab inference client for the VALPA world model on DroidIK envs."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
    ) -> None:
        super().__init__()
        self.remote_host = remote_host
        self.remote_port = int(remote_port)
        self.sock = self._connect()

    def _connect(self) -> socket.socket:
        print(f"[{self.__class__.__name__}] Waiting for VALPA server on {self.remote_host}:{self.remote_port}...")
        while True:
            try:
                sock = socket.create_connection((self.remote_host, self.remote_port), timeout=10)
                sock.settimeout(None)
                print(f"[{self.__class__.__name__}] Connected to VALPA server.")
                return sock
            except OSError:
                time.sleep(2)

    def _request(self, payload: dict) -> dict:
        _send_message(self.sock, payload)
        response = _recv_message(self.sock)
        if error := response.get("error"):
            raise RuntimeError(f"VALPA server error:\n{error}")
        return response

    def reset(self, *, env_id: int | None = None):
        if env_id is None:
            self._request({"method": "reset"})
        super().reset(env_id=env_id)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def set_goal_images(
        self,
        external_image,
        wrist_image,
        *,
        env_id: int = 0,
        instruction: str = "goal",
        run_idx: int | None = None,
    ):
        payload = {
            "method": "set_goal_images",
            "external_image": external_image,
            "wrist_image": wrist_image,
            "env_id": env_id,
            "instruction": instruction,
        }
        if run_idx is not None:
            payload["run_idx"] = int(run_idx)
        self._request(payload)

    def _extract_observation(self, obs_dict: dict, *, env_id: int) -> dict:
        from scipy.spatial.transform import Rotation

        external_image = obs_dict["image_obs"]["over_shoulder_right_camera"][env_id].clone().detach().cpu()
        wrist_image = obs_dict["image_obs"]["wrist_cam"][env_id].clone().detach().cpu()

        robot_state = obs_dict["proprio_obs"]
        # EE position
        ee_pos = robot_state["ee_pos"][env_id].clone().detach().cpu().numpy()
        # EE orientation
        ee_quat = robot_state["ee_quat"][env_id].clone().detach().cpu().numpy()
        ee_rpy = Rotation.from_quat(ee_quat[[1, 2, 3, 0]]).as_euler("xyz", degrees=False)
        # gripper state
        gripper_pos = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy()
        ee_pose = np.concatenate([ee_pos, ee_rpy, gripper_pos], axis=0).astype(np.float32)

        return {
            "external_image": external_image,
            "wrist_image": wrist_image,
            "ee_pose": ee_pose,
            "env_id": env_id,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "method": "infer",
            "obs": {
                "external_image": extracted_obs["external_image"],
                "wrist_image": extracted_obs["wrist_image"],
                "ee_pose": extracted_obs["ee_pose"],
            },
            "instruction": instruction,
            "env_id": extracted_obs["env_id"],
        }

    def _query_server(self, request: dict) -> dict:
        return self._request(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        action = response.get("action")
        if action is None:
            action = response.get("actions")
        if action is None:
            raise KeyError("VALPA response did not contain 'action' or 'actions'.")

        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        return action
    
    def report_ground_truth(self, obs_dict: dict, *, env_id: int = 0) -> dict:
        """Match the post-step simulator observation to the pending imagination."""
        extracted = self._extract_observation(obs_dict, env_id=env_id)
        return self._request({
            "method": "report_ground_truth",
            "obs": {
                "external_image": extracted["external_image"],
                "wrist_image": extracted["wrist_image"],
                "ee_pose": extracted["ee_pose"],
            },
            "env_id": env_id,
        })

    def finish_rollout(self, *, env_id: int = 0) -> dict:
        """Finalize a matched imagination/ground-truth analysis rollout."""
        return self._request({
            "method": "finish_rollout",
            "env_id": env_id,
        })

    def metadata(self):
        return self._request({"method": "metadata"})

MyPolicyClient = VALPADroidEEClient

