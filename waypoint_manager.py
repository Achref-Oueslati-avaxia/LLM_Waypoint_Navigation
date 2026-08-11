#!/usr/bin/env python3
"""
Waypoint Manager (Asynchronous & Update-Capable)
Section: System Architecture
"""

import ast
import json
import math
import os
import threading
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional
import traceback
import sys
import select

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from nav2_msgs.action import NavigateToPose
from ament_index_python.packages import get_package_share_directory
import tf2_ros


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

WORKSPACE = "/jazzy_ws"

WAYPOINTS_FILE = os.path.join(
    WORKSPACE,
    "src",
    "outlander_navigation",
    "config",
    "waypoints.json",
)

MANUAL_STATUS_TOPIC = '/navigate_to_pose/_action/status'
CMD_VEL_TOPIC = '/outlander_controller/cmd_vel'
SCAN_TOPIC = '/scan'
MAP_FRAME = 'map'
BASE_FRAME = 'base_footprint'

STRAIGHT_LINEAR_SPEED = 0.40
STRAIGHT_ANGULAR_SPEED = 0.4
POSITION_TOLERANCE = 0.10
HEADING_TOLERANCE = 0.08
DECEL_START_DISTANCE = 1.2
MIN_SPEED = 0.08
MAX_ANGLE_FOR_CORRECTION_DEG = 45.0
ANGLE_DEADBAND_DEG = 2.0

OBSTACLE_STOP_DISTANCE = 0.65
OBSTACLE_CONE_ANGLE_DEG = 40.0
MIN_VALID_SCAN_DISTANCE = 0.15

FILE_CHECK_INTERVAL = 1.0
MAIN_LOOP_HZ = 1.0
CONTROL_LOOP_HZ = 20.0


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def parse_priority(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return int(value) == 1
    s = str(value).strip().lower()
    return s in ('1', 'true', 'yes', 'y', 't')


def numeric_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    s = str(value).strip()
    return s if s.isdigit() else None


def yaw_to_quat_z_w(yaw: float):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ──────────────────────────────────────────────────────────────────────────────
# NODE
# ──────────────────────────────────────────────────────────────────────────────

class WaypointManager(Node):
    def __init__(self, waypoint_file=WAYPOINTS_FILE):
        super().__init__('waypoint_manager_final')

        self.waypoint_file = str(Path(waypoint_file).expanduser())
        self.waypoints = deque()
        self.seen_ids = set()
        self.completed_ids = set()

        # Runtime Flags & Mission States
        self.current_waypoint = None
        self.current_reverse = False

        # Used ONLY by STOP / RESUME
        self.paused_waypoint = None

        # Used ONLY by priority interruption (ENTER to continue)
        self.interrupted_waypoint = None
        self.manual_interrupted_waypoint = None
        self.waiting_for_resume = False
        self.resume_event = threading.Event()
        self.is_paused = False
        self.manual_active = False
        self.script_navigation_active = False
        self.canceling_script_goal = False

        # Straight-Line Controller State Machine
        self.straight_active = False
        self.straight_state = 'IDLE'
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_yaw_final = 0.0

        self.last_scan = None
        self.last_mtime = 0.0

        # Nav2 Action Client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.busy_pub = self.create_publisher(
            Bool,
            "/robot/mission_busy",
            10,
        )

        self._publish_busy(False)
        self.nav_goal_handle = None

        self.declare_parameter("cmd_vel_frame_id", "base_footprint")
        self.cmd_vel_frame_id = self.get_parameter("cmd_vel_frame_id").value

        # Publishers & Subscriptions
        self.cmd_pub = self.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)
        self.scan_sub = self.create_subscription(LaserScan, SCAN_TOPIC, self._scan_cb, 10)
        self.status_sub = self.create_subscription(GoalStatusArray, MANUAL_STATUS_TOPIC, self._status_cb, 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timers
        self.create_timer(FILE_CHECK_INTERVAL, self._file_timer)
        self.create_timer(1.0 / MAIN_LOOP_HZ, self._main_loop)
        self.create_timer(1.0 / CONTROL_LOOP_HZ, self._control_loop)

        self.get_logger().info(f'Watching waypoint file: {self.waypoint_file}')
        self._load_waypoints_from_file()
        
    def _publish_busy(self, busy: bool):

        msg = Bool()

        msg.data = busy

        self.busy_pub.publish(msg)
    # ──────────────────────────────────────────────────────────────────────────────
    # FILE HANDLING
    # ──────────────────────────────────────────────────────────────────────────────
    def _read_waypoint_file_raw(self):
        try:
            with open(self.waypoint_file, 'r') as fh:
                s = fh.read()
        except FileNotFoundError:
            self.get_logger().warn(f'Waypoint file not found: {self.waypoint_file}')
            return None

        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(s)
                if isinstance(data, dict):
                    return data
                self.get_logger().error('Waypoint file parsed but not a dict')
                return None
            except Exception as e:
                self.get_logger().error(f'Failed to parse waypoint file: {e}')
                return None
    def _remove_from_queue(self, id_digits: str):
        if not id_digits:
            return
        self.waypoints = deque([w for w in self.waypoints if w[1] != id_digits])

    def _find_in_queue(self, id_digits: str):

        for i, w in enumerate(self.waypoints):
            if w[1] == id_digits:
                return i
        return None

    def _mark_completed(self, id_digits: str):
        if not id_digits:
            return
        if id_digits in self.completed_ids:
            return
        self.completed_ids.add(id_digits)
        self._remove_from_queue(id_digits)
        self.get_logger().info(f'Waypoint {id_digits} marked completed.')


    def _load_waypoints_from_file(self):
        data = self._read_waypoint_file_raw()
        if data is None:
            return

        try:
            mtime = os.path.getmtime(self.waypoint_file)
        except Exception:
            mtime = time.time()
        if mtime == self.last_mtime:
            return
        self.last_mtime = mtime

        wps = data.get('waypoints', [])
        if not isinstance(wps, (list, tuple)):
            self.get_logger().error('Invalid waypoints structure; expected list')
            return

        added_priority = []
        added_normal = []

        for wp in wps:
            if not isinstance(wp, dict):
                continue

            if 'pose' in wp and isinstance(wp['pose'], dict):
                pose = wp['pose']
                x = pose.get('x', None)
                y = pose.get('y', None)
                yaw = pose.get('yaw', 0.0)
            else:
                x = wp.get('x', None)
                y = wp.get('y', None)
                yaw = wp.get('yaw', 0.0)

            if x is None or y is None:
                continue

            raw_id = wp.get('id', None)
            id_digits = numeric_id(raw_id)
            if id_digits is None:
                continue

            if id_digits in self.completed_ids:
                continue

            desc = wp.get('description', f'wp_{id_digits}')
            desc_upper = str(desc).upper()
            is_turn = str(desc).strip().lower().startswith('turn')

            # Prevent duplicate STOP/RESUME commands.
            if desc_upper in ("STOP", "RESUME"):
                already_pending = any(
                    str(w[2]).upper() == desc_upper
                    for w in self.waypoints
                )
                if already_pending:
                    continue

            is_prio = parse_priority(wp.get('priority', None))
            mode = str(wp.get('mode', 'nav2')).lower()
            if mode not in ('nav2', 'straight'):
                mode = 'nav2'
            reverse = bool(wp.get('reverse', False))

            ps = PoseStamped()
            ps.header.frame_id = MAP_FRAME
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            qz, qw = yaw_to_quat_z_w(float(yaw))
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw

            new_tup = (ps, id_digits, desc, is_prio, mode, reverse)

            if id_digits in self.seen_ids:
                idx = self._find_in_queue(id_digits)
                if idx is not None:
                    self.waypoints[idx] = new_tup
                    if is_prio:
                        self._remove_from_queue(id_digits)
                        self.waypoints.appendleft(new_tup)
                        self.get_logger().info(f'Updated + promoted waypoint {id_digits} -> PRIORITY')
                        
                        # Fix: Ignore system override commands from triggering physical interrupt saves
                        if self.current_waypoint and not self.current_waypoint[3] and desc_upper not in ("STOP", "RESUME"):
                            self._interrupt_current_and_save()
                    else:
                        self._remove_from_queue(id_digits)
                        self.waypoints.append(new_tup)
                else:
                    updated_here = False

                    if self.current_waypoint and self.current_waypoint[1] == id_digits:
                        old_x = self.current_waypoint[0].pose.position.x
                        old_y = self.current_waypoint[0].pose.position.y
                        new_x = ps.pose.position.x
                        new_y = ps.pose.position.y
                        moved = math.hypot(new_x - old_x, new_y - old_y)

                        self.current_waypoint = new_tup
                        updated_here = True

                        if self.script_navigation_active and mode == 'nav2' and self.nav_goal_handle:
                            self._cancel_current_navigation()
                            self.current_waypoint = None
                            self._start_waypoint(new_tup)
                            continue

                        if self.straight_active and mode == 'straight':
                            self.target_x = new_x
                            self.target_y = new_y
                            self.target_yaw_final = float(yaw)
                            self.current_reverse = bool(reverse) and not is_turn

                            if is_turn or moved <= 0.1:
                                self.straight_state = 'FINAL_ROTATION'
                            elif reverse:
                                # Fix: Removed forgotten reference. Hand execution back to control loop alignment.
                                self.straight_state = 'REVERSE_ALIGN'
                            else:
                                self.straight_state = 'ROTATING'

                    if self.interrupted_waypoint and self.interrupted_waypoint[1] == id_digits:
                        self.interrupted_waypoint = new_tup
                        updated_here = True

                    if self.paused_waypoint and self.paused_waypoint[1] == id_digits:
                        self.paused_waypoint = new_tup
                        updated_here = True

                    if not updated_here:
                        continue

                continue

            self.seen_ids.add(id_digits)
            if is_prio:
                added_priority.append(new_tup)
            else:
                added_normal.append(new_tup)

        # Fix: Safely clear queues and states BEFORE appending new mission data
        if self.waiting_for_resume and added_normal:

            self.get_logger().warn(
                "New normal mission received - discarding interrupted mission."
            )

            self.interrupted_waypoint = None
            self.waiting_for_resume = False
            self.waypoints.clear()
            self.resume_event.set()
            
        if added_normal and self.is_paused:
            if self.paused_waypoint:
                self.get_logger().info(
                    f"Discarding paused waypoint "
                    f"{self.paused_waypoint[1]} because a new mission arrived."
                )

            self.paused_waypoint = None
            self.is_paused = False
            self.waypoints.clear()

            if self.script_navigation_active or self.straight_active or self.canceling_script_goal:
                self._cancel_current_navigation()
                self.script_navigation_active = False
                self.canceling_script_goal = False
                self.nav_goal_handle = None
                self.straight_active = False
                self.straight_state = 'IDLE'
                self.current_reverse = False
                self.current_waypoint = None

        if added_priority:
            # Fix: Ensure normal priority tasks can still trigger physical interrupts
            has_real_prio = any(str(p[2]).upper() not in ("STOP", "RESUME") for p in added_priority)
            if has_real_prio and self.current_waypoint and not self.current_waypoint[3]:
                self._interrupt_current_and_save()
                
            for p in reversed(added_priority):
                self._remove_from_queue(p[1])
                self.waypoints.appendleft(p)

        if added_normal:
            for n in added_normal:
                if n[1] not in self.completed_ids:
                    self.waypoints.append(n)

            self.get_logger().info(
                f"Appended {len(added_normal)} waypoint(s)"
            )
            
                    
    def _file_timer(self):
        self._load_waypoints_from_file()

    # ──────────────────────────────────────────────────────────────────────────────
    # SENSOR / MANUAL OVERRIDE
    # ──────────────────────────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        self.last_scan = msg

    def _status_cb(self, msg: GoalStatusArray):
        active = [s for s in msg.status_list if s.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)]

        if self.script_navigation_active:
            return

        # Ignore active status check if we are in the middle of canceling a script goal
        if self.canceling_script_goal:
            if len(active) == 0:
                self.canceling_script_goal = False
            return

        if len(active) > 0 and not self.manual_active:
            self.get_logger().warn('Manual NavigateToPose detected - pausing script tracking')
            self.manual_active = True
            if self.current_waypoint:
                _, _, _, is_prio, _, _ = self.current_waypoint
                if not is_prio:
                    self.manual_interrupted_waypoint = deepcopy(self.current_waypoint)
                self._cancel_current_navigation()
                self.current_waypoint = None
            return

        if len(active) == 0 and self.manual_active:
            self.manual_active = False
            if self.manual_interrupted_waypoint:
                self.get_logger().warn("Manual goal finished. Restoring interrupted waypoint tracking...")
                self.waypoints.appendleft(deepcopy(self.manual_interrupted_waypoint))
                self.manual_interrupted_waypoint = None
            
    # ──────────────────────────────────────────────────────────────────────────────
    # NAVIGATION CONTROL
    # ──────────────────────────────────────────────────────────────────────────────

    def _cancel_current_navigation(self):
        """
        Cancel any active navigation (Nav2 or straight-line).
        """

        # Stop straight-line controller immediately
        if self.straight_active:
            self.stop_robot()
            self.straight_active = False
            self.current_reverse = False
            self.straight_state = 'IDLE'

        self.abort_pending_nav_goal = True

        # Cancel active Nav2 goal
        if self.script_navigation_active and self.nav_goal_handle:
            self.canceling_script_goal = True

            future = self.nav_goal_handle.cancel_goal_async()
            future.add_done_callback(self._cancel_done_callback)

        else:
            self.script_navigation_active = False
            self.nav_goal_handle = None
            
    def _cancel_done_callback(self, future):
        """
        Called once Nav2 confirms the goal has actually been cancelled.
        """

        try:
            future.result()
        except Exception as e:
            self.get_logger().error(
                f"Failed to cancel Nav2 goal: {e}"
            )
            return

        # Ignore stale callbacks if the mission state has already moved on.
        if not self.canceling_script_goal:
            return

        self.script_navigation_active = False
        self.canceling_script_goal = False
        self.nav_goal_handle = None

        self.get_logger().info("Navigation goal cancelled.")
    
            
    def _interrupt_current_and_save(self):

        if self.current_waypoint is None:
            return

        if self.current_waypoint[3]:
            return

        self.interrupted_waypoint = deepcopy(self.current_waypoint)

        self.get_logger().info(
            f"Saved interrupted waypoint "
            f"{self.interrupted_waypoint[1]}"
        )

        self._cancel_current_navigation()

        self.current_waypoint = None
        
    def _main_loop(self):
        if self.is_paused:
            self.stop_robot()

            if self.script_navigation_active or self.straight_active:
                return

            if not self.waypoints:
                return

            next_wp = self.waypoints[0]
            desc = str(next_wp[2]).upper()

            if desc not in ("RESUME", "STOP"):
                return

            self._start_waypoint(self.waypoints.popleft())
            return

        # FIX: Allow STOP commands to bypass the motion lock
        is_pending_stop = bool(self.waypoints and str(self.waypoints[0][2]).upper() == "STOP")

        if not is_pending_stop:
            if self.waiting_for_resume or self.manual_active or self.script_navigation_active or self.straight_active:
                return

        # if not self.waypoints:
        #     if self.interrupted_waypoint and not self.waiting_for_resume:
        #         self.waiting_for_resume = True
        #         self.get_logger().info('Priority task completed.')
        #         self.get_logger().info('Press ENTER to resume interrupted waypoint (or send a new command to overwrite)...')
        #         expected_wpid = self.interrupted_waypoint[1]
        #         self.resume_thread = threading.Thread(
        #             target=self._wait_enter_and_resume,
        #             args=(expected_wpid,),
        #             daemon=True
        #         )
        #         self.resume_thread.start()
        #     return
        if not self.waypoints:
            return
        wp = self.waypoints.popleft()

        self.get_logger().info(
            f"Dequeuing {wp[1]} ({wp[2]})"
        )

        self._start_waypoint(wp)
        
        
    def _start_waypoint(self, wp_tuple):
        # 1. Move unpacking to the top so variables exist
        ps, wpid, desc, is_prio, mode, reverse = wp_tuple

        # 2. Fix the log string to use the variables you just unpacked
        self.get_logger().info(
            f"Dequeuing {wpid} desc={desc} prio={is_prio}"
        )
        
        if wpid in self.completed_ids:
            return

        desc_upper = str(desc).upper()
        is_turn = str(desc).strip().lower().startswith("turn")

        # 1. INTERCEPT HARD STOP
        if desc_upper == "STOP":

            self.get_logger().info(
                "SYSTEM COMMAND: STOP -> Freezing robot."
            )

            if self.current_waypoint:
                self.paused_waypoint = deepcopy(self.current_waypoint)

                self.get_logger().info(
                    f"Saved paused waypoint {self.paused_waypoint[1]}"
                )

            self._cancel_current_navigation()

            self.stop_robot()

            self.is_paused = True

            self._mark_completed(wpid)

            self.current_waypoint = None

            return

        # 2. INTERCEPT RESUME
        if desc_upper == "RESUME":

            self.get_logger().info(
                "SYSTEM COMMAND: RESUME -> Unfreezing robot."
            )

            self.is_paused = False

            self._mark_completed(wpid)

            if self.paused_waypoint:

                _, pid, _, _, _, _ = self.paused_waypoint

                if pid not in self.completed_ids:

                    self.get_logger().info(
                        f"Restoring paused waypoint {pid}"
                    )

                    self.waypoints.appendleft(
                        deepcopy(self.paused_waypoint)
                    )

                self.paused_waypoint = None

            self.current_waypoint = None

            return

        # 3. NORMAL WAYPOINT INITIALIZATION
        self.current_waypoint = (ps, wpid, desc, is_prio, mode, reverse)
        self._publish_busy(True)
        self.current_reverse = bool(reverse) and not is_turn
        px = ps.pose.position.x
        py = ps.pose.position.y
        self.target_yaw_final = quat_to_yaw(ps.pose.orientation)

        self.get_logger().info(
            f'Starting waypoint {wpid} @ ({px:.2f},{py:.2f}) mode={mode} prio={is_prio} reverse={reverse}'
        )

        # 4. STRAIGHT MODE EXECUTION (Restored your original TF Math)
        if mode == 'straight':
            self.target_x = px
            self.target_y = py
            self.straight_active = True

            try:
                t = self.tf_buffer.lookup_transform(MAP_FRAME, BASE_FRAME, Time())
                rx = t.transform.translation.x
                ry = t.transform.translation.y
                ryaw = quat_to_yaw(t.transform.rotation)
                dist = math.hypot(px - rx, py - ry)

                if is_turn:
                    self.straight_state = 'FINAL_ROTATION'
                elif reverse:
                    angle_err = normalize_angle(self.target_yaw_final - ryaw)
                    if abs(angle_err) < HEADING_TOLERANCE:
                        self.straight_state = 'REVERSE_DRIVING'
                    else:
                        self.straight_state = 'REVERSE_ALIGN'
                else:
                    if dist < POSITION_TOLERANCE:
                        self.straight_state = 'FINAL_ROTATION'
                    else:
                        self.straight_state = 'ROTATING'
            except Exception:
                self.straight_state = 'FINAL_ROTATION' if is_turn else ('REVERSE_ALIGN' if reverse else 'ROTATING')
            return

        # 5. NAV2 MODE EXECUTION
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Nav2 action server not available; skipping')
            self.current_waypoint = None
            return

        self.script_navigation_active = True
        self.canceling_script_goal = False
        self.abort_pending_nav_goal = False  # Reset flag for new valid goals

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = ps

        self._send_goal_future = self.nav_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self._goal_response_cb)        
        
        
    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected execution response.')
            self.script_navigation_active = False
            self.current_waypoint = None
            return

        self.nav_goal_handle = goal_handle

        # KILL IN-FLIGHT GOAL IF STOP WAS PRESSED
        if getattr(self, 'abort_pending_nav_goal', False):
            self.get_logger().warn("Killing in-flight Nav2 goal due to STOP override.")
            self.nav_goal_handle.cancel_goal_async()
            self.abort_pending_nav_goal = False
            self.script_navigation_active = False
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_cb)

    def _get_result_cb(self, future):
        status = future.result().status
        success = (status == GoalStatus.STATUS_SUCCEEDED)
        self._handle_nav2_result(success)

    def _handle_nav2_result(self, success: bool):
        if not self.current_waypoint:
            self.script_navigation_active = False
            return

        _, wpid, desc, is_prio, _, _ = self.current_waypoint

        if success:
            self.get_logger().info(f'Nav2 reached waypoint {wpid}')
            self._mark_completed(wpid)
            if is_prio and str(desc).upper() != "STOP":
                self.get_logger().info('Priority task completed. Press ENTER to resume.')
                self.waiting_for_resume = True
                self.resume_event.clear()
                expected_wpid = self.interrupted_waypoint[1]
                threading.Thread(
                    target=self._wait_enter_and_resume,
                    args=(expected_wpid,),
                    daemon=True
                ).start()
            self.current_waypoint = None
            self.script_navigation_active = False

            if not self.waypoints and not self.interrupted_waypoint:
                self._publish_busy(False)
                return
        else:
            self.get_logger().warn(f'Nav2 goal execution terminated on {wpid}.')

        self.current_waypoint = None
        self.script_navigation_active = False

    def _wait_enter_and_resume(self, expected_wpid: str):
        try:
            self.get_logger().info('Operator resumed system.') # To clear terminal visually if needed
            self.get_logger().info('Press ENTER to resume interrupted waypoint (or send a new command to overwrite)...')
            
            # Non-blocking check for stdin keystrokes
            while rclpy.ok() and self.waiting_for_resume:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                if ready:
                    sys.stdin.readline()
                    self.get_logger().info('Operator resumed system via ENTER.')
                    break
            if not self.waiting_for_resume:
                return
            
            if not self.interrupted_waypoint:
                self.waiting_for_resume = False
                self.resume_event.set()
                return
            
            _, interrupted_wpid, _, _, _, _ = self.interrupted_waypoint

            if interrupted_wpid != expected_wpid:
                self.waiting_for_resume = False
                self.resume_event.set()
                return


            completed = interrupted_wpid in self.completed_ids

            queued = self._find_in_queue(interrupted_wpid)

            if (not completed) and queued is None:
                self.waypoints.appendleft(self.interrupted_waypoint)
               
            self.interrupted_waypoint = None
           
            self.waiting_for_resume = False

            self.current_waypoint = None
            self.resume_event.set()

            # self._main_loop()    
            self.get_logger().info(
                "Resume thread finished."
            )        
        except Exception:
            self.get_logger().error(traceback.format_exc())
    # ──────────────────────────────────────────────────────────────────────────────
    # STRAIGHT / REVERSE STATE MACHINE
    # ──────────────────────────────────────────────────────────────────────────────

    def _finish_straight_waypoint(self, label: str):
        wpid = None
        is_prio = False
        desc = ""
        if self.current_waypoint:
            _, wpid, desc, is_prio, _, _ = self.current_waypoint

        if wpid:
            self._mark_completed(wpid)

        self.stop_robot()
        self.straight_active = False
        self.current_reverse = False
        self.straight_state = 'IDLE'

        if is_prio and str(desc).upper() != "STOP":
            self.get_logger().info(
                f'Priority {label} waypoint {wpid} complete. Press ENTER to resume interrupted waypoint (if present).'
            )
            self.waiting_for_resume = True
            self.resume_event.clear()
            expected_wpid = self.interrupted_waypoint[1]
            threading.Thread(
                target=self._wait_enter_and_resume,
                args=(expected_wpid,),
                daemon=True
            ).start()
            self.current_waypoint = None
            return

        self.current_waypoint = None

        if not self.waypoints and not self.interrupted_waypoint:
            self._publish_busy(False)
    
    def _obstacle_in_cone(self, center_angle: float) -> bool:
        scan = self.last_scan
        if scan is None:
            return False

        half_cone = math.radians(OBSTACLE_CONE_ANGLE_DEG)
        angle = scan.angle_min

        for r in scan.ranges:
            rel = normalize_angle(angle - center_angle)
            if abs(rel) <= half_cone:
                if MIN_VALID_SCAN_DISTANCE < r < OBSTACLE_STOP_DISTANCE:
                    return True
            angle += scan.angle_increment

        return False

    def _obstacle_ahead(self) -> bool:
        return self._obstacle_in_cone(0.0)

    def _obstacle_behind(self) -> bool:
        return self._obstacle_in_cone(math.pi)

    def _control_loop(self):
        if not self.straight_active or self.manual_active:
            return

        try:
            t = self.tf_buffer.lookup_transform(MAP_FRAME, BASE_FRAME, Time())
            rx = t.transform.translation.x
            ry = t.transform.translation.y
            q = t.transform.rotation
            ryaw = quat_to_yaw(q)
        except Exception:
            return

        dx = self.target_x - rx
        dy = self.target_y - ry
        dist = math.hypot(dx, dy)
        arrival_tol = POSITION_TOLERANCE * 1.2
        deadband = math.radians(ANGLE_DEADBAND_DEG)

        # Reverse motion
        if self.current_reverse:
            angle_err = normalize_angle(self.target_yaw_final - ryaw)

            if self.straight_state == 'REVERSE_ALIGN':
                if abs(angle_err) < HEADING_TOLERANCE:
                    self.get_logger().info('Reverse alignment complete - start driving')
                    self.straight_state = 'REVERSE_DRIVING'
                    self.stop_robot()
                    return

                rot = max(0.1, min(STRAIGHT_ANGULAR_SPEED, abs(angle_err) * 2.0))
                cmd = Twist()
                cmd.angular.z = rot if angle_err > 0 else -rot
                self._publish_cmd(cmd)
                return

            if self.straight_state == 'REVERSE_WAITING_OBSTACLE':
                if not self._obstacle_behind():
                    self.get_logger().info('Obstacle cleared - resuming reverse motion')
                    self.straight_state = 'REVERSE_DRIVING'
                else:
                    if time.time() - getattr(self, '_wait_start', time.time()) > 60.0:
                        self.get_logger().error('Timeout waiting for obstacle. Aborting task.')
                        self.straight_active = False
                        self.straight_state = 'IDLE'
                        self.current_reverse = False
                        self.current_waypoint = None
                    return

            if self.straight_state == 'REVERSE_DRIVING':
                if dist < arrival_tol:
                    self.get_logger().info('Position reached, changing to reverse final-rotation state.')
                    self.straight_state = 'REVERSE_FINAL_ROTATION'
                    return

                if self._obstacle_behind():
                    self.get_logger().warn('Obstacle detected behind - stopping & waiting')
                    self.straight_state = 'REVERSE_WAITING_OBSTACLE'
                    self.stop_robot()
                    self._wait_start = time.time()
                    return

                if abs(angle_err) > math.radians(MAX_ANGLE_FOR_CORRECTION_DEG):
                    self.get_logger().warn('Large reverse heading error -> re-aligning')
                    self.straight_state = 'REVERSE_ALIGN'
                    self.stop_robot()
                    return

                v = STRAIGHT_LINEAR_SPEED if dist > DECEL_START_DISTANCE else max(MIN_SPEED, STRAIGHT_LINEAR_SPEED * (dist / DECEL_START_DISTANCE))

                cmd = Twist()
                cmd.linear.x = -v
                cmd.angular.z = 0.0 if abs(angle_err) < deadband else (angle_err * 0.5)
                self._publish_cmd(cmd)
                return

            if self.straight_state == 'REVERSE_FINAL_ROTATION':
                if abs(angle_err) < HEADING_TOLERANCE:
                    self.get_logger().info('Reverse tracking task completed.')
                    self._finish_straight_waypoint('reverse')
                    return

                rot = max(0.1, min(STRAIGHT_ANGULAR_SPEED, abs(angle_err) * 2.0))
                cmd = Twist()
                cmd.angular.z = rot if angle_err > 0 else -rot
                self._publish_cmd(cmd)
                return

            if self.straight_state == 'IDLE':
                return

        # Forward straight motion
        if self.straight_state == 'ROTATING':
            target_yaw = math.atan2(dy, dx)
            angle_err = normalize_angle(target_yaw - ryaw)

            if abs(angle_err) < HEADING_TOLERANCE:
                self.get_logger().info('Rotation aligned - start driving')
                self.straight_state = 'DRIVING'
                self.stop_robot()
                return

            rot = max(0.1, min(STRAIGHT_ANGULAR_SPEED, abs(angle_err) * 2.0))
            cmd = Twist()
            cmd.angular.z = rot if angle_err > 0 else -rot
            self._publish_cmd(cmd)
            return

        if self.straight_state == 'DRIVING':
            target_yaw = math.atan2(dy, dx)
            angle_err = normalize_angle(target_yaw - ryaw)

            if dist < arrival_tol:
                self.get_logger().info('Position reached, changing to orientation matching state.')
                self.straight_state = 'FINAL_ROTATION'
                return

            if self._obstacle_ahead():
                self.get_logger().warn('Safety envelope violation! Stopping vehicle.')
                self.straight_state = 'WAITING_OBSTACLE'
                self.stop_robot()
                self._wait_start = time.time()
                return

            if abs(angle_err) > math.radians(MAX_ANGLE_FOR_CORRECTION_DEG):
                self.get_logger().warn('Large heading error -> re-aligning')
                self.straight_state = 'ROTATING'
                self.stop_robot()
                return

            v = STRAIGHT_LINEAR_SPEED if dist > DECEL_START_DISTANCE else max(MIN_SPEED, STRAIGHT_LINEAR_SPEED * (dist / DECEL_START_DISTANCE))

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = 0.0 if abs(angle_err) < deadband else (angle_err * 0.5)
            self._publish_cmd(cmd)
            return

        if self.straight_state == 'WAITING_OBSTACLE':
            if not self._obstacle_ahead():
                self.get_logger().info('Obstacle cleared - resuming')
                self.straight_state = 'DRIVING'
            else:
                if time.time() - getattr(self, '_wait_start', time.time()) > 60.0:
                    self.get_logger().error('Timeout waiting for obstacle. Aborting task.')
                    self.straight_active = False
                    self.straight_state = 'IDLE'
                    self.current_waypoint = None
                return

        if self.straight_state == 'FINAL_ROTATION':
            # FIX: Compare directly against target_yaw_final instead of unstable atan2(dy, dx)
            angle_err = normalize_angle(self.target_yaw_final - ryaw)

            if abs(angle_err) < HEADING_TOLERANCE:
                self.get_logger().info('Straight-line: tracking task completed.')
                self._finish_straight_waypoint('straight')
                return

            rot = max(0.1, min(STRAIGHT_ANGULAR_SPEED, abs(angle_err) * 2.0))
            cmd = Twist()
            cmd.angular.z = rot if angle_err > 0 else -rot
            self._publish_cmd(cmd)
            return

    # ──────────────────────────────────────────────────────────────────────────────
    # LOW-LEVEL PUBLISH / STOP
    # ──────────────────────────────────────────────────────────────────────────────

    def _publish_cmd(self, twist: Twist):
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self.cmd_vel_frame_id
        stamped.twist = twist
        self.cmd_pub.publish(stamped)

    def stop_robot(self):
        self._publish_cmd(Twist())

    def destroy_node(self):
        self.get_logger().info('Shutting down WaypointManager')
        super().destroy_node()


def main():
    rclpy.init()
    node = WaypointManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()