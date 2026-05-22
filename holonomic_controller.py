#!/usr/bin/env python3
# ---------------------- Import Required Libraries ----------------------------
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from hb_interfaces.msg import Pose2D, Poses2D,BotCmd,BotCmdArray,Botarmcmd,PickDropTask,TaskDone

import numpy as np
import math
from linkattacher_msgs.srv import AttachLink,DetachLink
import json
from std_msgs.msg import String
# ---------------------- PID Controller Class --------------------------------
class PID:
    def __init__(self, kp, ki, kd, max_out, i_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.max_out = max_out
        self.i_max = i_max

        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def compute(self, error, dt):
        if dt <= 0.0:
            return 0.0

        # --- Derivative ---
        if not self.initialized:
            derivative = 0.0
            self.initialized = True
        else:
            derivative = (error - self.prev_error) / dt

        # --- Proportional ---
        p = self.kp * error

        # --- Integral (ANTI-WINDUP) ---
        tentative_integral = self.integral + error * dt
        tentative_integral = max(-self.i_max, min(tentative_integral, self.i_max))
        i = self.ki * tentative_integral

        # --- Unsaturated output ---
        output = p + i + self.kd * derivative

        # --- Saturation ---
        if abs(output) <= self.max_out:
            self.integral = tentative_integral
        else:
            output = max(-self.max_out, min(output, self.max_out))

        self.prev_error = error
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

# ---------------------- Main Node Class -------------------------------------
class HolonomicPIDController(Node):
    def __init__(self):
        
      
        super().__init__('holonomic_pid_controller')  # initializing ros node
         # ---------------- Robot Parameters ----------------
        self.declare_parameter("robot_id", 0)
    
 
        self.robot_id = self.get_parameter("robot_id").value
      
        self.declare_parameter("home_pose", [0.0, 0.0, 0.0])
        # ---------- PID PARAMETERS (DECLARATION) ----------
        self.declare_parameter("pid.x.kp", 3.0)
        self.declare_parameter("pid.x.ki", 0.01)
        self.declare_parameter("pid.x.kd", 0.30)
        self.declare_parameter("pid.x.max_out", 8.0)
        self.declare_parameter("pid.x.i_max", 25.0)

        self.declare_parameter("pid.y.kp", 3.0)
        self.declare_parameter("pid.y.ki", 0.01)
        self.declare_parameter("pid.y.kd", 0.30)
        self.declare_parameter("pid.y.max_out", 8.0)
        self.declare_parameter("pid.y.i_max", 25.0)

        self.declare_parameter("pid.theta.kp", 3.0)
        self.declare_parameter("pid.theta.ki", 0.0)
        self.declare_parameter("pid.theta.kd", 0.5)
        self.declare_parameter("pid.theta.max_out", 5.0)
        self.declare_parameter("pid.theta.i_max", 25.0)
        # ---------- Navigation fine tuning ----------
        self.declare_parameter("accept_radius", 0.020)
        self.declare_parameter("sift_x", 0.015)
        self.declare_parameter("sift_y", 0.06)

# ---------- Arm angles (degrees) ----------
        self.declare_parameter("arm.init.base", 80.0)
        self.declare_parameter("arm.init.elbow", 30.0)

        self.declare_parameter("arm.down.base", 130.0)
        self.declare_parameter("arm.down.elbow", 20.0)

        self.declare_parameter("arm.up.base", 80.0)
        self.declare_parameter("arm.up.elbow", 60.0)
        
        self.declare_parameter("arm.drop.base", 80.0)
        self.declare_parameter("arm.drop.elbow", 60.0)

        # ---------- Navigation tuning ----------
        self.accept_radius = self.get_parameter("accept_radius").value
        self.sift_x = self.get_parameter("sift_x").value
        self.sift_y = self.get_parameter("sift_y").value

# ---------- Arm presets ----------
        self.arm_init = {
        "base": self.get_parameter("arm.init.base").value,
        "elbow": self.get_parameter("arm.init.elbow").value,
         }

        self.arm_down = {
         "base": self.get_parameter("arm.down.base").value,
         "elbow": self.get_parameter("arm.down.elbow").value,
        }

        self.arm_up = {
         "base": self.get_parameter("arm.up.base").value,
         "elbow": self.get_parameter("arm.up.elbow").value,
        }


        self.arm_drop={
         "base": self.get_parameter("arm.drop.base").value,
         "elbow": self.get_parameter("arm.drop.elbow").value,
        }
  
        self.robot_pose=None
        self.all_crates={}
        
        self.last_time_xy=None
        self.last_time_theta=None
        
        self.state="WAIT_FOR_CRATE"
        self.attached_model = None
        self.attached_link = None
        self.drop_pose = None
        self.home_pose = tuple(self.get_parameter("home_pose").value)   
        self.wheel_vel=None
        
        self.target_crate_id=None
        self.locked_crate_pose = None
        self.locked_crate_id = None
        self.arm_state = "INIT"   # INIT, DOWN, UP
        
       
        

        #----------------DO NOT CHNAGE----------------------

        # ---------------- PID Parameters ----------------
        # ---------- PID INITIALIZATION ----------
        self.pid_x = PID(
            kp=self.get_parameter("pid.x.kp").value,
            ki=self.get_parameter("pid.x.ki").value,
            kd=self.get_parameter("pid.x.kd").value,
            max_out=self.get_parameter("pid.x.max_out").value,
            i_max=self.get_parameter("pid.x.i_max").value
         )

        self.pid_y = PID(
            kp=self.get_parameter("pid.y.kp").value,
            ki=self.get_parameter("pid.y.ki").value,
            kd=self.get_parameter("pid.y.kd").value,
            max_out=self.get_parameter("pid.y.max_out").value,
            i_max=self.get_parameter("pid.y.i_max").value
             )  

        self.pid_theta = PID(
            kp=self.get_parameter("pid.theta.kp").value,
            ki=self.get_parameter("pid.theta.ki").value,
            kd=self.get_parameter("pid.theta.kd").value,
            max_out=self.get_parameter("pid.theta.max_out").value,
            i_max=self.get_parameter("pid.theta.i_max").value
            )


        # Initialize PIDs
        
        self.arm_delay_active = False
        self.arm_delay_start_time = None
        self.arm_delay_sec = 5.0

        # ---------------- ROS 2 Publishers & Subscribers ----------------
        self.attach_confirmed = False
        self.attach_status_sub = self.create_subscription(String,"crate_attach_status",self.attach_status_cb,10)

        self.bot_poses_sub=self.create_subscription(Poses2D,'/bot_pose',self.robot_pose_cb,10) 
        self.crate_poses_sub=self.create_subscription(Poses2D,'/crate_pose',self.crate_cb,10)
        self.publisher = self.create_publisher(BotCmdArray,'bot_cmd',10)
        self.publisher_arm= self.create_publisher(Botarmcmd,'bot_cmd_arm',10)
        self.pick_cli = self.create_client(AttachLink, 'attach_link')
        self.drop_cli = self.create_client(DetachLink, 'detach_link')
        self.current_task = None

        self.task_sub = self.create_subscription(PickDropTask,'task',self.task_cb,10)

        self.done_pub = self.create_publisher(TaskDone, '/task_done', 10)
        
        # ---------------- Timer for Control Loop ----------------
    
        self.timer = self.create_timer(0.04, self.main_loop)
        
        
        
        self.arm_settled = False
        self.reach_delay_active = False
        self.reach_delay_done = False
        self.reach_delay_timer = None

        self.detach_future = None
        self.detach_waiting = False
        self.detach_result_ready = False
        self.detach_success = False

        self.pick_future = None
        self.pick_waiting = False
        self.pick_result_ready = False
        self.pick_success = False
        self.pick_message = ""
        
        self.drop_future = None
        self.drop_waiting = False
        self.drop_result_ready = False
        self.drop_success = False
        self.drop_message = ""
         
        # Reset per-crate arm logic
        self.arm_settled = False
        
        self.reach_start_time = None

        
        self.drop_delay_active = False
        self.drop_delay_done = False
        self.drop_delay_timer = None
        self.all_robots = {}
        # --- Robot interaction distances ---
        self.robot_slow_dist = 0.40  # start slowing
        self.robot_stop_dist = 0.20  # very close
        # -------- CRATE GEOMETRY & SAFETY (meters) --------
        self.crate_physical_radius = 0.03           # real crate size
        self.crate_inflated_radius = 0.20  # safety bubble while moving
        

        self.all_crates = {}

        self.crate_slow_dist = 0.05  # start slowing near non-target crate
        self.crate_stop_dist = 0.025 # very close → crawl speed
        self.current_task = None
        self.pickup_pose=None
        self.priority_order = [2, 0, 4]
         # ====================================================
        # --- NEW: A* Waypoint & Pyramid Tracking Variables ---
        # ====================================================
        self.pickup_path_x = []         # Holds the X coordinates to the crate
        self.pickup_path_y = []         # Holds the Y coordinates to the crate
        self.drop_path_x = []           # Holds the X coordinates to the drop zone
        self.drop_path_y = []           # Holds the Y coordinates to the drop zone
        self.current_waypoint_index = 0 # Tracks which breadcrumb we are currently driving to
        self.task_level = 0             # 0 = Floor, 1 = Stacked on top
        # ====================================================
        

   # ---------------- Subscriber Callback ----------------
    def task_cb(self, msg: PickDropTask):
        self.get_logger().info(f"New task {msg.task_id} received.")

        self.current_task = msg
        self.task_id = msg.task_id
        self.target_crate_id = msg.crate_id
      
        # Save the final destinations (we still need these for aligning!)
        self.pickup_pose = (msg.pickup_x, msg.pickup_y, msg.pickup_yaw)
        self.drop_pose   = (msg.drop_x, msg.drop_y, msg.drop_yaw)
        
        # Save the Z-height for the arm
        self.task_level = msg.task_level

        # --- NEW: Save the A* Breadcrumb Paths ---
        self.pickup_path_x = msg.pickup_path_x
        self.pickup_path_y = msg.pickup_path_y
        self.drop_path_x = msg.drop_path_x
        self.drop_path_y = msg.drop_path_y

        # Reset the waypoint counter for the new mission!
        self.current_waypoint_index = 0

        
        self.state = "MOVE_TO_CRATE"
  
    def robot_pose_cb(self, msg):
     
     
     for pose in msg.poses:
        x=pose.x/1000.0
        y = pose.y / 1000.0
        theta = math.radians(pose.w)
        self.all_robots[pose.id] = (x, y)
        
        if pose.id != self.robot_id:
            continue  # ignore other robots

        
        self.robot_pose = (x, y, theta)
        
        return  # stop after finding our robot

    
    def crate_cb(self, msg):

     self.all_crates.clear()

     for pose in msg.poses:

        x = pose.x / 1000.0
        y = pose.y / 1000.0
        theta = math.radians(pose.w)

        # Ignore only if target is defined AND matches
        if self.target_crate_id is not None and pose.id == self.target_crate_id:
            continue

        self.all_crates[pose.id] = (x, y, theta)

  
  
  
    # ---------------- Control Loop ----------------
   
   
    def control_cb(self, target_x, target_y,is_final=True):
    # ---------------- Pose availability ----------------
      if self.robot_pose is None:
        self.get_logger().debug("No robot pose yet")
        return False
      self.get_logger().info(f"robot pose{self.robot_pose}")
      self.get_logger().info(f"target pose{target_x,target_y}")
     # ---------------- Time handling ----------------
      now = self.get_clock().now()

      if self.last_time_xy is None:
        self.last_time_xy = now
        return False

      dt = (now - self.last_time_xy).nanoseconds / 1e9
      self.last_time_xy = now
      if dt <= 0.0 or dt > 1.0:
        
        return False

      

    # ---------------- Current pose (meters) ----------------
      x_mm, y_mm, theta = self.robot_pose
      x = x_mm 
      y = y_mm 

      target_x_m = target_x
      target_y_m = target_y

    # ---------------- Errors ----------------
      error_x = target_x_m - x
      error_y = target_y_m - y

      distance = math.hypot(error_x, error_y)
      self.get_logger().info(f"dis{distance}")
    # ---------------- Thresholds (meters) ----------------
      current_accept_radius = self.accept_radius if is_final else 0.15   # e.g. 0.15
      
    # ---------------- STOP command helper ----------------
      
    # =====================================================
    # ================= ACCEPT ZONE =======================
    # =====================================================
      if distance <= current_accept_radius:
          if is_final:
                msg = BotCmdArray()
                cmd = BotCmd()
                cmd.id = self.robot_id
                cmd.m1 = 0.0; cmd.m2 = 0.0; cmd.m3 = 0.0
                msg.cmds.append(cmd)
                self.publisher.publish(msg)
                self.pid_x.reset()
                self.pid_y.reset()
          return True  # TARGET REAC
          
         
            
             
             
             
             
    # ---------------- Robot-frame transform ----------------
      ex_r =  error_x * math.cos(theta) + error_y * math.sin(theta)
      ey_r = -error_x * math.sin(theta) + error_y * math.cos(theta)

    # ---------------- PID (m/s) ----------------
      v_x = self.pid_x.compute(ex_r, dt)
      v_y = self.pid_y.compute(ey_r, dt)
      
     # ... [PID calculations above remain the same] ...
      
      # ---------------- 1. ROBOT AVOIDANCE (Yielding) ----------------
      robot_dist, other_id = self.nearest_robot_distance()
      robot_fx, robot_fy = 0.0, 0.0
      if robot_dist is not None and robot_dist < self.robot_slow_dist:
             rx_other, ry_other = self.all_robots[other_id]
             dx_r = x - rx_other
             dy_r = y - ry_other
            
             if not self.has_priority(other_id):
                stop = self.current_stop_distance()
                if robot_dist <= stop:
                    v_x *= 0.0; v_y *= 0.0
                else:
                    v_x *= 0.3 # Slow down heavily to yield
                    
                strength = (1.0 / max(robot_dist, 0.01) - 1.0 / self.robot_slow_dist)
                norm_d = max(math.hypot(dx_r, dy_r), 0.01)
                dx_norm = dx_r / norm_d
                dy_norm = dy_r / norm_d
                
                robot_fx += dx_norm * strength
                robot_fy += dy_norm * strength
                robot_fx += -dy_norm * (strength * 0.5) # Swirl dodge
             else:
                if robot_dist < 0.20:
                    v_x *= 0.7; v_y *= 0.7

      if robot_fx != 0.0 or robot_fy != 0.0:
            robot_push_gain = 8.0  
            v_x += robot_fx * robot_push_gain
            v_y += robot_fy * robot_push_gain
      # ---------------- 2. CRATE AVOIDANCE (Potential Fields) ----------------
      avoid_fx, avoid_fy, num_threats = self.compute_repulsive_force(x, y)
      if num_threats > 0:
            push_gain = 5.0
            v_x += avoid_fx * push_gain
            v_y += avoid_fy * push_gain
            force_mag = math.hypot(avoid_fx, avoid_fy)
            slowdown_factor = 3.0 / (1.0 + force_mag)
            v_x *= slowdown_factor
            v_y *= slowdown_factor
           
           # Safety: If surrounded by many crates, reduce overall speed
           
           
      max_speed =12.0
      speed = math.hypot(v_x, v_y)
      # ---------------------------------------------------------------------
      if speed > max_speed:
              v_x = (v_x / speed) * max_speed
              v_y = (v_y / speed) * max_speed
              v_theta = 0.0 
      # ... [Kinematics and Publish below remain the same] ...
          
      v_theta = 0.0 
      self.get_logger().info(f"vx and vy are{v_x,v_y}")
    # ---------------- Wheel kinematics ----------------
      alpha_deg = np.array([120, 240, 0])
      alpha_rad = np.deg2rad(alpha_deg)
      L = 0.08

      M = np.array([
        [np.cos(alpha_rad[0]), np.cos(alpha_rad[1]), np.cos(alpha_rad[2])],
        [np.sin(alpha_rad[0]), np.sin(alpha_rad[1]), np.sin(alpha_rad[2])],
        [1/(3*L), 1/(3*L), 1/(3*L)]
      ])

      wheel_vel = np.linalg.inv(M) @ np.array([v_x, v_y, v_theta])

    # ---------------- Publish command ----------------
      cmd = BotCmd()
      cmd.id = self.robot_id
      cmd.m1 = float(wheel_vel[2])
      cmd.m2 = float(wheel_vel[1])
      cmd.m3 = float(wheel_vel[0])
      

      msg = BotCmdArray()
      msg.cmds.append(cmd)
      self.publisher.publish(msg)
      self.get_logger().info(f"msg{msg}")
      return False

    def reach_delay_callback(self):
        self.reach_delay_done = True
        self.get_logger().info("Reach delay completed.")

        if self.reach_delay_timer is not None:
           try:
            self.reach_delay_timer.destroy()
           except:
              pass
           self.reach_delay_timer = None


    def align_angle(self, target_theta):

     # ---------------- Pose availability ----------------
     if self.robot_pose is None:
        return False

     x_mm, y_mm, theta = self.robot_pose

     # ---------------- Angle error (wrapped) ----------------
     error_theta = (target_theta - theta + math.pi) % (2 * math.pi) - math.pi
     error_theta_deg = math.degrees(error_theta)
     self.get_logger().info(f"Yaw error: {error_theta_deg:.2f} deg")

     # ---------------- Accept zone (aligned) ----------------
     if abs(error_theta) < math.radians(3.0):
        # STOP wheels only
        stop_msg = BotCmdArray()
        stop_bot = BotCmd()
        stop_bot.id = self.robot_id
        stop_bot.m1 = 0.0
        stop_bot.m2 = 0.0
        stop_bot.m3 = 0.0
        stop_msg.cmds.append(stop_bot)
        self.publisher.publish(stop_msg)

        self.pid_theta.reset()
        self.last_time_theta = None
        return True

    # ---------------- Time handling ----------------
     now = self.get_clock().now()

     if self.last_time_theta is None:
        self.last_time_theta = now
        return False

     dt = (now - self.last_time_theta).nanoseconds / 1e9
     if dt <= 0.0 or dt > 1.0:
        self.last_time_theta = now
        return False

     self.last_time_theta = now

    # ---------------- Angular PID ----------------
     v_theta = self.pid_theta.compute(error_theta, dt)

    # ---------------- Wheel kinematics (pure rotation) ----------------
     alpha_deg = np.array([120, 240, 0])
     alpha_rad = np.deg2rad(alpha_deg)
     L = 0.08

     M = np.array([
        [np.cos(alpha_rad[0]), np.cos(alpha_rad[1]), np.cos(alpha_rad[2])],
        [np.sin(alpha_rad[0]), np.sin(alpha_rad[1]), np.sin(alpha_rad[2])],
        [1/(3*L), 1/(3*L), 1/(3*L)]
     ])

     wheel_vel = np.linalg.inv(M) @ np.array([0.0, 0.0, v_theta])

    # ---------------- Publish wheel command only ----------------
     cmd = BotCmd()
     cmd.id = self.robot_id
     cmd.m1 = float(wheel_vel[2])
     cmd.m2 = float(wheel_vel[1])
     cmd.m3 = float(wheel_vel[0])

     msg = BotCmdArray()
     msg.cmds.append(cmd)
     self.publisher.publish(msg)
     self.get_logger().info(f"{msg}")
     return False

    def get_crate_color(self, crate_id):
      r = crate_id % 3
      return "red" if r == 0 else "green" if r == 1 else "blue"

    def main_loop(self):
        # WAIT for a crate to appear
        if self.state == 'WAIT_FOR_CRATE':
              if self.current_task is None:
                self.get_logger().warn("No target crate assigned. Waiting.")
                return

                  # 2️⃣ Mission assigned but pose not received yet
              
              self.get_logger().info("Task received — moving to pickup")
              self.state = "MOVE_TO_CRATE"   # or MOVE_TO_PICKUP
              return
        # MOVE to crate coordinates
        if self.state == 'MOVE_TO_CRATE':
            
            # Safety check just in case the path is empty
            if not self.pickup_path_x:
                self.get_logger().warn("No pickup path received!")
                return

            # 1. Grab the CURRENT breadcrumb using the index (and convert mm to meters)
            target_x = self.pickup_path_x[self.current_waypoint_index] / 1000.0
            target_y = self.pickup_path_y[self.current_waypoint_index] / 1000.0
            
            # 2. Check if this is the final breadcrumb in the list
            is_final = (self.current_waypoint_index == len(self.pickup_path_x) - 1)
            
            # 3. Send to PID Controller (Pass the is_final flag!)
            
            reached = self.control_cb(target_x, target_y, is_final=is_final)
            
            if reached:
                if not is_final:
                    # We just passed a middle waypoint! Advance to the next one.
                    self.current_waypoint_index += 1
                else:
                    # We reached the FINAL standing spot!
                    self.get_logger().info('Reached final crate position.')
                    self.locked_crate_pose = self.pickup_pose
                    self.locked_crate_id = self.target_crate_id
                    self.set_arm_state("DOWN")
                    
                    # Reset the waypoint counter back to 0 so it's ready for the drop path
                    self.current_waypoint_index = 0 
                    self.state = 'ALIGN_TO_CRATE'
            return

       
      
            # 3) NEW STATE: ALIGN ANGLE
        # 3) ALIGN ANGLE (Pickup)
        if self.state == 'ALIGN_TO_CRATE':
            # Extract the coordinates and the yaw sent by the Allocator
            x, y, pickup_yaw = self.locked_crate_pose
           
            # Use the dynamically calculated yaw instead of hardcoded 0.0!
            aligned = self.align_angle(pickup_yaw)

            if aligned:
                self.get_logger().info(f"Angle aligned to {math.degrees(pickup_yaw):.1f} deg. Attaching crate.")
                self.state = 'ATTACH_CRATE'
            return

        # ATTACH 
        if self.state == 'ATTACH_CRATE':

    # --------------------------------------
    # Step 1: Send pick request (once)
    # --------------------------------------
          if not self.pick_waiting and not self.pick_result_ready:

            req = AttachLink.Request()
            req.crate_id = self.locked_crate_id
            req.color = self.get_crate_color(self.locked_crate_id)
                # already known / locked earlier

            self.get_logger().info(
            f"Sending PICK request: crate_id={req.crate_id}, color={req.color}"
             )

            self.pick_future = self.pick_cli.call_async(req)
            self.pick_future.add_done_callback(self.pick_response_cb)

            self.pick_waiting = True
            return

             # --------------------------------------
             # Step 2: Waiting → do nothing
             # --------------------------------------
          if self.pick_waiting:
          
            return

             # --------------------------------------
             # Step 3: Result received
             # --------------------------------------
          if self.pick_result_ready:

            if self.pick_success:
               if not self.arm_delay_active:
                 self.arm_delay_active = True
                 self.arm_delay_start_time = self.get_clock().now()
                 self.get_logger().info(
                   "Pick OK. Waiting 5 seconds before raising arm."
                  )
                 return
               elapsed = (
                  self.get_clock().now() - self.arm_delay_start_time
               ).nanoseconds / 1e9

               if elapsed < self.arm_delay_sec:
                return   
             
               self.set_arm_state("UP")
               self.get_logger().info("Crate picked successfully.")
              #  gate transition on attach confirmation
               if not self.attach_confirmed:
                 self.get_logger().warn(
                   "Arm UP but attach not confirmed yet. Holding state."
                 )
                 return   

              # reset flags
               self.pick_result_ready = False
               self.pick_waiting = False
               self.attach_confirmed = False 
               self.state = "MOVE_TO_DROP"
               self.pick_waiting = False
               self.pick_result_ready = False
               self.attach_confirmed = False
               self.arm_delay_active = False
               self.pick_future = None
               return

            else:
             self.get_logger().error(
                f"Pick failed: {self.pick_message}"
               )

               # reset flags
             self.pick_result_ready = False
             self.pick_waiting = False
            
             self.state = "DONE"   # or RETRY if you want
             return

        

        # MOVE to drop location
       # MOVE to drop location
        if self.state == 'MOVE_TO_DROP':
            
            if not self.drop_path_x:
                self.get_logger().warn("No drop path received!")
                return

            # 1. Grab the current waypoint for the DROP path (convert mm to meters)
            target_x = self.drop_path_x[self.current_waypoint_index] / 1000.0
            target_y = self.drop_path_y[self.current_waypoint_index] / 1000.0
            
            # 2. Check if this is the final breadcrumb
            is_final = (self.current_waypoint_index == len(self.drop_path_x) - 1)
            
            # 3. Drive! (Notice we do not subtract self.sift_y anymore!)
            if self.control_cb(target_x, target_y, is_final=is_final):
                
                if not is_final:
                    # We hit a middle point. Keep driving!
                    self.current_waypoint_index += 1
                else:
                    # We reached the final standing spot to drop the crate!
                    self.get_logger().info('Reached final drop standing spot.')
                    self.state = 'DROP_ALIGN'
            return

          

        # ALIGN ANGLE (Drop)
        if self.state == 'DROP_ALIGN':
            # Grab the drop yaw that we saved in task_cb
            drop_x, drop_y, drop_yaw = self.drop_pose
           
            # Align to the correct side of the Pyramid
            aligned = self.align_angle(drop_yaw)

            if aligned:
                self.get_logger().info(f"Angle aligned to {math.degrees(drop_yaw):.1f} deg. Setting Arm Level.")
                if not self.drop_delay_active:
                  
                  # --- PYRAMID HEIGHT LOGIC ---
                  if self.task_level == 1:
                      self.set_arm_state("DROP") # Hover height for Level 1!
                  else:
                      self.set_arm_state("DOWN") # Floor height for Level 0!
                  
                  self.drop_delay_active = True
                  self.drop_delay_done = False
                  self.drop_delay_timer = self.create_timer(5.0, self.drop_delay_callback)
                  return

                if self.drop_delay_done:
                   self.drop_delay_active = False
                   self.drop_delay_done = False
                   self.drop_delay_timer = None
                   self.state = "DROP_CRATE"
                   return
            return
        
         
         
         
         
         
         
         # DROP crate (detach)
        if self.state == "DROP_CRATE":

    # --------------------------------------
    # Step 1: Send drop request (only once)
    # --------------------------------------
           if not self.drop_waiting and not self.drop_result_ready:

             req = DetachLink.Request()
             req.crate_id = self.locked_crate_id
             req.color = self.get_crate_color(self.locked_crate_id)
             self.get_logger().info(
              f"Sending DROP request for crate_id={req.crate_id}"
                 )

             try:
                self.drop_future = self.drop_cli.call_async(req)
                self.drop_future.add_done_callback(self.drop_response_cb)
                self.drop_waiting = True
             except Exception as e:
                self.get_logger().error(f"Drop request failed to send: {e}")
                self.drop_result_ready = True
                self.drop_success = False
                self.drop_message = str(e)

             return

    # --------------------------------------
    # Step 2: Waiting → do nothing
    # --------------------------------------
           if self.drop_waiting:
            return

    # --------------------------------------
    # Step 3: Result received
    # --------------------------------------
           if self.drop_result_ready:

              if self.drop_success:
                done = TaskDone()
                done.robot_id = self.robot_id
                done.crate_id = self.target_crate_id
                done.success = True
                self.done_pub.publish(done)
                self.current_task = None

                self.get_logger().info("Drop SUCCESS → Returning home")
                self.set_arm_state("INIT")
                

            # Forget crate completely
                self.locked_crate_pose = None
                self.locked_crate_id = None

            # Reset flags
                self.drop_waiting = False
                self.drop_result_ready = False

                self.state = "RETURN_HOME"
                return

              else:
                self.get_logger().error(
                f"Drop FAILED: {self.drop_message}"
                 )

                 # Reset flags
                self.drop_waiting = False
                self.drop_result_ready = False

                self.state = "DONE"
                return

        
  
        
        


        # RETURN HOME
        if self.state == 'RETURN_HOME':
            hx, hy, ht = self.home_pose
            hx=hx/1000
            hy=hy/1000
            
            if self.control_cb(hx, hy):
                self.get_logger().info('Returned home. Task finished.')
                self.state = 'align to intial position'
            return

        if self.state == 'align to intial position':
            crate_theta=math.radians(0)
            aligned_new = self.align_angle(crate_theta)

            if aligned_new:
                self.get_logger().info("Angle aligned. Attaching crate.")
                self.state = 'DONE'
            return 
        # DONE
       
        if self.state == 'DONE':
            # idle — you could reset state to WAIT_FOR_CRATE to process more crates
            pass
    
    
    

    
    def pick_response_cb(self, future):
      try:
        res = future.result()
        self.pick_success = res.success
        self.pick_message = res.message
      except Exception as e:
        self.get_logger().error(f"Pick service exception: {e}")
        self.pick_success = False
        self.pick_message = str(e)

      self.pick_result_ready = True
      self.pick_waiting = False
 
    
    def drop_response_cb(self, future):
     try:
        res = future.result()
        self.drop_success = res.success
        self.drop_message = res.message
     except Exception as e:
        self.get_logger().error(f"Drop service exception: {e}")
        self.drop_success = False
        self.drop_message = str(e)

     self.drop_result_ready = True
     self.drop_waiting = False
    
    def drop_delay_callback(self):
      self.drop_delay_done = True
      self.get_logger().info("Drop delay completed.")

      if self.drop_delay_timer is not None:
        try:
            self.drop_delay_timer.destroy()
        except:
            pass
        self.drop_delay_timer = None

    def set_arm_state(self, new_state):
     if self.arm_state == new_state:
        return  # already in this state, do nothing

     arm_cmd = Botarmcmd()
     arm_cmd.id = self.robot_id

     if new_state == "INIT":
        arm_cmd.base = self.arm_init["base"]
        arm_cmd.elbow = self.arm_init["elbow"]

     elif new_state == "DOWN":
        arm_cmd.base = self.arm_down["base"]
        arm_cmd.elbow = self.arm_down["elbow"]

     elif new_state == "UP":
        arm_cmd.base = self.arm_up["base"]
        arm_cmd.elbow =self.arm_up["elbow"]   # example lift
     elif new_state == "DROP":
        arm_cmd.base = self.arm_drop["base"]
        arm_cmd.elbow =self.arm_drop["elbow"]
     self.publisher_arm.publish(arm_cmd)
     self.arm_state = new_state
     self.get_logger().info(f"{arm_cmd}")
    
    
    
    def attach_status_cb(self, msg):
     if msg.data.startswith("ATTACHED"):
        self.attach_confirmed = True
     elif msg.data.startswith("FAILED"):
        self.attach_confirmed = False



    def nearest_robot_distance(self):
      if self.robot_pose is None:
        return None

      x, y, _ = self.robot_pose
      min_dist = None
      nearest_id=None
      for rid, (rx, ry) in self.all_robots.items():
        if rid == self.robot_id:
            continue 

        d = math.hypot(x - rx, y - ry)

        if min_dist is None or d < min_dist:
            min_dist = d
            nearest_id = rid
      return min_dist,nearest_id 
   
   
   # --- THE VIP BYPASS & SWIRL LOGIC ---
    def compute_repulsive_force(self, rx, ry):
        total_fx, total_fy = 0.0, 0.0
        active_threats = 0
        radius = self.current_crate_radius()      
        slow_dist = self.crate_slow_dist
        eps = 1e-6   
        
        target_color = self.get_crate_color(self.target_crate_id) if self.target_crate_id is not None and self.target_crate_id != -1 else None

        for cid, (cx, cy, _) in self.all_crates.items():
            # 1. Ignore the crate we are actively holding/targeting
            if cid == self.target_crate_id: continue

            # --- VIP BYPASS: SIDE-BY-SIDE PLACEMENT ---
            # If we are dropping a crate, ignore repulsion from crates of the same color!
            if self.state in ['MOVE_TO_DROP', 'DROP_ALIGN'] and target_color is not None:
                if self.get_crate_color(cid) == target_color:
                    continue

            dx = rx - cx; dy = ry - cy
            dist = math.hypot(dx, dy)
            effective_dist = dist - radius
            if effective_dist >= slow_dist: continue

            active_threats += 1
            if dist > eps:
                dx /= dist; dy /= dist
            else: continue

            d = max(effective_dist, eps)
            strength = (1.0/d - 1.0/slow_dist) / (1.0/eps - 1.0/slow_dist)   
            strength = max(0.0, min(strength, 1.0))

            # Radial Push
            total_fx += dx * strength
            total_fy += dy * strength
            
            # Tangential Swirl (slide sideways around crate)
            total_fx += -dy * (strength * 0.8) 
            total_fy +=  dx * (strength * 0.8)
            
        return total_fx, total_fy, active_threats
    
    def current_crate_radius(self):

      if self.state in [
        "ALIGN_TO_CRATE",
        "ATTACH_CRATE",
        "DROP_CRATE",
        "MOVE_TO_DROP",
        "MOVE_TO_CRATE",
        "DROP_ALIGN"
                         ]:
        return self.crate_physical_radius

       #Default: moving in open space
      return self.crate_inflated_radius

    def get_crate_color(self, crate_id):
      r = crate_id % 3
      return "red" if r == 0 else "green" if r == 1 else "blue"
    

    def current_stop_distance(self):

         # Handling crate → precise motion → need tighter stop
     if self.state in [
        "RETURN_HOME",
        "align to intial position"
     ]:
        return 0.01

    # Free roaming → higher speed → larger safety
     return 0.40
    def has_priority(self, other_id):

      order = self.priority_order

       # rank = position in list
         # smaller index → higher priority

      if self.robot_id in order:
           my_rank = order.index(self.robot_id)
      else:
           my_rank = len(order)   # lowest priority

      if other_id in order:
        other_rank = order.index(other_id)
      else:
        other_rank = len(order)

      return my_rank < other_rank

# ---------------------- Main Function -------------------------------------
def main(args=None):
    rclpy.init(args=args)
    controller = HolonomicPIDController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Keyboard interrupt received.")
    finally:
        
        controller.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
