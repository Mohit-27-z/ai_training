#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from hb_interfaces.msg import BotCmd,BotCmdArray,Botarmcmd
import json
import paho.mqtt.client as mqtt
from std_msgs.msg import String
from linkattacher_msgs.srv import  AttachLink,DetachLink
import time





class PWMcalci:
    def __init__(self):
        # ================= PWM VALUES =================
        self.PWM_STOP     = 1500

        self.PWM_CCW_MAX  = 1000
        self.PWM_CW_MAX   = 2000

        self.PWM_CCW_MIN  = 1440   # minimum effective CCW (outside dead zone)
        self.PWM_CW_MIN   = 1550   # minimum effective CW  (outside dead zone)

        # ================= VELOCITY LIMITS =================
        self.V_STOP       = .020    # true stop band (ignore velocity)
        self.V_MIN_CCW    = .6192   # minimum CCW velocity to enter table
        self.V_MIN_CW     = .540    # minimum CW velocity to enter table
        self.V_MAX        = 2.080
         # -------- Servo limits --------
        self.ANGLE_MIN = 0.0
        self.ANGLE_MAX = 180.0

        # -------- PWM limits (safe) --------
        self.PWM_MIN = 600     # µs (adjust if needed)
        self.PWM_MAX = 2400    # µs
        self.PWM_CENTER = 1500
        # ================= CALIBRATED TABLE =================
        self.VEL_PWM_TABLE = [
            (-self.V_MAX,  self.PWM_CCW_MAX),
            (-1.9476, 1050),
            (-1.7028, 1100),
            (-1.5156, 1150),
            (-1.3608, 1200),
            (-1.0476, 1250),
            (-.9072,  1300),
            (-.8496,  1350),
            (-.7164,  1400),
            (-.6192,  self.PWM_CCW_MIN),
            ( 0.0, 1500),
            ( .5400,  self.PWM_CW_MIN),
            ( .6804,  1600),
            ( .8496,  1650),
            (1.0440,  1700),
            (1.1340,  1750),
            (1.3608,  1800),
            (1.5120,  1850),
            (1.9410,  1900),
            (1.9800,  1950),
            ( self.V_MAX, self.PWM_CW_MAX),
        ]

    # ======================================================
    # VELOCITY → PWM
    # ======================================================
    def velocity_to_pwm(self, v):
        # ---------- 1. TRUE STOP ----------
        if abs(v) < self.V_STOP:
            return self.PWM_STOP

        # ---------- 2. MINIMUM EFFECTIVE MOTION ----------
        if self.V_STOP < v < self.V_MIN_CW:
            return self.PWM_CW_MIN

        if -self.V_MIN_CCW < v < -self.V_STOP:
            return self.PWM_CCW_MIN

        # ---------- 3. CLAMP VELOCITY ----------
        v = max(min(v, self.V_MAX), -self.V_MAX)

        # ---------- 4. TABLE INTERPOLATION ----------
        for i in range(len(self.VEL_PWM_TABLE) - 1):
            v1, pwm1 = self.VEL_PWM_TABLE[i]
            v2, pwm2 = self.VEL_PWM_TABLE[i + 1]

            if v1 <= v <= v2:
                ratio = (v - v1) / (v2 - v1)
                pwm = pwm1 + ratio * (pwm2 - pwm1)
                pwm = int(round(pwm))
                pwm = max(self.PWM_CCW_MAX, min(pwm, self.PWM_CW_MAX))
                return pwm


        # ---------- 5. SAFETY SATURATION ----------
        return self.PWM_CW_MAX if v > 0 else self.PWM_CCW_MAX


class Ros2MqttBridge(Node):
    def __init__(self):
        super().__init__('ros2_mqtt_bridge')
        
        self.declare_parameter("mqtt_ns", "crystal")
        self.mqtt_ns = self.get_parameter("mqtt_ns").value

        
        
        
        
        
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_message = self.on_mqtt_message
        
        self.mqtt_client.connect("localhost", 1883, 60)
        self.mqtt_cmd_pwm  = f"{self.mqtt_ns}/esp32/cmd/pwm"
        self.mqtt_cmd_arm  = f"{self.mqtt_ns}/esp32/cmd/arm"
        self.mqtt_fb_ir    = f"{self.mqtt_ns}/esp32/feedback/ir"
        self.mqtt_cmd_em  = f"{self.mqtt_ns}/esp32/cmd/em" 
        
        
        
        self.mqtt_client.subscribe(self.mqtt_fb_ir, qos=0)

        self.mqtt_client.loop_start()
        
        self.bot_vel_sub=self.create_subscription(BotCmdArray,'bot_cmd',self.pwm_conv,10)
       
       
       
       
        self.arm_position_sub=self.create_subscription(Botarmcmd,'bot_cmd_arm',self.arm_conv,10)
        self.service_pick= self.create_service(AttachLink,'attach_link',self.attach_crate)
        self.service_drop=self.create_service(DetachLink,'detach_link',self.detach_crate)
        self.attach_status_pub = self.create_publisher(
           String,
          "crate_attach_status",
           10
          )

        self.attach_status_published = False

        self.pwm_calc = PWMcalci() 
        self.prev_m1=None
        self.prev_m2=None
        self.prev_m3=None
        self.prev_base = None
        self.prev_elbow = None
        self.ir_state = None          # 0 or 1
        self.attach_in_progress = False
        
        self.attach_start_time = None
      
        

        self.attach_timer = self.create_timer(0.05, self._attach_watchdog)
         # seconds (debounce)

    
    
    
    def pwm_conv(self, msg: BotCmdArray):

     if not msg.cmds:
        payload = {
            "id": int(-1),
            "m1": int(1500.0),
            "m2": int(1500.0),
            "m3": int(1500.0),
        }
        self.mqtt_client.publish(
            self.mqtt_cmd_pwm,
            json.dumps(payload),
            qos=0
        )
     for bot in msg.cmds:
        bot_id = bot.id

        # Velocity → PWM
        m1 = float(self.pwm_calc.velocity_to_pwm(bot.m1))
        m2 = float(self.pwm_calc.velocity_to_pwm(bot.m2))
        m3 = float(self.pwm_calc.velocity_to_pwm(bot.m3))

        # ---------- FIRST COMMAND (INITIALIZATION) ----------
        if self.prev_m1 is None:
            self.prev_m1 = m1
            self.prev_m2 = m2
            self.prev_m3 = m3

            send_m1, send_m2, send_m3 = m1, m2, m3

        # ---------- NORMAL FILTERED OPERATION ----------
        else:
            send_m1, self.prev_m1 = self._filtered_value(m1, self.prev_m1)
            send_m2, self.prev_m2 = self._filtered_value(m2, self.prev_m2)
            send_m3, self.prev_m3 = self._filtered_value(m3, self.prev_m3)

        payload = {
            "id": int(bot_id),
            "m1": int(send_m1),
            "m2": int(send_m2),
            "m3": int(send_m3),
        }

        self.get_logger().info(f"PWM payload: {payload}")

        self.mqtt_client.publish(
            self.mqtt_cmd_pwm,
            json.dumps(payload),
            qos=0
        )
    def _filtered_value(self, new_val, prev_val, threshold=0):
     if abs(new_val - prev_val) > threshold:
        return new_val, new_val
     else:
        return prev_val, prev_val
    
    def arm_conv(self, msg: Botarmcmd):

        bot_id = msg.id
        base = msg.base
        elbow = msg.elbow

        # -------- BASE JOINT --------
        base_changed, send_base, self.prev_base = \
            self._arm_changed(base, self.prev_base)

        # -------- ELBOW JOINT --------
        elbow_changed, send_elbow, self.prev_elbow = \
            self._arm_changed(elbow, self.prev_elbow)

        # If nothing changed → DO NOT PUBLISH
        if not base_changed and not elbow_changed:
           return

        payload = {
            "id": int(bot_id),
            "base": int(send_base),
            "elbow": int(send_elbow),
        }

        self.get_logger().info(f"ARM payload: {payload}")

        self.mqtt_client.publish(
            self.mqtt_cmd_arm,
            json.dumps(payload),
            qos=0
        )
    def _arm_changed(self, new_val, prev_val):
      """
      Returns (changed, value_to_send, updated_prev)
      """
      if new_val is None:
        new_val = 0

      if prev_val is None or new_val != prev_val:
        return True, new_val, new_val
      else:
        return False, prev_val, prev_val

    
    
    def set_electromagnet(self, mode: int):
      """
      EM_OFF    → magnet OFF
     EM_ATTACH → full power (pickup)
     EM_HOLD   → reduced power (hold)
     """
      EM_OFF    = 0
      EM_ATTACH = 1
      EM_HOLD   = 2

      if mode == EM_ATTACH:
         pwm = 300         # full power pickup
      elif mode == EM_HOLD:
        pwm = 300         # reduced holding power (example)
      else:  # EM_OFF
        pwm = 0         # OFF / neutral

      payload = {
        "id": 0,
        "mosfet": int(pwm),
      }

      self.get_logger().info(f"Electromagnet PWM → {pwm}")

      self.mqtt_client.publish(
         self.mqtt_cmd_em,
        json.dumps(payload),
        qos=0
      )

    def attach_crate(self, request, response):
      if self.attach_in_progress:
         response.success = False
         response.message = "Attach already in progress"
         return response

      self.get_logger().info(
        f"Attach request: crate_id={request.crate_id}, color={request.color}"
       )

         # Start attach
      self.attach_in_progress = True
      self.attach_start_time = self.get_clock().now()
    
      self.attach_status_published = False

      self.set_electromagnet(1)  # EM_ATTACH

      response.success = True
      response.message = "Attach started"
      return response

         
    def detach_crate(self, request, response):
      crate_id = request.crate_id
      color = request.color
      EM_OFF    = 0
      self.get_logger().info(
        f"Detach request: crate_id={crate_id}, color={color}"
     )

    # Prevent concurrent operations
      if self.attach_in_progress:
        response.success = False
        response.message = "Another attach/detach operation in progress"
        return response

      self.attach_in_progress = True

      # Turn OFF electromagnet
      self.set_electromagnet(EM_OFF)

      start_time = self.get_clock().now()
      timeout_sec = 10.0
      success = False
      if self.ir_state is None:
       self.get_logger().warn("IR state not received yet")

      while (self.get_clock().now() - start_time).nanoseconds < timeout_sec * 1e9:
        if self.ir_state == 1:   # detached confirmed
            success = True
            break

        time.sleep(0.05)

      if success:
        response.success = True
        response.message = (
            f"Crate {crate_id} ({color}) detached successfully"
        )
        self.get_logger().info(response.message)
      else:
        response.success = False
        response.message = (
            f"Detach failed for crate {crate_id} ({color}) – timeout"
        )
        self.get_logger().warn(response.message)
 
      self.attach_in_progress = False
      return response
    
    def on_mqtt_message(self, client, userdata, msg):
     
     self.get_logger().info(f"RAW MQTT: {msg.topic} {msg.payload}")
 
     if msg.topic == self.mqtt_fb_ir:
        self.get_logger().info(f"{msg.topic}")
        try:
            data = json.loads(msg.payload.decode())
            self.ir_state = int(data.get("ir"))
        except Exception as e:
            self.get_logger().error(f"IR parse error: {e}")
   
   
    def _attach_watchdog(self):

      if not self.attach_in_progress:
        return

    # initialize timer once
    

      elapsed = (
        self.get_clock().now() - self.attach_start_time
         ).nanoseconds / 1e9
      self.get_logger().info(f"{self.ir_state}")
    # -------- SUCCESS --------
      if self.ir_state == 0:
        msg = String()
        msg.data = "ATTACHED"
        self.attach_status_pub.publish(msg)

        self.attach_in_progress = False
        self.attach_start_time = None
        return

    # -------- TIMEOUT --------
      if elapsed >= 20.0:
        self.set_electromagnet(0)

        msg = String()
        msg.data = "FAILED"
        self.attach_status_pub.publish(msg)

        self.attach_in_progress = False
        self.attach_start_time = None

    
   


    

def main():
    rclpy.init()
    node = Ros2MqttBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
