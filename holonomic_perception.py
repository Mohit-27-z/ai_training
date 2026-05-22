#!/usr/bin/env python3

import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from hb_interfaces.msg import Pose2D, Poses2D


class PoseDetector(Node):
    def __init__(self):
        super().__init__('localization_node')

        # CvBridge for image conversion
        self.bridge = CvBridge()

        # ---------- PARAMETERS ----------
        self.crates_marker_length = 0.045  # meters
        self.bots_marker_length = 0.105  # meters

        # ---------- ARUCO SETUP ----------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # ---------- TOPICS ----------
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.crate_poses_pub = self.create_publisher(Poses2D, '/crate_pose', 10)
        self.bot_poses_pub = self.create_publisher(Poses2D, '/bot_pose', 10)
        self.camera_height = 2.62          # meters (measure carefully)
        self.robot_marker_height = 0.09   # 9 cm
        self.crate_marker_height = 0.05   # 5 cm

          # ---------- CAMERA PARAMETERS ----------
        self.camera_matrix = np.array([
           [1004.13094, 0.0, 955.61117],
           [0.0, 1008.08423, 544.42723],
           [0.0, 0.0, 1.0]
            ], dtype=np.float64)
        self.dist_coeffs = np.array(
        [-0.066168, -0.000822, 0.001705, -0.007098, 0.000000],
         dtype=np.float64
        ).reshape(1, 5) # 1xN
        # ---------- HOMOGRAPHY ----------
        # World corner coordinates (e.g., in mm) for TL, TR, BL, BR
        self.corner_world_matrix = np.array(
            [
                (0.0, 0.0),        # TL
                (2.4384, 0.0),   
                (2.4384, 2.4384),  # BR
                 (0.0, 2.4384)],
            dtype=np.float32
        )
        self.H_matrix = None  # 3x3

        # ---------- STATE ----------
        self.position_bot = {}    # {id: {'position': [(x,y)], 'yaw_deg': float}}
        self.position_crate = {}  # {id: {'position': [(x,y)], 'yaw_deg': float}}
        self.get_logger().info('PoseDetector initialized')
        self.robot_ids = {0, 2, 4}   # example: three robots

   

    def image_callback(self, msg: Image):
        try:
            # Reset per-frame pose caches (avoids stale data)
            self.position_bot = {}
            self.position_crate = {}

            # Step 1: Convert ROS image -> OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Step 2: Undistort if calibration is available
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                undistorted_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
            else:
                undistorted_image = cv_image

            # Step 3: Grayscale
            gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

            # Step 4: Detect ArUco markers
            corners, ids, rejected = self.detector.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                self.get_logger().warn('No ArUco markers detected.')
                cv2.imshow('Detected Markers', undistorted_image)
                cv2.waitKey(1)
                return

            ids = ids.flatten()

            # Prepare to compute homography from four known reference IDs
            # order: 0:TL(id=1), 1:TR(id=3), 2:BL(id=5), 3:BR(id=7)
            corner_pixel_matrix = [None, None, None, None]

            # Cache centers for world mapping
            pixel_centers = {}  # {id: (cx, cy)}

            # First pass: collect homography correspondences and per-marker centers
            for marker_id, marker_corners in zip(ids, corners):
                pts = marker_corners.reshape((4, 2)).astype(np.float32)  # TL, TR, BR, BL per OpenCV
                tl, tr, br, bl = pts

                # References for homography
                if marker_id == 1:
                    corner_pixel_matrix[0] = tuple(tl)   # TL
                elif marker_id == 3:
                    corner_pixel_matrix[1] = tuple(tr)   # TR
                elif marker_id == 5:
                    corner_pixel_matrix[3] = tuple(bl)   # BL
                elif marker_id == 7:
                    corner_pixel_matrix[2] = tuple(br)   # BR

                # For all non-reference markers, record center for world mapping
                if marker_id not in [1, 3, 5, 7]:
                    cx = float(np.mean(pts[:, 0]))
                    cy = float(np.mean(pts[:, 1]))
                    pixel_centers[marker_id] = (cx, cy)
                
            # Compute homography once all four references are seen
            if None not in corner_pixel_matrix:
                src_pts = np.array(corner_pixel_matrix, dtype=np.float32)  # image
                dst_pts = self.corner_world_matrix                         # world
                H, mask = cv2.findHomography(src_pts, dst_pts, 0)          # exact 4-point method
                self.H_matrix = H
            world_nadir = None

            if self.H_matrix is not None:
               cx = float(self.camera_matrix[0, 2])
               cy = float(self.camera_matrix[1, 2])

               nadir_pix = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
               nadir_world = cv2.perspectiveTransform(nadir_pix, self.H_matrix)

               world_nadir = (
                 float(nadir_world[0, 0, 0]),
                 float(nadir_world[0, 0, 1])
                )
            # Second pass: estimate yaw (requires intrinsics) and map centers to world (requires H) 
            for marker_id, marker_corners in zip(ids, corners):
                if marker_id in [1, 3, 5, 7]:
                    continue  # skip reference markers for pose/center publishing

                pts = marker_corners.reshape((4, 2)).astype(np.float32)

                # 1) Estimate yaw in camera frame if intrinsics available
                yaw_deg = None
                
                    # Choose marker length per class
                marker_length = (
                         self.bots_marker_length
                         if marker_id in self.robot_ids
                         else self.crates_marker_length
                          )

                    # 3D object points of marker (square, Z=0)
                half = marker_length / 2.0
                objp = np.array([
                       [-half,  half, 0.0],
                       [ half,  half, 0.0],
                       [ half, -half, 0.0],
                       [-half, -half, 0.0],
                    ], dtype=np.float32)

                if self.H_matrix is not None:
                    # take two adjacent corners
                  p1 = pts[0]          # TL
                  p2 = pts[1]          # TR

                  corners_pix = np.array([p1, p2], dtype=np.float32).reshape(-1, 1, 2)
                  corners_world = cv2.perspectiveTransform(corners_pix, self.H_matrix)

                  x1, y1 = corners_world[0, 0]
                  x2, y2 = corners_world[1, 0]

                  yaw = math.atan2(y2 - y1, x2 - x1)
                  yaw_deg = (math.degrees(yaw) + 360) % 360
                

                # 2) Map pixel center to world using homography if available   
                world_xy = None
                

                if (
                  self.H_matrix is not None and
                  marker_id in pixel_centers and
                  world_nadir is not None
                ):

                  cx_pix, cy_pix = pixel_centers[marker_id]

                  pixel_point = np.array([[cx_pix, cy_pix]], dtype=np.float32).reshape(-1, 1, 2)
                  world_point = cv2.perspectiveTransform(pixel_point, self.H_matrix)

                  X0 = float(world_point[0, 0, 0])
                  Y0 = float(world_point[0, 0, 1])

                   # choose correct height
                  if marker_id in self.robot_ids:
                     h = self.robot_marker_height
                  else:
                     h = self.crate_marker_height

                  V = self.camera_height

                          # height scaling
                  s = (V - h) /V

                  Xn, Yn = world_nadir

                  Xc = Xn + s * (X0 - Xn)
                  Yc = Yn + s * (Y0 - Yn)

                  world_xy = [(Xc, Yc)]

                # Populate outputs if both components are available
                if yaw_deg is not None and world_xy is not None:
                    if marker_id in self.robot_ids:
                        self.position_bot[marker_id] = {'position': world_xy, 'yaw_deg': yaw_deg}
                    else:
                        self.position_crate[marker_id] = {'position': world_xy, 'yaw_deg': yaw_deg}

            # Publish if any
            if len(self.position_bot) > 0:
                self.publisher_bot_poses()
            if len(self.position_crate) > 0:
                self.publisher_crate_poses()

               # Draw on original image
            cv2.aruco.drawDetectedMarkers(undistorted_image, corners, ids)

              # Resize ONLY for display
            scale = 0.5
            small = cv2.resize(
               undistorted_image,
                None,
                fx=scale,
                fy=scale,
              interpolation=cv2.INTER_AREA
             )

            cv2.imshow('Detected Markers', small)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def publisher_crate_poses(self):
        crate_poses_msg = Poses2D()
        for m_id, crate_info in self.position_crate.items():
            x, y = crate_info['position'][0]
            yaw = crate_info['yaw_deg']

            crate_pose = Pose2D()
            crate_pose.id = int(m_id)
            crate_pose.x = float(x*1000)
            crate_pose.y = float(y*1000)
            crate_pose.w = float(yaw)

            crate_poses_msg.poses.append(crate_pose)

        self.crate_poses_pub.publish(crate_poses_msg)

    def publisher_bot_poses(self):
        bot_poses_msg = Poses2D()
        for m_id, bot_info in self.position_bot.items():
            x, y = bot_info['position'][0]
            yaw = bot_info['yaw_deg']

            bot_pose = Pose2D()
            bot_pose.id = int(m_id)
            bot_pose.x = float(x*1000)
            bot_pose.y = float(y*1000)
            bot_pose.w = float(yaw)

            bot_poses_msg.poses.append(bot_pose)

        self.bot_poses_pub.publish(bot_poses_msg)


def main(args=None):
    rclpy.init(args=args)
    pose_detector = PoseDetector()
    try:
        rclpy.spin(pose_detector)
    except KeyboardInterrupt:
        pass
    finally:
        pose_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
