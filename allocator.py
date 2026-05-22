#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from hb_interfaces.msg import Pose2D, Poses2D,BotCmd,BotCmdArray,Botarmcmd,PickDropTask,TaskDone
import heapq

IDLE = 0
BUSY = 1

UNASSIGNED = 0
ASSIGNED = 1
DONE = 2

RED = 0
GREEN = 1
BLUE = 2
MAX_SLOTS = 5
SPACING = 100
VALID_CRATE_IDS = {15,12,30}

class TaskAllocator(Node):

    def __init__(self):
        super().__init__('task_allocator')


        self.robots = {}


        self.crates = {}
         # --- GLOBAL PLANNER SETTINGS (50mm Grid) ---
        self.arena_size = 2500.0  
        self.cell_size = 50.0     
        self.grid_dim = int(self.arena_size / self.cell_size)
        
        # --- DOCKING ZONES ---
        # Format: Robot ID -> {"x": center_x, "y": center_y, "radius": radius_in_mm}
        self.robot_docks = {
            4: {"name": "Glacio",    "x": 870.0,  "y": 225.0, "radius": 90.0},
            0: {"name": "Crystal",   "x": 1220.0, "y": 225.0, "radius": 90.0},
            2: {"name": "Frostbite", "x": 1570.0, "y": 225.0, "radius": 90.0},
        }
        
        # Replace your old drop_zones dict with this:
        # Note: I used the center points of your old min/max boundaries
        self.drop_zones = {
            RED:   self.build_pyramid_zone(1215.0, 1215.0, 55.0),
            GREEN: self.build_pyramid_zone(820.0, 2017.0, 55.0),
            BLUE:  self.build_pyramid_zone(1616.0, 2017.0, 55.0),
        }
        
        self.pickup_zones = {
            "P1": {"xmin": 172.0, "xmax": 376.0, "ymin": 542.0, "ymax": 1095.0},
            "P2": {"xmin": 2062.0, "xmax": 2265.0, "ymin": 542.0, "ymax": 1095.0},
            "P3": {"xmin": 172.0, "xmax": 376.0, "ymin": 1342.0, "ymax": 1895.0},
            "P4": {"xmin": 2062.0, "xmax": 2265.0, "ymin": 1342.0, "ymax": 1895.0},
        }
        self.task_id_counter = 0
 
        self.bot_poses_sub=self.create_subscription(Poses2D,'/bot_pose',self.robot_callback,10) 
        self.crate_poses_sub=self.create_subscription(Poses2D,'/crate_pose',self.crates_callback,10)
        self.task_publishers = {
        0: self.create_publisher(PickDropTask, '/crystal/task', 10),
        4: self.create_publisher(PickDropTask, '/glacio/task', 10),
        2: self.create_publisher(PickDropTask, '/frostbite/task', 10),
         }
        self.create_subscription(  TaskDone,'/task_done', self.task_done_callback,10)
        self.robots_id={0,2,4}
        
    def distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def get_color(self, cid):
        return cid % 3
    
    def get_pickup_zone(self, x, y):
        """
        Checks if a given (x,y) coordinate falls within ANY of the 4 pickup zones.
        Returns the zone name ('P1', 'P2', 'P3', 'P4') or None if it's outside.
        """
        for zone_name, bounds in self.pickup_zones.items():
            if (bounds["xmin"] <= x <= bounds["xmax"]) and \
               (bounds["ymin"] <= y <= bounds["ymax"]):
                return zone_name
        
       
        return None  # Return None if it is not in any pickup zone
    
    
    
    def crates_callback(self, msg):
      for pose in msg.poses:   # each element is Pose2D
         
         cid = pose.id
         if cid not in VALID_CRATE_IDS:
            self.get_logger().warn(f"Ignoring invalid crate id {cid}")
            continue
         
         x = pose.x
         y = pose.y
         
         # 1. Get the specific zone name ("P1", "P2", "P3", "P4") or None
         zone_name = self.get_pickup_zone(x, y)
         
         if cid not in self.crates:
            # 2. Only add the crate if it is inside a valid zone
            if zone_name is not None:
                self.crates[cid] = {
                    "x": x,  
                    "y": y,
                    "color": self.get_color(cid),
                    "state": UNASSIGNED,
                    "zone": zone_name  # <--- NEW: Save the specific zone string here
                }
         else:
            # 3. Update the coordinates for known crates
            self.crates[cid]["x"] = x
            self.crates[cid]["y"] = y
            
            # Optional: Update the zone tracking if it gets bumped, 
            # but don't delete the zone if the robot is currently carrying it across the map!
            if zone_name is not None:
                self.crates[cid]["zone"] = zone_name

      self.try_allocate()
   
    def robot_callback(self, msg):

      

      for pose in msg.poses:
          rid = pose.id

            # allow only known robots
          if rid not in self.robots_id:
             continue

            # create robot first time
          if rid not in self.robots:
            self.robots[rid] = {
                "state": IDLE,
                "x": pose.x,
                "y": pose.y
            }
          else:
            self.robots[rid]["x"] = pose.x
            self.robots[rid]["y"] = pose.y

        

        
      
      self.try_allocate()

    def build_pyramid_zone(self, center_x, center_y, spacing):
        """
        Builds a 3-crate pyramid. 
        Level 0 = Ground. Level 1 = Stacked on top.
        """
        return {
            "slots": [
                # Slot 0: Base Left
                {"x": center_x - (spacing / 2.0), "y": center_y, "level": 0},
                # Slot 1: Base Right
                {"x": center_x + (spacing / 2.0), "y": center_y, "level": 0},
                # Slot 2: Top Center
                {"x": center_x, "y": center_y, "level": 1}
            ],
            "next_index": 0 # Tracks which slot to fill next
        }
   
   
      # =========================================================
    def task_done_callback(self, msg: TaskDone):

        rid = msg.robot_id
        cid = msg.crate_id

        self.robots[rid]["state"] = IDLE

        if msg.success:
            self.crates[cid]["state"] = DONE
        else:
            self.crates[cid]["state"] = UNASSIGNED

        self.try_allocate()
 
 
 
 
 
 
    def try_allocate(self):
       # 1. Who is available?
       idle = [r for r in self.robots if self.robots[r]["state"] == IDLE]
       free = [c for c in self.crates if self.crates[c]["state"] == UNASSIGNED]

       if not idle or not free:
         return

       # 2. Check what the BUSY robots are already doing
       targeted_zones = set()
       color_counts = {0: 0, 1: 0, 2: 0} # RED=0, GREEN=1, BLUE=2
       
       total_active_crates = sum(1 for c in self.crates.values() if c["state"] != DONE)

       for c_id, c_data in self.crates.items():
           if c_data["state"] == ASSIGNED:
               c_zone = c_data.get("zone")
               if c_zone is not None:
                   targeted_zones.add(c_zone)
               color_counts[c_data["color"]] += 1

       # 3. Calculate Distances with PENALTIES
       pairs = []
       for rid in idle:
         for cid in free:
            # Real physical distance
            base_dist = self.distance(
                self.robots[rid]["x"], self.robots[rid]["y"],
                self.crates[cid]["x"], self.crates[cid]["y"]
            )
            
            # Start the "Cost" equal to the real distance
            cost = base_dist
            
            c_zone = self.crates[cid].get("zone")
            c_color = self.crates[cid]["color"]

            # --- PENALTY 1: ZONE CROWDING ---
            # If the zone has a robot in it, add 3000mm of fake distance
            if total_active_crates > 3 and c_zone is not None and c_zone in targeted_zones:
                cost += 3000.0  

            # --- PENALTY 2: COLOR DISTRIBUTION ---
            # If 2 robots are already doing this color, add 3000mm of fake distance
            if color_counts[c_color] >= 2:
                cost += 3000.0  

            # Save the cost, robot, and crate together
            pairs.append((cost, rid, cid))

       # 4. Sort by the new COST, not the real distance
       pairs.sort(key=lambda x: x[0])

       used_robots = set()
       used_crates = set()

       # 5. Assign the Tasks
       for cost, rid, cid in pairs:
         if rid in used_robots or cid in used_crates:
             continue

         self.assign_task(rid, cid)
 
         used_robots.add(rid)
         used_crates.add(cid)
         
         # Update tracking immediately so the next robot in this loop knows!
         c_zone = self.crates[cid].get("zone")
         if c_zone is not None:
             targeted_zones.add(c_zone)
         color_counts[self.crates[cid]["color"]] += 1
 
   
    def get_best_docking_pose(self, robot_x, robot_y, crate_x, crate_y, arm_offset):
        """
        Calculates 4 approach positions (N, S, E, W) around the crate.
        Filters out blocked paths and returns the closest safe pose.
        """
        # Define the 4 virtual ports based on the arm offset
        # Format: (stand_x, stand_y, facing_yaw_in_radians)
        candidates = [
            (crate_x - arm_offset, crate_y, 3*math.pi/2),           # Stand West, Face East
            (crate_x + arm_offset, crate_y, math.pi/2),       # Stand East, Face West
            (crate_x, crate_y - arm_offset, 0),     # Stand South, Face North
            (crate_x, crate_y + arm_offset, math.pi)     # Stand North, Face South
        ]

        best_pose = None
        min_cost = float('inf')

        for px, py, pyaw in candidates:
            # --- TEST 1: ARENA BOUNDARIES ---
            # Based on your homography code, the arena is ~2438mm x 2438mm.
            # We discard any port that forces the robot into the wall (e.g., < 150mm).
            if px < 150 or px > 2288 or py < 150 or py > 2288:
                continue # Skip! Too close to the wall.

            # --- TEST 2: CRATE COLLISIONS ---
            # Check if another crate is sitting on this exact port
            blocked = False
            for other_cid, other_crate in self.crates.items():
                # Don't check against the target crate itself
                if other_crate["x"] == crate_x and other_crate["y"] == crate_y:
                    continue
                
                # If a different crate is within 150mm of our standing spot, it's blocked
                dist_to_obstacle = self.distance(px, py, other_crate["x"], other_crate["y"])
                if dist_to_obstacle < 150.0: 
                    blocked = True
                    break
            
            if blocked:
                continue # Skip! Another crate is in the way.

            # --- WINNER SELECTION ---
            # If the port is safe, calculate how far the robot has to drive to reach it
            dist_to_travel = self.distance(robot_x, robot_y, px, py)
            
            # Save the one with the shortest driving distance
            if dist_to_travel < min_cost:
                min_cost = dist_to_travel
                best_pose = (px, py, pyaw)

        return best_pose # Returns (x, y, yaw) or None if totally trapped
   
   
    def get_best_drop_pose(self, robot_x, robot_y, drop_x, drop_y, arm_offset, target_color):
        """
        SUPER SIMPLE DROP POSE: 
        Only checks North and South. Picks the one closest to the robot.
        No obstacle checks. No ghost crates. 
        """
        candidates = [
            (drop_x, drop_y - arm_offset, 0.0),        # Stand South (Face East)
            (drop_x, drop_y + arm_offset, math.pi)     # Stand North (Face West)
        ]

        best_pose = None
        min_cost = float('inf')

        for px, py, pyaw in candidates:
            # 1. Basic Wall Check (Just so we don't drive out of the physical arena)
            if px < 150 or px > 2288 or py < 150 or py > 2288:
                continue 

            # 2. Pick the closest standing point to the robot
            dist_to_travel = self.distance(robot_x, robot_y, px, py)
            if dist_to_travel < min_cost:
                min_cost = dist_to_travel
                best_pose = (px, py, pyaw)

        return best_pose
    # A* GLOBAL PLANNER FUNCTIONS
    # =========================================================
    def generate_grid_map(self, active_robot_id, target_color=-1):
        grid = [[0 for _ in range(self.grid_dim)] for _ in range(self.grid_dim)]

        # 1. Block Docks
        for dock_id, dock_data in self.robot_docks.items():
            if dock_id == active_robot_id:
                continue 
            gx = max(0, min(self.grid_dim - 1, int(dock_data["x"] / self.cell_size)))
            gy = max(0, min(self.grid_dim - 1, int(dock_data["y"] / self.cell_size)))
            for dx in range(-2, 3): 
                for dy in range(-2, 3):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_dim and 0 <= ny < self.grid_dim:
                        grid[ny][nx] = 1

        # 2. Block Pyramids
        drop_centers = {0: (1215, 1215), 1: (820, 2017), 2: (1616, 2017)}
        for color, (cx, cy) in drop_centers.items():
            if color != target_color:
                
              gx = max(0, min(self.grid_dim - 1, int(cx / self.cell_size)))
              gy = max(0, min(self.grid_dim - 1, int(cy / self.cell_size)))
              for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_dim and 0 <= ny < self.grid_dim:
                        grid[ny][nx] = 1
            else:
                # --- SAME COLOR (Our Target Pyramid) ---
                # Only block the slots that have already been assigned to a robot
                zone = self.drop_zones[color]
                for i in range(zone["next_index"]):
                    slot = zone["slots"][i]
                    gx = max(0, min(self.grid_dim - 1, int(slot["x"] / self.cell_size)))
                    gy = max(0, min(self.grid_dim - 1, int(slot["y"] / self.cell_size)))
                    
                    # Block only this block, 1 in front (dy=1), and 1 in back (dy=-1).
                    # We keep dx=0 so we don't accidentally block the side-by-side slot!
                    for dy in [-1, 0, 1]:
                        nx, ny = gx, gy + dy
                        if 0 <= nx < self.grid_dim and 0 <= ny < self.grid_dim:
                            grid[ny][nx] = 1
                            
        
        return grid

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def find_path(self, start_mm, goal_mm, grid):
        start_grid = (int(start_mm[0] / self.cell_size), int(start_mm[1] / self.cell_size))
        goal_grid = (int(goal_mm[0] / self.cell_size), int(goal_mm[1] / self.cell_size))

        start_grid = (max(0, min(self.grid_dim - 1, start_grid[0])), max(0, min(self.grid_dim - 1, start_grid[1])))
        goal_grid = (max(0, min(self.grid_dim - 1, goal_grid[0])), max(0, min(self.grid_dim - 1, goal_grid[1])))

        if grid[start_grid[1]][start_grid[0]] == 1 or grid[goal_grid[1]][goal_grid[0]] == 1:
            return None

        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        neighbors = [(0,1,1), (1,0,1), (0,-1,1), (-1,0,1), (1,1,1.414), (1,-1,1.414), (-1,1,1.414), (-1,-1,1.414)]

        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            for dx, dy, cost in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid_dim and 0 <= neighbor[1] < self.grid_dim:
                    if grid[neighbor[1]][neighbor[0]] == 1: continue
                    tentative_g = g_score[current] + cost
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self.heuristic(neighbor, goal_grid)
                        heapq.heappush(open_set, (f_score, neighbor))
        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        path_mm = [(p[0] * self.cell_size + 25.0, p[1] * self.cell_size + 25.0) for p in path]
        return self.simplify_path(path_mm)

    def simplify_path(self, path):
        if len(path) < 3: return path
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            p1, p2, p3 = simplified[-1], path[i], path[i+1]
            cross_product = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
            if abs(cross_product) > 0.1: simplified.append(p2)
        simplified.append(path[-1])
        return simplified

    # =========================================================
    # TASK ASSIGNMENT WITH PATH PLANNING
    # =========================================================
    def assign_task(self, rid, cid):
        crate = self.crates[cid]
        robot = self.robots[rid]
        ARM_OFFSET = 150.0 

        # --- 1. PICKUP PHASE ---
        best_pickup = self.get_best_docking_pose(
            robot["x"], robot["y"], crate["x"], crate["y"], ARM_OFFSET
        )

        if best_pickup is None:
            self.get_logger().warn(f"Crate {cid} is trapped! Will try later.")
            return

        pickup_x, pickup_y, pickup_yaw = best_pickup

        # Run A* to find the path to the crate
        pickup_grid = self.generate_grid_map(rid, target_color=-1)
        pickup_path = self.find_path((robot["x"], robot["y"]), (pickup_x, pickup_y), pickup_grid)

        if not pickup_path:
            self.get_logger().warn(f"No path to crate {cid}!")
            return

        # --- 2. DROP PHASE ---
        zone = self.drop_zones[crate["color"]]
        
        if zone["next_index"] >= len(zone["slots"]):
            self.get_logger().warn(f"{crate['color']} Drop Zone is FULL!")
            return

        target_slot = zone["slots"][zone["next_index"]]
        drop_slot_x = target_slot["x"]
        drop_slot_y = target_slot["y"]
        task_level = target_slot["level"]

        
        
        # --- NEW: DYNAMIC ARM OFFSET ---
        # Adjust the Level 1 value based on your physical arm measurements!
        if task_level == 1:
            DROP_ARM_OFFSET = 100.0 # Example: Arm reaches further out when lifted, park closer!
        else:
            DROP_ARM_OFFSET = 160.0 # Floor level (same as pickup)

        best_drop = self.get_best_drop_pose(
            pickup_x, pickup_y, drop_slot_x, drop_slot_y, DROP_ARM_OFFSET, crate["color"]
        )
        if best_drop is None:
            self.get_logger().warn(f"Drop zone for {crate['color']} is currently blocked!")
            return

        drop_x, drop_y, drop_yaw = best_drop

        # Run A* to find the path to the drop zone
        drop_grid = self.generate_grid_map(rid, target_color=crate["color"])
        drop_path = self.find_path((pickup_x, pickup_y), (drop_x, drop_y), drop_grid)

        if not drop_path:
            self.get_logger().warn(f"No path from crate {cid} to drop zone!")
            return

        # --- 3. LOCK IT IN ---
        self.robots[rid]["state"] = BUSY
        self.crates[cid]["state"] = ASSIGNED
        zone["next_index"] += 1

        # --- 4. PUBLISH TASK ---
        self.task_id_counter += 1
        task = PickDropTask()
        task.task_id = self.task_id_counter
        task.robot_id = rid
        task.crate_id = cid
        
        task.pickup_x = pickup_x
        task.pickup_y = pickup_y
        task.pickup_yaw = pickup_yaw
        
        task.drop_x = drop_x
        task.drop_y = drop_y
        task.drop_yaw = drop_yaw
        
        task.task_level = task_level 

    
        # --- NEW: ATTACH A* PATHS TO MESSAGE ---
        task.pickup_path_x = [float(p[0]) for p in pickup_path]
        task.pickup_path_y = [float(p[1]) for p in pickup_path]
        
        task.drop_path_x = [float(p[0]) for p in drop_path]
        task.drop_path_y = [float(p[1]) for p in drop_path]
        self.task_publishers[rid].publish(task)
        self.get_logger().info(f"Assigned Robot {rid} to Crate {cid}. Level: {task_level}")

        
   
   
    

# =========================================================
def main():
    rclpy.init()
    node = TaskAllocator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
