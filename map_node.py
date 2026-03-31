import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf_transformations as tf_t
import numpy as np
from .mapping import QuadMap, MapType


class LidarMapper(Node):
    def __init__(self):
        super().__init__('lidar_mapper')
        
        # Initialize the QuadMap
        # Adjust these parameters based on your environment size
        self.quad_map = QuadMap(
            max_depth=6,  # Higher depth = higher resolution
            size=10.0,    # 10x10 meter map
            origin=np.array([0.0, 0.0])  # Centered at origin
        )
        
        # Setup TF buffer and listener for robot pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # QoS profile for subscriptions
        qos_profile = QoSProfile(depth=10)
        
        # Subscribe to laser scan data
        # Note: Based on your robot model, the topic might be /demo/scan
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/en613/scan',  # Adjust topic name if different
            self.scan_callback,
            qos_profile
        )
        
        # Publisher for occupancy grid (optional - for visualization)
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid,
            '/en613/map',
            qos_profile
        )
        
        # Timer for publishing occupancy grid
        self.map_publish_timer = self.create_timer(1.0, self.publish_occupancy_grid)
        
        self.get_logger().info("LidarMapper node started")

    def scan_callback(self, scan_msg):
        """
        Callback function for processing laser scan messages
        """
        try:
            # Get robot pose in odom frame
            robot_pose = self.get_robot_pose()
            if robot_pose is None:
                return
                
            # Process the scan data
            self.process_scan(scan_msg, robot_pose)
            
        except Exception as e:
            self.get_logger().error(f"Error in scan callback: {str(e)}")

    def get_robot_pose(self):
        """
        Get the current robot pose from TF
        """
        try:
            # Get transform from odom to laser_link
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'laser_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Extract position
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Extract orientation (yaw)
            quaternion = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            roll, pitch, yaw = tf_t.euler_from_quaternion(quaternion)
            
            return np.array([x, y, yaw])
            
        except tf2_ros.TransformException as ex:
            self.get_logger().debug(f'Could not transform: {ex}')
            return None

    def process_scan(self, scan_msg, robot_pose):
        """
        Process laser scan data and update the QuadMap
        """
        # Extract robot position and orientation
        robot_x, robot_y, robot_yaw = robot_pose
        robot_position = np.array([robot_x, robot_y])
        
        # Process each laser beam
        angle = scan_msg.angle_min
        for i, range_val in enumerate(scan_msg.ranges):
            # Skip invalid readings
            if (range_val < scan_msg.range_min or 
                range_val > scan_msg.range_max or 
                np.isnan(range_val) or 
                np.isinf(range_val)):
                angle += scan_msg.angle_increment
                continue
            
            # Calculate the global angle of this beam
            global_angle = robot_yaw + angle
            
            # Calculate the endpoint of the laser beam
            endpoint_x = robot_x + range_val * np.cos(global_angle)
            endpoint_y = robot_y + range_val * np.sin(global_angle)
            endpoint = np.array([endpoint_x, endpoint_y])
            
            # Update the map using ray casting
            self.quad_map.ray_update(robot_position, endpoint)
            
            # Move to next beam angle
            angle += scan_msg.angle_increment
        
        self.get_logger().debug("Updated map with new scan data")

    def publish_occupancy_grid(self):
        """
        Publish the current map as an OccupancyGrid message for visualization
        """
        try:
            # Get the occupancy grid data from QuadMap
            grid_data = self.quad_map.to_occupancygrid()
            
            # Create OccupancyGrid message
            occupancy_grid = OccupancyGrid()
            
            # Header
            occupancy_grid.header.stamp = self.get_clock().now().to_msg()
            occupancy_grid.header.frame_id = 'odom'
            
            # Map metadata
            occupancy_grid.info.resolution = self.quad_map.size / np.power(2, self.quad_map.max_depth)
            occupancy_grid.info.width = int(np.power(2, self.quad_map.max_depth))
            occupancy_grid.info.height = int(np.power(2, self.quad_map.max_depth))
            
            # Map origin (bottom-left corner)
            occupancy_grid.info.origin.position.x = (self.quad_map.root.origin[0] - 
                                                    self.quad_map.size / 2)
            occupancy_grid.info.origin.position.y = (self.quad_map.root.origin[1] - 
                                                    self.quad_map.size / 2)
            occupancy_grid.info.origin.position.z = 0.0
            occupancy_grid.info.origin.orientation.w = 1.0
            
            # Convert QuadMap data to ROS occupancy grid format
            # QuadMap: -1=unknown, 0=free, 1=occupied
            # ROS: -1=unknown, 0=free, 100=occupied
            ros_data = []
            # Map was showing up on the opposite side of the robot...
            # need to flip it for some reason
            grid_data_flipped = np.flipud(grid_data)
            flat_grid = grid_data_flipped.flatten()
            
            for cell in flat_grid:
                if cell == MapType.UNKNOWN:
                    ros_data.append(-1)
                elif cell == MapType.UNOCCUPIED:
                    ros_data.append(0)
                elif cell == MapType.OCCUPIED:
                    ros_data.append(100)
                else:
                    ros_data.append(-1)  # Default to unknown
            
            occupancy_grid.data = ros_data
            
            # Publish the occupancy grid
            self.occupancy_pub.publish(occupancy_grid)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing occupancy grid: {str(e)}")

    def get_map(self):
        """
        Public method to get the current QuadMap
        """
        return self.quad_map

def main(args=None):
    rclpy.init(args=args)
    
    lidar_mapper = LidarMapper()
    
    try:
        rclpy.spin(lidar_mapper)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
