#!/usr/bin/env python

__author__ = "Li-Yu Lin"
__contact__ = "liyu8561501@gmail.com"

from re import M
import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import std_msgs.msg as std_msgs
from geometry_msgs.msg import  TransformStamped
from compute_flowpipes import get_flowpipes
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class CameraNetworkVisualizer(Node):
    def __init__(self):
        super().__init__("visualizer_camera_network")
        # Configure subscritpions
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Initialize the transform broadcaster
        self.flowpipes = get_flowpipes()
        # with open('flowpipes.npy', 'rb') as f:
        #     self.flowpipes = np.load(f)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.pcd_publisher = self.create_publisher(Marker, '/flowpipes', 10)
        self.points = []

    def cmdloop_callback(self):
        self.points = []
        for i in range(30):
            for j in range(self.flowpipes[i].shape[0]):
                point =np.zeros((3,1))
                point[0] = self.flowpipes[i][j,0]
                point[1] = self.flowpipes[i][j,1]
                point[2] = self.flowpipes[i][j,2]
                self.points.append(point)

        self.points = np.array(self.points).reshape(len(self.points),3)
        self.pcd = convex_hull(self.points)
        print("publishing")
        self.pcd_publisher.publish(self.pcd)
    
def convex_hull(points):
    marker = Marker()
    marker.header.frame_id = 'odom'
    marker.color.b = 0.3 #0.3 #0.3
    marker.color.r = 1.0 #1.0
    marker.color.g = 0.8 #0.3 #0.8 #0.1
    marker.color.a = 0.6
    marker.type = Marker.LINE_STRIP
    marker.scale.x = 0.01
    marker.pose.orientation.w = 1.0
    marker.points = []
    for i in range(points.shape[0]):
        point = Point()
        point.x = points[i, 0]
        point.y = points[i, 1]
        point.z = points[i, 2]
        marker.points.append(point)
    return marker
       
def main(args=None):
    rclpy.init(args=args)
    cam_visualizer = CameraNetworkVisualizer()
    rclpy.spin(cam_visualizer)
    cam_visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()