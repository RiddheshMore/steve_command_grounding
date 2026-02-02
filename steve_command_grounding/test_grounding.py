#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import time

class CommandTester(Node):
    def __init__(self):
        super().__init__('command_tester')
        self.publisher = self.create_publisher(String, '/command', 10)
        self.subscription = self.create_subscription(
            PoseStamped,
            '/navigation_goal',
            self.goal_callback,
            10
        )
        self.last_goal = None

    def goal_callback(self, msg):
        self.last_goal = msg
        self.get_logger().info(f"RECEIVED GOAL: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def send_command(self, text):
        msg = String()
        msg.data = text
        self.get_logger().info(f"SENDING COMMAND: {text}")
        self.publisher.publish(msg)

def test():
    rclpy.init()
    tester = CommandTester()
    
    # We need to run the command grounding node in the background or another terminal
    # For this test, we'll just wait a bit
    print("Run: ros2 run steve_command_grounding command_grounding_node")
    print("Then press Enter here to test.")
    input()
    
    test_commands = [
        "Go to the kitchen"
    ]
    
    for cmd in test_commands:
        tester.send_command(cmd)
        # Wait until goal reached
        print(f"Waiting for robot to reach '{cmd}'...")
        start_time = time.time()
        while time.time() - start_time < 60.0:
            rclpy.spin_once(tester, timeout_sec=0.1)
            if tester.last_goal:
                print(f"Test '{cmd}': SUCCESS (Goal Received)")
                break
        
        if not tester.last_goal:
            print(f"Test '{cmd}': FAILED (No goal received within 60s)")
        print("-" * 20)

    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    test()
