#!/usr/bin/env python
import rosbag
import csv
import glob
import os

class RosbagReader:
    def __init__(self, bag_file):
        self.bag_file = bag_file

    def extract_gps_data(self, msg):
        return [msg.header.stamp.to_sec(), msg.latitude, msg.longitude, msg.altitude]

    def extract_imu_data(self, msg):
        return [
            msg.header.stamp.to_sec(),
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ]

    def save_to_csv(self, data, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

    def read_rosbag(self):
        gps_data = [["time", "latitude", "longitude", "altitude"]]
        imu_data = [["time", "orientation_x", "orientation_y", "orientation_z", "orientation_w",
                     "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                     "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]]

        with rosbag.Bag(self.bag_file) as bag:
            for topic, msg, t in bag.read_messages(topics=['/gps/fix', '/imu/imu_uncompensated']):
                if topic == '/gps/fix':
                    gps_data.append(self.extract_gps_data(msg))
                elif topic == '/imu/imu_uncompensated':
                    imu_data.append(self.extract_imu_data(msg))

        self.save_to_csv(gps_data, "gps_data.csv")
        self.save_to_csv(imu_data, "imu_data.csv")

if __name__ == "__main__":
    directory = '/home/tides/Desktop/EECE7150/hw5/sol/'
    bag_files = glob.glob(os.path.join(directory, '*.bag'))
    if not bag_files:
        raise FileNotFoundError("No .bag files found in the specified directory.")

    bag_file_path = bag_files[0]
    print(f"Using ROS bag file: {bag_file_path}")

    rosbag_reader = RosbagReader(bag_file_path)
    rosbag_reader.read_rosbag()
    print("Data saved to 'gps_data.csv' and 'imu_data.csv'")
