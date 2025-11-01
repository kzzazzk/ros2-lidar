import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
import rosbag2_py
from std_msgs.msg import String


class SimpleBagReader(Node):

    def __init__(self):
        super().__init__('simple_bag_reader')
        self.reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri='my_bag',
            storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.reader.open(storage_options, converter_options)

        self.publisher = self.create_publisher(String, 'chatter', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        while self.reader.has_next():
            msg = self.reader.read_next()
            if msg[0] != 'chatter':
                continue
            self.publisher.publish(msg[1])
            self.get_logger().info('Publish serialized data to ' + msg[0])
            break


def main(args=None):
    try:
        with rclpy.init(args=args):
            sbr = SimpleBagReader()
            rclpy.spin(sbr)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
