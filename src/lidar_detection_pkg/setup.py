import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'lidar_detection_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*'))
    ],
    install_requires=[
        'setuptools',
        'torch',
        'setuptools',
        'numpy',
        'opencv-python',
        'cv_bridge',
        'ultralytics',
        'scikit-learn'
    ],
    zip_safe=True,
    maintainer='kzzazzk, arturo, yago, adrian',
    maintainer_email='kzzazzk@todo.todo, 69714460+Jarturog@users.noreply.github.com, yagobf.ybf@gmail.com, adrian.estoquera@example.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_obstacle_detector = lidar_detection_pkg.image_obstacle_detector:main',
            'lidar_object_detector = lidar_detection_pkg.lidar_object_detector:main',
            'fusion_data = lidar_detection_pkg.fusion_data:main',
            'data_snapshot_server = lidar_detection_pkg.data_snapshot_server:main',
        ],
    },
)
