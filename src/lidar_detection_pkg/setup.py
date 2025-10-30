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
    ],
    install_requires=['setuptools','scikit-learn','numpy'],
    zip_safe=True,
    maintainer='kzzazzk',
    maintainer_email='kzzazzk@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lidar_object_detector = lidar_detection_pkg.lidar_object_detector:main',
        ],
    },
)
