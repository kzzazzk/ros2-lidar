from setuptools import find_packages, setup

package_name = 'bag_reader_node_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'cv_bridge',
        'ultralytics'
    ],
    zip_safe=True,
    maintainer='arturo',
    maintainer_email='69714460+Jarturog@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_obstacle_detector = bag_reader_node_py.image_obstacle_detector:main',
        ],
    },
)
