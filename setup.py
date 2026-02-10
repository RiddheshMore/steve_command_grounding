import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'steve_command_grounding'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'requests', 'opencv-python', 'numpy', 'pillow'],
    zip_safe=True,
    maintainer='ritz',
    maintainer_email='riddheshmore311@gmail.com',
    description='Steve Command Grounding - NLP to Robot Actions with SAM3 Perception',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'command_grounding_node = steve_command_grounding.command_grounding_node:main',
            'steve_search_node = steve_command_grounding.steve_search_node:main',
            'command_tester = steve_command_grounding.test_grounding:test',
            'search_testing = steve_command_grounding.search_testing:main',
        ],
    },
)
