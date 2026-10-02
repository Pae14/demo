from setuptools import find_packages, setup

package_name = 'vb_tactile_sensors'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: ['*.pt', '*.ckpt', '*.rviz'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'gs_image = vb_tactile_sensors.gs_pub_image:main',
		'gs_image_depth = vb_tactile_sensors.gs_pub_img_depth:main',
		'digit_image = vb_tactile_sensors.digit_pub_image:main',
		'digit_image_depth = vb_tactile_sensors.digit_pub_img_depth:main',
        ],
    },
)
