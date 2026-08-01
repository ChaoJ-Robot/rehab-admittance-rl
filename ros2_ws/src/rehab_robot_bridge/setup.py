from setuptools import find_packages, setup

package_name = "rehab_robot_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "robot_bridge_node = rehab_robot_bridge.node:main",
        ],
    },
)
