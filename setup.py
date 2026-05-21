from setuptools import setup, find_packages

setup(
    name="nautical-graph-toolkit",
    version="0.1.5",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "nautical_graph_toolkit": [
            "data/**/*.csv",
            "data/**/*.yml",
        ],
    },
    python_requires=">=3.11",
)
