from setuptools import setup, find_packages

setup(
    name="reservoir-compiler",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "matlabengine==24.1.2",
        "matplotlib==3.9.2",
        "numpy==2.1.1",
        "sympy==1.13.1",
        "networkx==3.3",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pylint",
            "black",
        ],
    },
)
