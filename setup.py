from setuptools import setup, find_packages

setup(
    name="hopf_delay_toolbox_py",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    author="Borjan Milinkovic",
    author_email="[your-email@example.com]",
    description="A Python implementation of the Hopf Delay Toolbox for analysing brain network dynamics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/borjan/hopf_delay_toolbox_py",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
) 