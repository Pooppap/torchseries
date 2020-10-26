import setuptools

with open("README.md", "r") as r:
    long_description = r.read()

setuptools.setup(
    name="torchseries",
    version="0.1.6",
    author="CA1216",
    author_email="ca1216@ic.ac.uk",
    description="PyTorch Time-Series Augmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pooppap/torchseries.git",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.6.0",
        "numpy>=1.18.0",
        "scipy>=1.4.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
