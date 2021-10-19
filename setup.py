import setuptools

with open("README.txt", "r") as fn:
    long_description = fn.read()

setuptools.setup(
    name="PySonde",
    version="0.4.1",
    author="Christopher Phillips of UAH, Huntsville, Alabama",
    author_email="cephillips574@gmail.com",
    description="A python package for reading and analyzing several common weather balloon sounding formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sodoesaburningbus/pysonde",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    install_requires=["metpy>=1.0"]
)
