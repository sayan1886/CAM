from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='CAM',
    version='1.3.0',
    description='A Class Activaton Mapping implementation Tensorflow and Keras',
    long_description='A Class Activaton Mapping e.g CAM, GRAD-CAM',
    url='https://github.com/sayan1886/CAM',
    author='Sayan Chatterjee',
    author_email='sayan1886@gmail.com',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    keywords='cam grad-cam tf keras xai',
    license='MIT',
    install_requires=['tensorflow', 'keras', 'opencv-python', 'imutils', 'numpy', 'scipy', 'pillow'],
)