from setuptools import setup
from setuptools import find_packages

setup(name='adetector',
      version='0.1.0',
      description='Detecting audio ads in a radio stream',
      url='https://github.com/ohadmich/Adetector',
      author='Ohad Michel',
      author_email='ohadmich@gmail.com',
      license='MIT',
      packages=['adetector'],
      install_requires = [
      'numpy',
      'matplotlib',
      'librosa',
      'TensorFlow',
      'keras'],
      zip_safe=False)
