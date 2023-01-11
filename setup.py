import setuptools
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="modoptima",
    version="0.1.0",
    url="https://github.com/VikasOjha666/modoptima",
    author="Vikas Kumar Ojha",
    author_email="vikas.ojha894@gmail.com",
    description="ModOptima is a library that democratize deep learning model optimizations to developers. Letting them optimize their models with few lines of code while avoiding the headache of setting up dependencies.",
    long_description=long_description,
    packages=['modoptima','modoptima.Train','modoptima.Train.yolov7tiny','modoptima.Train.yolov7tiny.data','modoptima.Train.yolov7tiny.models','modoptima.Train.yolov7tiny.data',
       'modoptima.Train.yolov7tiny.utils','modoptima.Train.yolov7tiny.utils.wandb_logging'],
    install_requires=['sparseml','numpy','opencv-python','onnx','onnxruntime','tqdm'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    include_package_data=True,
    package_data={'': ['Train/yolov7tiny/data/*.*']},
)
