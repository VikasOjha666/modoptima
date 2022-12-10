import setuptools

setuptools.setup(
    name="modoptima",
    version="0.1.0",
    url="https://github.com/VikasOjha666/modoptima",
    author="Vikas Kumar Ojha",
    author_email="vikas.ojha894@gmail.com",
    description="ModOptima is a library that democratize deep learning model optimizations to developers. Letting them optimize their models with few lines of code while avoiding the headache of setting up dependencies.",
    long_description="Description",
    packages=['modoptima','modoptima.Train','modoptima.Train.yolov7tiny','modoptima.Train.yolov7tiny.data','modoptima.Train.yolov7tiny.models','modoptima.Train.yolov7tiny.data',
       'modoptima.Train.yolov7tiny.utils','modoptima.Train.yolov7tiny.utils.wandb_logging'],
    install_requires=['sparseml','deepsparse'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    include_package_data=True,
    package_data={'': ['Train/yolov7tiny/data/*.*']},
)


#long_description=open('DESCRIPTION.rst').read()