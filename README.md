# ModOptima

Modoptima is a model optimization library that aims at democratizing model deployment for software or machine learning engineers who want to deploy their systems in production. It is currently in development and it aims at optimizing and running models in  least lines of code possible.

### Models Supported.
* Tiny YOLO V7 (Supported)
* Base YOLO V7 (Will be supported in future.)

### Prerequisites.

##### Install deepsparse
`pip install deepsparse[yolo]`


### Installation

The modoptima can be installed from the source at the moment but will be available via pip real soon.

#### Build from source
You can install modoptima by following below steps:

* Clone the repo.

`git clone https://github.com/VikasOjha666/modoptima.git`

* Navigate to the cloned folder.

`cd modoptima`

* Install by running the setup.

`python setup.py install`

#### Install via pip
Coming soon.

#### USAGE

##### Optimizing Model

Modoptima follows two types of optimization i.e Training and non training based the models. 
YOLO V7 optimization is based on pruning and quantization hence the model is pruned during training to reach a certain sparsity.
If quantization option is enabled then quantization aware training is done.

For each model supported there is a notebook showing the step by step process to optimize a model and run inference.
Please check notebooks folder in the repo.



