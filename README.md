# Conflicting bundles
This is the original implementation of the conflicting bundles [paper](https://arxiv.org/abs/2011.02956).

    @inproceedings{peer2021conflicting,
      title={Conflicting Bundles: Adapting Architectures Towards the Improved Training of Deep Neural Networks},
      author={Peer, David and Stabinger, Sebastian and Rodriguez-Sanchez, Antonio},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
      pages={256--265},
      year={2021}
    }

# Dependencies
For this installation we assume that python3, pip3 and all nvidia drivers
(GPU support) are already installed. Then execute the following
to create a virtual environment and install all necessary dependencies:

1. Create virtual environment: ```python3 -m venv env```
2. Activate venv: ```source env/bin/activate```
3. Update your pip installation: ```pip3 install --upgrade pip```
4. Install all requirements. Use requirements-gpu if a gpu is available, requirements-cpu otherwise: ```pip3 install -r requirements.txt```

Note: If the dataset is not available in your home dir, it will be downloaded 
automatically and can take a few minutes.


# Execute 
We provide a ```run.sh``` file to show how all experiments can be trained and evaluated.
Note that it is important to train the model to check reproducibility 
and therefore, no checkpoints are provided.

## Training
Models are either trained on a single or on multiple gpu's. 
A checkpoint is created after each epoch to be able to evaluate the test-accuracy 
and to evaluate the bundle entropy. Therefore, 
about 2TB of storage are needed. Details of all parameters that are available 
can be found in ```config.py```.

## Evaluation
To evaluate a model trained with fixed architecture setups or for a model 
pruned with auto-tune, the ```evaluate.py``` script can be used. Results are 
written into a csv files to e.g. create graphs automatically or also 
tensorboard can be used to show all results (gradients, bundle entropy etc.).
Per default only the last layer L is evaluated. To evaluate the bundle 
entropy for each layer set ```--all_conflict_layers="True" ```.
Details of all parameters that are available can be found in ```config.py```.

## Modification 

This project is a fork and is being modified by implementing *Layerwise Batch Entropy* (LBE) as a pruning criterion from this [paper](https://arxiv.org/pdf/2208.01134.pdf) 


We changed the version of tensorflow-gpu to 2.4.0 from 2.3.1 and added keras to `requirements.txt` in order to make the project compatible with an rtx 3090 card
