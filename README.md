# Rubik's Cube State Detection using ML

<img src='doc/appdemo.gif' title='app demo' style='max-width:320px'></img> 

This repository is still in progress, currently showcases how the model detects the cube and translates the stickers in order to a stored state. Future works include finalizing the collection process and improving the robustness of the model. (Currently detects orange as red on my cube, will improve the dataset generation process)

There will be a more technical blog post explaining how I built this app, from dataset generation to model training, hopefully soon!

## Getting Started

1. clone the repository and enter the directory

```shell
git clone https://github.com/ammaraljerwi/RubiksDetector.git
cd RubiksDetector
```

2. Initialize a virtual environment either using `venv` or `conda`. 

```shellw
conda create -n RubiksDetector python=3.12
conda activate RubiksDetector
```

3. Install the requirements file
```shell
pip install -r requirements.txt
```

4. Run the main file!
```shell
python main.py
```

5. Press `Q` on the frame screen to exit the app

