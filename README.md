# Audio Ad Detection in Radio Streams
This is a consulting project for Veritonic Inc. done as part of Insight Artificial Inetligence Fellowship.

## Installation
The code is written as a Python 3.6 package. For installing it on your computer, clone the repository, then edit the `WEIGHTS_FOLDER` variable in `confing.py` file to point to a location of your choice and place the saved model weights in it.
Finally use the terminal to install the package:
```bash
cd path/to/cloned/repository
pip install .
```
The package should now be avialable for use. You can import the package and use it to generate a prediction for some radio stream audio file as shown below:
```python
import adetector as adt

radio_stream_path =  '../Data/Z100 Stream Recording.mp3'
X = adt.core.audio2features(radio_stream_path)
timestamps, probs = adt.core.find_ads(X, T=0.85, n=10, show=True)
```
The `audio2features` function converts the audio file to an array of features which is then input to `find_ads` which returns an array of timestamps and a vector of ad probabilities corresponding to each timestamp. The argument `T` defines the threshold for detection, `n` defines a window size for the moving average which is done before the threshold is taken and when show is set to be `True` a graph of probability over time is showed with the threshold overlaid. 
