# MPSCNNfeeder
Parameter feeder for Metal Performance Shaders' CNN. All you need to do is to put the exported parameter file (hdf5 or dat) from Keras into the Xcode project.

* SlimMPSCNN.swift: Wrapper for MPS's CNN. It reads dat files as model's parameters. If dat files don't exist, it converts the hdf5 file to dat files.
* swapaxes.swift: Helper class that changes the input flat array from order of weights [kH kW iC oC] to MPS accepted order of weights i.e. [oC kH kW iC]
* training: Python file to train the model (MPS doesn's have a capability to train the model right now, it only predicts based on the trained parameters.)
  * mnist_keras.py: Training model used Keras that is used for the example projects below. It can export both hdf5 and dat file.
  * hdf5parser.py: Not necessarily required for the project (because swift code includes hdf5 parser inside), but just for the proof of concept to convert from hdf5 to dat file on Python.
* examples: sample mnist apps
  * mnist_MPS_using_dat: Project using dat files as parameters
  * mnist_MPS_using_hdf5: Project using hdf5 file as parameters
