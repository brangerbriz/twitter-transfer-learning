const tf = require('@tensorflow/tfjs')

// this is how you import the tensorflow backend. You don't have to save it.
// I also had trouble in Ubuntu 16.04 running tfjs-node at first. Followed
// the instructions to add a ppa: and re-install libstdc++6 and that worked.
// https://github.com/tensorflow/serving/issues/819
require('@tensorflow/tfjs-node')
// this is the same but for GPU. Must have CUDA 9.0 and CuDNN 7+
// require('@tensorflow/tfjs-node-gpu')


tf.setBackend('tensorflow')