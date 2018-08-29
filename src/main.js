// tfjs must be at least v0.12.6 which is needed for stateful RNNs
const tf = require('@tensorflow/tfjs')
const utils = require('./src/utils')
const util = require('util')
const fs = require('fs')

async function main(){

  const [text, data] = await loadData()
  const options = {
    batchSize: 128,
    valSplit: 0,
    oneHotLabels: true
  }
  
  const trainGenerator = utils.batchGenerator(data, options)
  const [ xBatch, yBatch ] = trainGenerator.next().value
  // for some reason, this length 130749 for brannondorsey.txt while the python version length is 130560...
  console.log(text.length)
  
  let model = await loadModel()
  let inferenceModel = buildInferenceModel(model)
  console.log(inferenceModel.getWeights()[0].dataSync())
  const seed = "This is a seed message. I hope that it is long enough. If not I don't know what to do."
  const generated = generateText(inferenceModel, seed)
  console.log(generated)

  // NOTE: Don't forget to reset states on each epoch! This must be done manually!
  // with model.resetStates()

}

function generateText(model, seed, length, topN) {
  topN = topN || 10
  length = length || 512
  console.info(`generating ${length} characters from top ${topN} choices.`)
  console.info(`generating with seed: ${seed}`)
  let generated = seed
  let encoded = utils.encodeText(seed)
  model.resetStates()
  console.log(encoded.length)
  encoded.slice(0, encoded.length - 1).forEach(idx => {
    const x = tf.tensor([[idx]])
    // input shape (1, 1)
    // set internal states
    const pred = model.predict(x, { verbose: true })
    // console.log(pred.dataSync())
  })

  let nextIndex = encoded.length - 1
  for (let i = 0; i < length; i++) {
    const x = tf.tensor([[nextIndex]])
    // input shape (1, 1)
    const probs = (model.predict(x)).dataSync()
    // console.log(probs)
    // output shape: (1, 1, VOCABSIZE)
    const sample = SJS.Discrete(probs).draw()
    generated += utils.ID2CHAR.get(sample)
    nextIndex = sample
  }
  return generated
}

function buildInferenceModel(model, options) {
  options = options || {}
  const batchSize = options.batchSize || 1
  const seqLen = options.seqLen || 1
  const config = model.getConfig()
  config[0].config.batchInputShape = [batchSize, seqLen]
  model.trainable = false
  // COME BACK HERE
  const inferenceModel = tf.Sequential.fromConfig(tf.Sequential, config)
  // this line matters, without it weights differ...
  inferenceModel.setWeights(model.getWeights())
  inferenceModel.trainable = false
  return inferenceModel
}

async function loadModel() {
  const path = '../char-rnn-text-generation/checkpoints/base-model-10M/tfjs/model.json'
  return await tf.loadModel(path)
}

async function loadData() {
  const path = '/home/bbpwn2/Documents/Branger_Briz/bbchi-code/char-rnn-text-generation/data/brannondorsey.txt'
  const text = (await util.promisify(fs.readFile)(path)).toString()
  const encoded = utils.encodeText(text)
  return [text, encoded]
}

main().catch(console.error)
