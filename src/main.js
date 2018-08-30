// tfjs must be at least v0.12.6 which is needed for stateful RNNs
const tf = require('@tensorflow/tfjs')
const utils = require('./src/utils')
const util = require('util')
const fs = require('fs')

require('@tensorflow/tfjs-node-gpu')
// tf.setBackend('cpu')
console.log(tf.getBackend())

const BATCH_SIZE = 128
const FINETUNE_EPOCHS = 5
const VAL_SPLIT=0.2

async function main(){

  const [text, data] = await loadData()
  const options = {
    batchSize: BATCH_SIZE,
    oneHotLabels: true
  }
  
  const valSplitIndex = Math.floor(data.length * VAL_SPLIT)
  const valGenerator = utils.batchGenerator(data.slice(0, valSplitIndex), options)
  const trainGenerator = utils.batchGenerator(data.slice(valSplitIndex), options)
  const [ xBatch, yBatch ] = trainGenerator.next().value
  // for some reason, this length 130749 for brannondorsey.txt while the python version length is 130560...
  console.log(text.length)
  
  let model = await loadModel()
  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: 'categoricalAccuracy'})
  const histories = await fineTuneModel(model, trainGenerator, valGenerator)
  console.log(histories)
  
  let inferenceModel = buildInferenceModel(model)
  const seed = "@brannondorsey"
  const generated = generateText(inferenceModel, seed, 2048)
  console.log(generated)

  // NOTE: Don't forget to reset states on each epoch! This must be done manually!
  // with model.resetStates()

}

async function fineTuneModel(model, trainGenerator, valGenerator) {
  const histories = []
  model.resetStates()
  let lastEpoch = 0
  // epochs
  while (true) {
    const [x, y, epoch] = trainGenerator.next().value
    const history = await model.fit(x, y, { 
      batchSize: BATCH_SIZE, 
      epochs: 1,
      shuffle: false,
      stepsPerEpoch: 1,
      callbacks: {
        onBatchBegin,
        onBatchEnd
      } })
    console.log(epoch)
    if (lastEpoch !== epoch) {
      const [x, y] = valGenerator.next().value
      console.log('evaluating model')
      const eval = await model.evaluate(x, y, { batchSize: BATCH_SIZE })
      const valLoss = eval.dataSync()[0]
      console.log(`Epoch: ${epoch}, Training loss: ${history.history.loss[0]}, Validation loss: ${valLoss}`)
      histories.push(history)
      model.resetStates()
      lastEpoch = epoch
    }
    if (epoch == FINETUNE_EPOCHS) {
      x.dispose()
      y.dispose()
      return histories
    }
    await tf.nextFrame()
  }
  function onBatchBegin() {
    console.log('batch begin')
  }
  function onBatchEnd() {
    console.log('batch end')
  }
}

function generateText(model, seed, length, topN) {
  topN = topN || 10
  length = length || 512
  console.info(`generating ${length} characters from top ${topN} choices.`)
  console.info(`generating with seed: ${seed}`)
  let generated = seed
  let encoded = utils.encodeText(seed)
  model.resetStates()
  encoded.slice(0, encoded.length - 1).forEach(idx => {
    const x = tf.tensor([[idx]])
    // input shape (1, 1)
    // set internal states
    model.predict(x, { verbose: true })
    x.dispose()
  })

  let nextIndex = encoded.length - 1
  for (let i = 0; i < length; i++) {
    const x = tf.tensor([[nextIndex]])
    // input shape (1, 1)
    const probs = (model.predict(x)).dataSync()
    x.dispose()
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
  const path = '/home/bbpwn2/Documents/Branger_Briz/bbchi-code/char-rnn-text-generation/data/realdonaldtrump.txt'
  const text = (await util.promisify(fs.readFile)(path)).toString()
  const encoded = utils.encodeText(text)
  return [text, encoded]
}

main().catch(console.error)
