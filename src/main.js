// tfjs must be at least v0.12.6 which is needed for stateful RNNs
const tf = require('@tensorflow/tfjs')
const utils = require('./src/utils')

require('@tensorflow/tfjs-node-gpu')
// tf.setBackend('cpu')
console.log(tf.getBackend())

const TWEET_SERVER = 'http://localhost:3000'

const BATCH_SIZE = 32 // 128
const SEQ_LEN = 64
const DROPOUT = 0.1
const FINETUNE_EPOCHS = 10
const VAL_SPLIT = 0.2

async function main() {

  // const dataPath = 'data/text/realdonaldtrump.txt'
  // const [text, data] = await loadData(dataPath)

  const twitterUser = 'gray_gold'
  console.log(`fetching tweets for user @${twitterUser}`)
  const [text, data] = await loadTwitterData(twitterUser)
  console.log('download complete.')

  const options = {
    batchSize: BATCH_SIZE,
    seqLen: SEQ_LEN,
    oneHotLabels: true
  }

  const valSplitIndex = Math.floor(data.length * VAL_SPLIT)
  const valGenerator = utils.batchGenerator(data.slice(0, valSplitIndex), options)
  const trainGenerator = utils.batchGenerator(data.slice(valSplitIndex), options)

  // for some reason, this length 130749 for brannondorsey.txt while the python version length is 130560...
  console.log(text.length)

  // const modelPath = 'indexeddb://realdonaldtrump'  
  const modelPath = '../char-rnn-text-generation/checkpoints/base-model/tfjs/model.json'
  let model = await tf.loadModel(modelPath)
  model = buildInferenceModel(model, { batchSize: BATCH_SIZE, seqLen: SEQ_LEN, dropout: DROPOUT })
  model.trainable = true

  // Fine tuning/transfer learning
  model.compile({ optimizer: 'rmsprop', loss: 'categoricalCrossentropy', metrics: 'categoricalAccuracy' })
  const histories = await fineTuneModel(model, trainGenerator, valGenerator)
  const result = await model.save('indexeddb://gray_gold')

  let inferenceModel = buildInferenceModel(model)
  // console.log(result)

  const seed = "Fake news media"
  const generated = await generateText(inferenceModel, seed, 2048, 3)
  console.log(generated)
}

async function fineTuneModel(model, trainGenerator, valGenerator) {
  const histories = []
  model.resetStates()
  let lastEpoch = 0
  // epochs
  while (true) {
    const [x, y, epoch] = trainGenerator.next().value
    // console.log(tf.memory())
    const history = await model.fit(x, y, {
      batchSize: BATCH_SIZE,
      epochs: 1,
      shuffle: false,
      stepsPerEpoch: 1,
      callbacks: {
        onBatchBegin,
        onBatchEnd
      }
    })

    if (lastEpoch !== epoch) {
      const [x, y] = valGenerator.next().value
      console.log('evaluating model')
      const eval = await model.evaluate(x, y, { batchSize: BATCH_SIZE })
      const valLoss = eval.dataSync()[0]
      console.log(`Epoch: ${epoch}, Training loss: ${history.history.loss[0]}, Validation loss: ${valLoss}`)
      histories.push(history)
      // NOTE: Don't forget to reset states on each epoch! This must be done manually!
      model.resetStates()
      lastEpoch = epoch
      x.dispose()
      y.dispose()
    }
    if (epoch == FINETUNE_EPOCHS) {
      x.dispose()
      y.dispose()
      return histories
    }
    await tf.nextFrame()
  }
  function onBatchBegin() {
    // console.log('batch begin')
  }
  function onBatchEnd() {
    // console.log('batch end')
  }
}

async function generateText(model, seed, length, topN) {
  topN = topN || 10
  length = length || 512
  console.info(`generating ${length} characters from top ${topN} choices.`)
  console.info(`generating with seed: ${seed}`)
  let generated = seed
  let encoded = utils.encodeText(seed)
  model.resetStates()

  encoded.slice(0, encoded.length - 1).forEach(idx => {
    tf.tidy(() => {
      // input shape (1, 1)
      const x = tf.tensor([[idx]])
      // set internal states
      model.predict(x, { verbose: true })
    })
  })

  let nextIndex = encoded.length - 1
  for (let i = 0; i < length; i++) {
    const x = tf.tensor([[nextIndex]])
    // input shape (1, 1)
    const probsTensor = model.predict(x)
    // output shape: (1, 1, VOCABSIZE)
    x.dispose()
    const probs = await probsTensor.data()
    const sample = utils.sampleFromProbs(probs, topN)
    generated += utils.ID2CHAR.get(sample)
    nextIndex = sample
  }
  return generated
}

function buildInferenceModel(model, options) {

  options = options || {}
  const batchSize = options.batchSize || 1
  const seqLen = options.seqLen || 1
  const dropout = options.dropout
  const config = model.getConfig()
  config[0].config.batchInputShape = [batchSize, seqLen]

  config.forEach(layer => {
    if (dropout && layer.className == 'Dropout') {
      layer.config.rate = dropout
      console.log(`set dropout rate to ${dropout}`)
    }
  })

  const inferenceModel = tf.Sequential.fromConfig(tf.Sequential, config)
  // this line matters, without it weights differ...
  inferenceModel.setWeights(model.getWeights())
  inferenceModel.trainable = false
  return inferenceModel
}

async function loadData(path) {
  const text = await fetch(path).then(res => res.text())
  const encoded = utils.encodeText(text)
  return [text, encoded]
}

async function loadTwitterData(user) {

  const response = await fetch(`${TWEET_SERVER}/api/${user}`)
  if (response.ok) {
    const json = await response.json()
    if (json.tweets) {
      const text = json.tweets.join('\n')
      const encoded = utils.encodeText(text)
      return [text, encoded]
    }
  }

  throw TypeError(`Failed to load tweets for ${user}`)
}

main().catch(console.error)
