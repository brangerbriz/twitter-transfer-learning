// tfjs must be at least v0.12.6 which is needed for stateful RNNs
const tf = require('@tensorflow/tfjs')
const fs = require('fs')
const path = require('path')
const utils = require('./utils')

require('@tensorflow/tfjs-node')
console.log(tf.getBackend())

const TWITTER_USER = 'realdonaldtrump'
const TWEET_SERVER = 'http://localhost:3000'

const BATCH_SIZE = 64 // 128
const SEQ_LEN = 64
const DROPOUT = 0.1
const FINETUNE_EPOCHS = 1
const VAL_SPLIT = 0.2

async function main() {

    console.log(`fetching tweets for user @${TWITTER_USER}`)
    const [text, data] = await utils.loadTwitterData(TWITTER_USER, TWEET_SERVER)
    console.log('download complete.')

    const options = {
        batchSize: BATCH_SIZE,
        seqLen: SEQ_LEN,
        oneHotLabels: true
    }

    const valSplitIndex = Math.floor(data.length * VAL_SPLIT)
    const valGenerator = utils.batchGenerator(data.slice(0, valSplitIndex), options)
    const trainGenerator = utils.batchGenerator(data.slice(valSplitIndex), options)
    const modelPath = 'file://' + path.resolve(__dirname, '..', 'checkpoints', 'base-model', 'tfjs', 'model.json')
    let model = await tf.loadModel(modelPath)
    model = utils.updateModelArchitecture(model, { batchSize: BATCH_SIZE, seqLen: SEQ_LEN, dropout: DROPOUT })
    model.trainable = true

    // Fine tuning/transfer learning
    model.compile({ optimizer: 'rmsprop', loss: 'categoricalCrossentropy' })
    await utils.fineTuneModel(model, FINETUNE_EPOCHS, BATCH_SIZE, trainGenerator, valGenerator)
    
    // save the model for re-use
    const saveDir = path.resolve(__dirname, '..', 'checkpoints', TWITTER_USER)
    fs.mkdirSync(saveDir)
    await model.save(`file://${ path.join(saveDir, 'model.json') }`)

    let inferenceModel = utils.updateModelArchitecture(model)
    model.trainable = false

    const seed = "This is a seed sentence."
    const generated = await utils.generateText(inferenceModel, seed, 2048, 3)
    console.log(generated)
}

main().catch(console.error)
