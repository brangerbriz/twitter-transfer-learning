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
    // const [text, data] = await utils.loadData(dataPath)

    const twitterUser = 'gray_gold'
    console.log(`fetching tweets for user @${twitterUser}`)
    const [text, data] = await utils.loadTwitterData(twitterUser)
    console.log('download complete.')

    const options = {
        batchSize: BATCH_SIZE,
        seqLen: SEQ_LEN,
        oneHotLabels: true
    }

    const valSplitIndex = Math.floor(data.length * VAL_SPLIT)
    const valGenerator = utils.batchGenerator(data.slice(0, valSplitIndex), options)
    const trainGenerator = utils.batchGenerator(data.slice(valSplitIndex), options)

    // // for some reason, this length 130749 for brannondorsey.txt while the python version length is 130560...
    // console.log(text.length)

    // const modelPath = 'indexeddb://realdonaldtrump'  
    const modelPath = '../char-rnn-text-generation/checkpoints/base-model/tfjs/model.json'
    let model = await tf.loadModel(modelPath)
    model = utils.updateModelArchitecture(model, { batchSize: BATCH_SIZE, seqLen: SEQ_LEN, dropout: DROPOUT })
    model.trainable = true

    // Fine tuning/transfer learning
    model.compile({ optimizer: 'rmsprop', loss: 'categoricalCrossentropy', metrics: 'categoricalAccuracy' })
    const histories = await utils.fineTuneModel(model, FINETUNE_EPOCHS, BATCH_SIZE, trainGenerator, valGenerator)
    const result = await model.save('indexeddb://gray_gold')

    let inferenceModel = utils.updateModelArchitecture(model)
    model.trainable = false

    const seed = "Fake news media"
    const generated = await utils.generateText(inferenceModel, seed, 2048, 3)
    console.log(generated)
}

main().catch(console.error)
