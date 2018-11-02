#!/usr/bin/env node
const tf = require('@tensorflow/tfjs')
const fs = require('fs')
const path = require('path')
const utils = require('../src/utils')

// exit if the twitter-user parameter isn't included
if (process.argv[2] == null) {
    console.error(`usage: ${path.basename(process.argv[1])} twitter-user`)
    process.exit(1)
}

require('@tensorflow/tfjs-node')
console.log(`using tfjs backend "${tf.getBackend()}"`)

// remove the leading @ character if it exists
const TWITTER_USER = process.argv[2].replace(/^@/, '')
const TWEET_SERVER = 'http://localhost:3000'

const BATCH_SIZE = 64
const SEQ_LEN = 64
const DROPOUT = 0.0
const OPTIMIZER = 'adam'
const FINETUNE_EPOCHS = 10
const VAL_SPLIT = 0.2

async function main() {

    console.log(`fetching tweets for user @${TWITTER_USER}`)
    let text, data
    try {
        [text, data] = await utils.loadTwitterData(TWITTER_USER, TWEET_SERVER)
    } catch(err) {
        console.error('Error downloading tweets.')
        if (err.message) console.error(err.message)
        process.exit(1)
    }
    console.log('download complete.')

    // these options will be reused between several of the utility functions
    const options = {
        batchSize: BATCH_SIZE,
        seqLen: SEQ_LEN,
        dropout: DROPOUT,
        oneHotLabels: true
    }

    const valSplitIndex = Math.floor(data.length * VAL_SPLIT)
    const valGenerator = utils.batchGenerator(data.slice(0, valSplitIndex), 
                                              options)
    const trainGenerator = utils.batchGenerator(data.slice(valSplitIndex), 
                                                options)

    const modelPath = 'file://' + path.resolve(__dirname, 
                                               '..', 
                                               'checkpoints', 
                                               'base-model', 
                                               'tfjs', 
                                               'model.json')
    let model = await tf.loadModel(modelPath)
    // update the model architecture to use the BATCH_SIZE and SEQ_LEN
    // we've chosen for the fine-tune process.
    model = utils.updateModelArchitecture(model, options)
    model.trainable = true
    model.compile({ optimizer: OPTIMIZER, loss: 'categoricalCrossentropy' })

    // Fine-tune the model using transfer learning
    await utils.fineTuneModel(model, 
                              FINETUNE_EPOCHS, 
                              BATCH_SIZE, 
                              trainGenerator, 
                              valGenerator)

    // save the model in checkpoints/TWITTER_USER
    const saveDir = path.resolve(__dirname, '..', 'checkpoints', TWITTER_USER)
    if(!fs.existsSync(saveDir)) fs.mkdirSync(saveDir)
    await model.save(`file://${ path.join(saveDir, 'tfjs') }`)

    // we'll update the model architecture one more time, this time for
    // inference. We set both the BATCH_SIZE and SEQ_LEN to 1 and make
    // the model weights untrainable.
    let inferenceModel = utils.updateModelArchitecture(model)
    model.trainable = false

    // Generate 2048 characters using the fine-tuned model.
    const seed = "This is a seed sentence."
    const generated = await utils.generateText(inferenceModel, seed, 2048, 5)
    console.log(generated)
}

main().catch(console.error)
