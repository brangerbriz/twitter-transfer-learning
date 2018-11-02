#!/usr/bin/env node

// tfjs must be at least v0.12.6 which is needed for stateful RNNs
const tf = require('@tensorflow/tfjs')
const hpjs = require('hyperparameters')
const path = require('path')
const fs = require('fs')
const Json2csvParser = require('json2csv').Parser
const utils = require('../src/utils')

require('@tensorflow/tfjs-node')
console.log(`using tfjs backend "${tf.getBackend()}"`)

const NUM_TRIALS = 60
const NUM_EPOCHS = 5
const EXPERIMENT_PATH = path.join('checkpoints', 'hyperparameter-search-60-trials-5-epochs-3')

const TWITTER_USER = 'brannondorsey'
const TWEET_SERVER = 'http://localhost:3000'
const VAL_SPLIT = 0.2

const SEARCH_SPACE = {
  batchSize: hpjs.choice([16, 32, 64]),
  seqLen: hpjs.choice([16, 32, 64, 128]),
  dropout: hpjs.choice([0.0, 0.1, 0.25]),
  optimizer: hpjs.choice(['rmsprop', 'adagrad', 'adadelta', 'adam'])
}

let trialNum = 1

async function trial(data) {

    const params = hpjs.sample.randomSample(SEARCH_SPACE)

    const valSplitIndex = Math.floor(data.length * VAL_SPLIT)
    const valGenerator = utils.batchGenerator(data.slice(0, valSplitIndex), { ...params, oneHotLabels: true })
    const trainGenerator = utils.batchGenerator(data.slice(valSplitIndex), { ...params, oneHotLabels: true })

    const modelPath = 'file://' + path.resolve(__dirname, 
                                              '..', 
                                              'checkpoints', 
                                              'base-model', 
                                              'tfjs', 
                                              'model.json')
    let model = await tf.loadModel(modelPath)
    model = utils.updateModelArchitecture(model, params)
    model.trainable = true

    const epochSeconds = []
    let epochStartTime = null

    const callbacks = {
        onEpochBegin: () => {
            epochStartTime = Date.now()
            console.log(`onEpochBegin: ${epochStartTime}`)
        },
        /*
        * onEpochEnd: Logs include `acc` and `loss`, and optionally include `valLoss`
        *   (if validation is enabled in `fit`), and `valAcc` (if validation and
        *   accuracy monitoring are enabled).
        */
        onEpochEnd: () => {
            console.log(`onEpochEnd: ${Date.now()}`)
            console.log(`diff: ${Date.now() - epochStartTime}`)
            console.log(`diff seconds: ${(Date.now() - epochStartTime) / 1000}`)

            epochSeconds.push((Date.now() - epochStartTime) / 1000)
            epochStartTime = null
        }
    }

    // Fine tuning/transfer learning
    model.compile({
        optimizer: params.optimizer,
        loss: 'categoricalCrossentropy', 
        metrics: 'categoricalAccuracy'
    })

    console.log('running trial with hyperparameters:')
    console.log(params)

    let losses
    
    try {
        losses = await utils.fineTuneModel(model, 
                                           NUM_EPOCHS, 
                                           params.batchSize, 
                                           trainGenerator, 
                                           valGenerator,
                                           callbacks)
    } catch (err) {
        console.error('error during model training:')
        console.error(err)
        return null
    }
    
    
    const trialDir = path.join(EXPERIMENT_PATH, trialNum.toString())
    if (!fs.existsSync(trialDir)) {
        fs.mkdirSync(trialDir)
    }

    await model.save('file://' + path.resolve(trialDir, 'model.json'))
    const result = {
        avgEpochSeconds: epochSeconds.length > 0 ?
            epochSeconds.reduce((total, val) => total + val) / epochSeconds.length : 0,
        params,
        losses,
        trialNum
    }
    trialNum++
    return result
}

async function main() {

    console.log(`fetching tweets for user @${TWITTER_USER}...`)
    let text, data
    try {
        [text, data] = await utils.loadTwitterData(TWITTER_USER, TWEET_SERVER)
    } catch (err) {
        console.error(`Error downloading tweets: ${err.message}`)
        console.error(`Is a tweet-server (https://github.com/brangerbriz/tweet-server) instance running at ${TWEET_SERVER}?`)
        process.exit(1)
    }
    console.log('download complete.')

    if (!fs.existsSync(EXPERIMENT_PATH)) {
        fs.mkdirSync(EXPERIMENT_PATH)
    }

    // { avgEpochSeconds: 0,
    //     params: 
    //      { batchSize: 128,
    //        seqLen: 64,
    //        dropout: 0.1,
    //        optimizer: 'adadelta' },
    //     losses: 
    //    { loss: [ 1.8399657011032104, 1.7735635042190552, 1.7588552236557007 ],
    //     valLoss: [ 1.7922641038894653, 1.6736118793487549, 1.7574846744537354 ] },
    //     trialNum: 1 }
      

    const trials = []
    for (let i = 1; i <= NUM_TRIALS; i++) {
        const result = await trial(data)
        if (result) {
            trials.push(result)
            fs.writeFileSync(path.resolve(EXPERIMENT_PATH, 'trials.json'), JSON.stringify(trials))
            writeTrialsCSV(path.resolve(EXPERIMENT_PATH, 'trials.csv'), trials)
        }
    }
}

function writeTrialsCSV(path, trials) {
   
   const copy = JSON.parse(JSON.stringify(trials))

   // TODO: For whatever reason, this sorting sometimes fails... wtf.
   copy.sort((a, b) => {
       return Math.min(...a.losses.valLoss) > Math.min(...b.losses.valLoss)
   })

   const fields = ['rank', 
                   'val_loss',
                   'train_loss',
                   'min_val_loss_epoch',
                   'min_train_loss_epoch', 
                   'num_epochs', 
                   'avg_epoch_seconds', 
                   'batch_size', 
                   'seq_len', 
                   'drop_rate', 
                   'optimizer']

   const entries = copy.map((trial, index) => {

        const valLoss = Math.min(...trial.losses.valLoss)
        const trainLoss = Math.min(...trial.losses.loss)
        return {
           "rank": index + 1,
           "val_loss": valLoss,
           "train_loss": trainLoss,
           "min_val_loss_epoch": trial.losses.valLoss.indexOf(valLoss) + 1,
           "min_train_loss_epoch": trial.losses.loss.indexOf(trainLoss) + 1, 
           "num_epochs": trial.losses.loss.length, 
           "avg_epoch_seconds": parseInt(trial.avgEpochSeconds), 
           "batch_size": trial.params.batchSize, 
           "seq_len": trial.params.seqLen, 
           "drop_rate": trial.params.dropout, 
           "optimizer": trial.params.optimizer
        }
   })

   const json2csvParser = new Json2csvParser({ fields })
   const csv = json2csvParser.parse(entries)
   fs.writeFileSync(path, csv)
}

main().catch(console.error)
