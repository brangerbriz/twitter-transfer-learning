// tfjs must be at least v0.12.6 which is needed for stateful RNNs
const tf = require('@tensorflow/tfjs')
const utils = require('./src/utils')

require('@tensorflow/tfjs-node')
// tf.setBackend('cpu')
console.log(tf.getBackend())

const TWEET_SERVER = 'http://localhost:3000'

const BATCH_SIZE = 32 // 128
const SEQ_LEN = 64
const DROPOUT = 0.1
const FINETUNE_EPOCHS = 1 // 10
const VAL_SPLIT = 0.2

// -----------------------------------------------------------------------------
const app = new Vue({
    el:'#app',
    data:{
      twitter: {
        user: 'barackobama',
        status: 'Click button to download a user\s tweets.'
      },
      data: {
          user: null,
          data: null
      },
      model: {
          name: null,
          path: null,
          model: null,
          training: false,
          status: 'Select a model to use. Training "base-model" with twitter data will create a new model.'
      },
      models: [],
      generatedTweets: []
    },
    mounted: async function (){
        const models = await tf.io.listModels()
        this.models = Object.keys(models).map(path => {
            return {
                path: path,
                name: path.split('//')[1]
            }
        })

        if (!this.models.map(m => m.name).includes('base-model')) {
            await this.loadModel('./checkpoints/base-model/tfjs/model.json')
            await this.model.model.save('indexeddb://base-model')
            this.models.push({
                name: 'base-model',
                path: 'indexeddb://base-model'
            })
            this.model.path = 'indexeddb://base-model'
        }


    },
    methods: {
        async downloadTweets() {
            console.log(this.twitter.user)

            this.twitter.status = `Downloading twitter data for ${this.twitter.user}...`
            try {
                const [text, data] = await utils.loadTwitterData(this.twitter.user, TWEET_SERVER)
                this.data.data = data
                this.data.user = this.twitter.user
                this.twitter.status = `Downloaded twitter data for ${this.twitter.user}`
            } catch (err) {
                console.error(err)
                this.twitter.status = `Error downloading twitter data for ${this.twitter.user}`
            }
        },
        async loadModel(path) {
            this.model.status = `Loading model from "${path}"...`
            try {
                this.model.model = await tf.loadModel(path)
                this.model.path = path
                this.model.name = path.split('//')[1]
                this.model.status = `Model loaded from "${path}"`
            } catch (err) {
                console.error(err)
                this.model.model = null
                this.model.path = null
                this.model.status = `Error loading model from "${path}"`
            }
        },
        async train() {

            if (this.model.model && this.data.data) {
                this.model.training = true
                this.model.status = 'Updating model architecture...'
                this.model.model = utils.updateModelArchitecture(this.model.model, { batchSize: BATCH_SIZE, seqLen: SEQ_LEN, dropout: DROPOUT })
                this.model.trainable = true
                this.model.model.compile({ optimizer: 'rmsprop', loss: 'categoricalCrossentropy', metrics: 'categoricalAccuracy' })
                this.model.status = 'Training model...'

                const options = {
                     batchSize: BATCH_SIZE,
                     seqLen: SEQ_LEN,
                     oneHotLabels: true
                }

                const valSplitIndex = Math.floor(this.data.data.length * VAL_SPLIT)
                const valGenerator = utils.batchGenerator(this.data.data.slice(0, valSplitIndex), options)
                const trainGenerator = utils.batchGenerator(this.data.data.slice(valSplitIndex), options)

                const histories = await utils.fineTuneModel(this.model.model, FINETUNE_EPOCHS, BATCH_SIZE, trainGenerator, valGenerator)

                if (this.model.name !== this.twitter.user) {
                    console.log('this is the base model')

                    const newModel = {
                        name: this.twitter.user,
                        path: `indexeddb://${this.twitter.user}`,
                    }
                    this.models.push(newModel)
                    this.model.path = newModel.path
                    this.model.name = newModel.name
                }

                this.model.status = `Saving trained model to ${this.model.path}`
                await this.model.model.save(this.model.path)
                this.model.status = `Saved ${this.model.path}`

                this.model.training = false
            }
        },
        async generate() {

            if (this.model.model) {
                this.model.status = 'Updating model architecture...'
                let inferenceModel = utils.updateModelArchitecture(this.model.model)
                inferenceModel.trainable = false
                const seed = "I'm feeling great today how are you?"
                this.model.status = `Generating text using ${this.model.path}`
                const generated = await utils.generateText(inferenceModel, seed, 512, 3)
                
                const tweets = generated.split('\n')

                // remove the first and last tweets, as they usually are garbage
                tweets.shift()
                tweets.pop()

                this.generatedTweets = tweets
            }
        }
    }
})
