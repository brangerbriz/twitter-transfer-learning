const tf = require('@tensorflow/tfjs')

/**
 * Create char2id, id2char and vocab_size
 * from printable ascii characters.
 * @returns [Map<string, number>, Map<number, string>, number]
 */
function createDictionary() {

    const printable = ('\t\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRST'
                    + 'UVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~').split('')

    // prepend the null character
    printable.unshift('')

    const char2id = new Map()
    const id2char = new Map()
    const vocabSize = printable.length

    printable.forEach((char, i) => {
        char2id.set(char, i)
        id2char.set(i, char)
    })

    return [char2id, id2char, vocabSize]
}

const [ CHAR2ID, ID2CHAR, VOCABSIZE ] = createDictionary()

// encode text to array of integers with CHAR2ID
/**
 * 
 * @param {string} text 
 * @param {Map<string, number>} [char2id]
 * @returns number[]
 */
function encodeText(text, char2id) {
    const dict = char2id || CHAR2ID
    return text.split('').map(char => {
        const number = dict.get(char)
        return typeof number === 'undefined' ? 0 : number
    })
}

/**
 * Decode array of integers to text with ID2CHAR
 * @param {number[]} numbers 
 * @param {Map<number, string>} id2char 
 * @returns string
 */
function decodeText(numbers, id2char) {
    const dict = id2char || ID2CHAR
    const chars = []
    // when numbers was a map function it was returning only NaNs, wtf...
    numbers.forEach(number => {
        const char = dict.get(parseInt(number))
        chars.push((typeof char === 'undefined') ? '' : char)
    })
    return chars.join('')
}

// options = {
//     batchSize: 64,
//     seqLen: 64,
//     oneHotFeatures: false,
//     oneHotLabels: false,
//     valSplit: 0.0
// }
// batch generator for sequence
// ensures that batches generated are continuous along axis 1
// so that hidden states can be kept across batches and epochs
/**
 * 
 * @param {number[]} sequence 
 * @param {*} options 
 * @returns Generator
 */
function* batchGenerator(sequence, options) {
    const batchSize = options.batchSize || 64
    const seqLen = options.seqLen || 64
    const oneHotFeatures = options.oneHotFeatures || false
    const oneHotLabels = options.oneHotLabels || false
    const valSplit = options.valSplit || 0.0

    const numBatches = Math.floor((sequence.length - 1) / (batchSize * seqLen))

    if (numBatches == 0) {
        throw Error('No batches created. Use smaller batch size or sequence length.')
    }
    console.info(`number of batches: ${numBatches}`)
    const roundedLen = numBatches * batchSize * seqLen
    console.info(`effective text length: ${roundedLen}`)
    const part = sequence.slice(0, roundedLen)
    // console.log(part.length)
    // console.log(part[part.length - 3], part[part.length - 2], part[part.length - 1])

    const [x, y] = tf.tidy(() => {
        let x = tf.tensor(sequence.slice(0, roundedLen))
        if (options.oneHotFeatures) {
            x = tf.oneHot(tf.cast(x, 'int32'), VOCABSIZE)
            x = x.reshape([batchSize, numBatches * seqLen, VOCABSIZE])
        } else {
            x = x.reshape([batchSize, numBatches * seqLen])
        }    
        console.info(`x shape: ${x.shape}`)

        let y = tf.tensor(sequence.slice(1, roundedLen + 1))
        if (options.oneHotLabels) {
            y = tf.oneHot(tf.cast(y, 'int32'), VOCABSIZE)
            y = y.reshape([batchSize, numBatches * seqLen, VOCABSIZE])
        } else {
            y = y.reshape([batchSize, numBatches * seqLen])
        }    
        console.info(`y shape: ${x.shape}`)
        return [x, y]
    })
    
    let epoch = 0
    const axis = 1
    const xEpoch = x.split(numBatches, axis)
    const yEpoch = y.split(numBatches, axis)
    x.dispose()
    y.dispose()

    while (true) {
        for (let i = 0; i < numBatches; i++) {
            yield [xEpoch[i], yEpoch[i], epoch]
        }
        console.info(`epoch ${epoch} finished`)
        epoch++
    }
}

// truncated weight random choice
function sampleFromProbs(probs, topN) {
    
    topN = topN || 10
    // probs is a Float32Array, so we will copy it manually
    const copy = []
    probs.forEach(prob => copy.push(prob))

    // now that it is a regular array we can use the JSON hack to copy it again
    const sorted = JSON.parse(JSON.stringify(copy))
    sorted.sort((a, b) => b - a)

    const truncated = sorted.slice(0, topN)
    
    // zero out all probability values that didn't make the topN
    copy.forEach((prob, i) => {
        if (!truncated.includes(prob)) copy[i] = 0
    })

    const sum = copy.reduce((a, b) => a + b, 0)
    const rescaled = copy.map(prob => prob /= sum)

    return SJS.Discrete(rescaled).draw()
}

async function generateText(model, seed, length, topN) {
    topN = topN || 10
    length = length || 512
    console.info(`generating ${length} characters from top ${topN} choices.`)
    console.info(`generating with seed: ${seed}`)
    let generated = seed
    let encoded = encodeText(seed)
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
        const sample = sampleFromProbs(probs, topN)
        generated += ID2CHAR.get(sample)
        nextIndex = sample
    }

    return generated
  }
  
function updateModelArchitecture(model, options) {
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
  
    const updatedModel = tf.Sequential.fromConfig(tf.Sequential, config)
    // this line matters, without it weights differ...
    updatedModel.setWeights(model.getWeights())
    return updatedModel
}

// load data from text file
async function loadData(path) {
    const text = await fetch(path).then(res => res.text())
    const encoded = utils.encodeText(text)
    return [text, encoded]
}
  
// load data using a tweet-server
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

async function fineTuneModel(model, numEpochs, batchSize, trainGenerator, valGenerator) {
    const histories = []
    model.resetStates()
    let lastEpoch = 0
    // epochs
    while (true) {
        const [x, y, epoch] = trainGenerator.next().value
        // console.log(tf.memory())
        const history = await model.fit(x, y, {
            batchSize: batchSize,
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
            const eval = await model.evaluate(x, y, { batchSize: batchSize })
            const valLoss = eval.dataSync()[0]
            console.log(`Epoch: ${epoch}, Training loss: ${history.history.loss[0]}, Validation loss: ${valLoss}`)
            histories.push(history)
            // NOTE: Don't forget to reset states on each epoch! This must be done manually!
            model.resetStates()
            lastEpoch = epoch
            x.dispose()
            y.dispose()
        }
    
        if (epoch == numEpochs) {
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
module.exports = {
    loadData,
    loadTwitterData,
    updateModelArchitecture,
    fineTuneModel,
    generateText,
    createDictionary,
    encodeText,
    decodeText,
    batchGenerator,
    sampleFromProbs,
    ID2CHAR
}
