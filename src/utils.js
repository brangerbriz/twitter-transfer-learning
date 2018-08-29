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

    let epoch = 0
    const axis = 1
    const xEpoch = x.split(numBatches, axis)
    const yEpoch = y.split(numBatches, axis)

    while (true) {
        for (let i = 0; i < numBatches; i++) {
            yield [xEpoch[i], yEpoch[i]]
        }
        console.info(`epoch ${epoch} finished`)
        epoch++
    }
}

// truncated weight random choice
function sampleFromProbs(probs, topN) {
    topN = topN || 10

}

module.exports = {
    createDictionary,
    encodeText,
    decodeText,
    batchGenerator,
    ID2CHAR
}

const message = 'This is a test message. 😎🎉Those were some Emoji.'
const encoded = encodeText(message)
console.log('message: ', message)
console.log('encoded: ', encoded.join(','))
console.log('decoded: ', decodeText(encoded))