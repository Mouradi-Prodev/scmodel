
const tf = require('@tensorflow/tfjs')

require("tfjs-node-save");

const loadData = (csvPath) =>{
    const transform = ({ xs, ys}) => {
        const zeros = (new Array(7)).fill(0);
        return {
            xs : tf.tensor(xs, [28, 28, 3]),

            ys: tf.tensor1d(zeros.map((z, i) => {
                if (i === parseInt(ys)) {
                    return 1
                }
                return 0
            }))
        }
    };

    return tf.data
           .csv(csvPath, { columnConfigs: {label: {isLabel:true}}})
           .map(({ xs, ys }) =>{
            return {
                xs : Object.values(xs).map(x => x/255),
                ys : ys.label

            };

           })
           .map(transform)
           .shuffle(1000) // Shuffle the dataset with a buffer of 1000 elements
           .batch(4172)
};




async function train(model) {
    let data = loadData('file://./React Native App/SCDmodel/train.csv');
    await model.fitDataset(data, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs)=> {
                console.log(`Batch ${batch+1} - loss: ${logs.loss.toFixed(4)} , acc:${logs.acc.toFixed(4)}`);
            },
            onEpochBegin: async (epoch, logs,epochs=10) => {
                console.log(`Epoch ${epoch + 1} of ${epochs} ...`)
            },
            onEpochEnd: async (epoch, logs) => {
                console.log("Training Loss is : " ,logs.loss.toFixed(4))
                console.log(`training accuracy: ${logs.acc.toFixed(4)}`)
            }
        }
    });
    //EVALUATING
    let testData = loadData('file://./React Native App/SCDmodel/test.csv');
    const result = await model.evaluateDataset(testData);
    const testLoss = result[0].dataSync()[0];
    const testAcc = result[1].dataSync()[0];
    console.log(`test-set loss: ${testLoss.toFixed(4)}`);
    console.log(`test set accuracy: ${testAcc.toFixed(4)}`);
    //SAVE
    await model.save('file://./model');
    return 'done';
}

const buildModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({ inputShape:[28, 28, 3], filters: 32, kernelSize: 3, activation: 'relu'}));
    model.add(tf.layers.maxPool2d({ poolSize: 2}));

    model.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu'}));
    model.add(tf.layers.maxPool2d({ poolSize: 2}));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 64, activation: 'relu'}));
    model.add(tf.layers.dense({ units: 7, activation: 'softmax'}));
    //model.add(tf.layers.dense({ units: 1, activation: 'sigmoid'}));
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: 'adam',
        metrics: ['accuracy']
    });

    return model;
}


const model = buildModel()
train(model)