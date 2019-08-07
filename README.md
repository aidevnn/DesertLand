# DesertLand
A Deep Learning framework for education purpose in C#

## Example Iris dataset

```
(var trainX, var trainY, var testX, var testY) = ImportIris.SplittedDatasets<Type>(ratio: 0.8);
Console.WriteLine($"Train on {trainX.Shape[0]}; Test on {testX.Shape[0]}");

var net = new Network<Type>(new SGD<Type>(0.05), new SquareLoss<Type>(), new ArgmaxAccuracy<Type>());
net.AddLayer(new DenseLayer<Type>(4, 5, new TanhActivation<Type>()));
net.AddLayer(new DenseLayer<Type>(3, new SigmoidActivation<Type>()));

net.Summary();

var sw = Stopwatch.StartNew();
net.Fit(trainX, trainY, epochs: 50, batchSize: 10, displayEpochs: 5);
Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");

net.Test(testX, testY);
```

### Output


```
Hello World! Iris MLP. Network<Double>
Train on 120; Test on 30
Summary
Input  Shape:4
Layer: DenseLayer-Tanh      Parameters:    25 Nodes[In: 4 -> Out:5]
Layer: DenseLayer-Sigmoid   Parameters:    18 Nodes[In: 5 -> Out:3]
Output Shape:3
Total Parameters:43

Start Training...
Epochs     0/50 Loss:0.122382 Acc:0.4083
Epochs     5/50 Loss:0.068604 Acc:0.7250
Epochs    10/50 Loss:0.051477 Acc:0.8667
Epochs    15/50 Loss:0.042436 Acc:0.8833
Epochs    20/50 Loss:0.038008 Acc:0.9417
Epochs    25/50 Loss:0.032516 Acc:0.9417
Epochs    30/50 Loss:0.029528 Acc:0.9333
Epochs    35/50 Loss:0.025101 Acc:0.9333
Epochs    40/50 Loss:0.022236 Acc:0.9500
Epochs    45/50 Loss:0.020407 Acc:0.9667
Epochs    50/50 Loss:0.019197 Acc:0.9583
End Training.
Time:195 ms
TestResult Loss:0.009323 Acc:1.0000
```

## Example Digits from scipy dataset

```
(var trainX, var trainY, var testX, var testY) = ImportDigits.SplittedDatasets<Type>(ratio: 0.9);

var net = new Network<Type>(new SGD<Type>(0.05), new SquareLoss<Type>(), new ArgmaxAccuracy<Type>());
net.AddLayer(new DenseLayer<Type>(64, 32, new TanhActivation<Type>()));
net.AddLayer(new DenseLayer<Type>(10, new SigmoidActivation<Type>()));

net.Summary();

var sw = Stopwatch.StartNew();
net.Fit(trainX, trainY, epochs: 100, batchSize: 100, displayEpochs: 10);
Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");

net.Test(testX, testY);
```

### Output


```
Hello World! Digits MLP. Network<Double>
Train on 1617; Test on 180
Summary
Input  Shape:64
Layer: DenseLayer-Tanh      Parameters:  2080 Nodes[In:64 -> Out:32]
Layer: DenseLayer-Sigmoid   Parameters:   330 Nodes[In:32 -> Out:10]
Output Shape:10
Total Parameters:2410

Start Training...
Epochs     0/100 Loss:0.052491 Acc:0.2044
Epochs    10/100 Loss:0.006142 Acc:0.9469
Epochs    20/100 Loss:0.002374 Acc:0.9813
Epochs    30/100 Loss:0.001639 Acc:0.9894
Epochs    40/100 Loss:0.001064 Acc:0.9950
Epochs    50/100 Loss:0.000802 Acc:0.9956
Epochs    60/100 Loss:0.000583 Acc:0.9969
Epochs    70/100 Loss:0.000466 Acc:0.9981
Epochs    80/100 Loss:0.000404 Acc:0.9981
Epochs    90/100 Loss:0.000338 Acc:0.9981
Epochs   100/100 Loss:0.000282 Acc:0.9981
End Training.
Time:42496 ms
TestResult Loss:0.005300 Acc:0.9333
```

#### References.
Base code for layers / activations / network was in python and comes from this very great and useful ML repo https://github.com/eriklindernoren/ML-From-Scratch

NDArray was inspired from NumSharp repo https://github.com/SciSharp/NumSharp and Proxem.NumNet repo https://github.com/Proxem/NumNet
