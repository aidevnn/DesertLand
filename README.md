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
Epochs     0/50 Loss:0.121808 Acc:0.3250
Epochs     5/50 Loss:0.105686 Acc:0.5417
Epochs    10/50 Loss:0.092431 Acc:0.6667
Epochs    15/50 Loss:0.073587 Acc:0.6833
Epochs    20/50 Loss:0.063041 Acc:0.7500
Epochs    25/50 Loss:0.057486 Acc:0.7667
Epochs    30/50 Loss:0.054366 Acc:0.8583
Epochs    35/50 Loss:0.051425 Acc:0.8333
Epochs    40/50 Loss:0.049853 Acc:0.8833
Epochs    45/50 Loss:0.047732 Acc:0.9000
Epochs    50/50 Loss:0.046406 Acc:0.8750
End Training.
Time:235 ms
TestResult Loss:0.044790 Acc:0.9667
```

## Example Digits from scipy dataset

```
(var trainX, var trainY, var testX, var testY) = ImportDigits.SplittedDatasets<Type>(ratio: 0.9);

var net = new Network<Type>(new SGD<Type>(0.05), new SquareLoss<Type>(), new ArgmaxAccuracy<Type>());
net.AddLayer(new DenseLayer<Type>(64, 32, new SigmoidActivation<Type>()));
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
Summary
Input  Shape:64
Layer: DenseLayer-Sigmoid   Parameters:  2080 Nodes[In:64 -> Out:32]
Layer: DenseLayer-Sigmoid   Parameters:   330 Nodes[In:32 -> Out:10]
Output Shape:10
Total Parameters:2410

Start Training...
Epochs     0/100 Loss:0.052926 Acc:0.0869
Epochs    10/100 Loss:0.017278 Acc:0.8469
Epochs    20/100 Loss:0.006884 Acc:0.9575
Epochs    30/100 Loss:0.004312 Acc:0.9694
Epochs    40/100 Loss:0.003357 Acc:0.9769
Epochs    50/100 Loss:0.002736 Acc:0.9819
Epochs    60/100 Loss:0.002310 Acc:0.9838
Epochs    70/100 Loss:0.001997 Acc:0.9875
Epochs    80/100 Loss:0.001736 Acc:0.9888
Epochs    90/100 Loss:0.001562 Acc:0.9931
Epochs   100/100 Loss:0.001386 Acc:0.9938
End Training.
Time:47123 ms
TestResult Loss:0.005358 Acc:0.9333
```

#### References.
Base code for layers / activations / network was in python and comes from this very great and useful ML repo https://github.com/eriklindernoren/ML-From-Scratch

NDArray was inspired from NumSharp repo https://github.com/SciSharp/NumSharp and Proxem.NumNet repo https://github.com/Proxem/NumNet
