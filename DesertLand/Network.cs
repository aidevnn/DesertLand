using System;
using System.Collections.Generic;
using System.Linq;
using DesertLand.Activations;
using DesertLand.Layers;
using DesertLand.Losses;
using DesertLand.Optimizers;
using NDarrayLib;

namespace DesertLand
{
    public class Network<Type>
    {
        public Network(IOptimizer<Type> optimizer, ILoss<Type> loss, IAccuracy<Type> accuracy)
        {
            this.optimizer = optimizer;
            this.loss = loss;
            this.accuracy = accuracy;
        }

        public readonly IOptimizer<Type> optimizer;
        public readonly ILoss<Type> loss;
        public readonly IAccuracy<Type> accuracy;
        public List<ILayer<Type>> layers = new List<ILayer<Type>>();

        public void SetTrainable(bool train = true) => layers.ForEach(l => l.IsTraining = train);

        public void AddLayer(ILayer<Type> layer)
        {
            if (layers.Count != 0)
                layer.SetInputShape(layers.Last().OutputShape);

            layer.Initialize(optimizer);
            layers.Add(layer);
        }

        public NDarray<Type> ForwardPass(NDarray<Type> X, bool isTraining = true)
        {
            var layerOutput = X.Copy;
            foreach (var layer in layers)
                layerOutput = layer.Forward(layerOutput, isTraining);

            return layerOutput;
        }

        public void BackwardPass(NDarray<Type> lossGrad)
        {
            foreach (var layer in layers.Reverse<ILayer<Type>>())
                lossGrad = layer.Backward(lossGrad);
        }

        public NDarray<Type> Predict(NDarray<Type> X) => ForwardPass(X, false);

        (double, double) TestOnBatch(NDarray<Type> X, NDarray<Type> y)
        {
            SetTrainable(false);
            var yp = ForwardPass(X, false);
            var lossv = loss.Loss(y, yp).MeanAll();
            var accv = accuracy.Acc(y, yp).MeanAll();

            return (lossv, accv);
        }

        public (double, double) TrainOnBatch(NDarray<Type> X, NDarray<Type> y)
        {
            var yp = ForwardPass(X);
            var lossv = loss.Loss(y, yp).MeanAll();
            var accv = accuracy.Acc(y, yp).MeanAll();
            var lossGrad = loss.Grad(y, yp);
            BackwardPass(lossGrad);

            return (lossv, accv);
        }

        public void Summary()
        {
            Console.WriteLine("Summary");
            Console.WriteLine($"Input  Shape:{layers[0].InputShape.Glue()}");
            int tot = 0;
            foreach (var layer in layers)
            {
                Console.WriteLine($"Layer: {layer.Name,-20} Parameters: {layer.Params,5} Nodes[In:{layer.InputShape.Glue(),2} -> Out:{layer.OutputShape.Glue()}]");
                tot += layer.Params;
            }

            Console.WriteLine($"Output Shape:{layers.Last().OutputShape.Glue()}");
            Console.WriteLine($"Total Parameters:{tot}");
            Console.WriteLine();
        }

        public void Fit(NDarray<Type> X, NDarray<Type> y, int epochs, int batchSize = 64, int displayEpochs = 1)
        {
            Console.WriteLine("Start Training...");

            SetTrainable();

            for (int k = 0; k <= epochs; ++k)
            {
                List<double> losses = new List<double>();
                List<double> accs = new List<double>();

                var batchData = ND.BatchIterator(X, y, batchSize);
                foreach (var batch in batchData)
                {
                    var (loss, acc) = TrainOnBatch(batch.Item1, batch.Item2);
                    losses.Add(loss);
                    accs.Add(acc);
                }

                if (k % displayEpochs == 0)
                    Console.WriteLine("Epochs {0,5}/{1} Loss:{2:0.000000} Acc:{3:0.0000}", k, epochs, losses.Average(), accs.Average());
            }
            Console.WriteLine("End Training.");
        }

        public void Test(NDarray<Type> testX, NDarray<Type> testY)
        {
            var (loss, acc) = TestOnBatch(testX, testY);
            Console.WriteLine("TestResult Loss:{0:0.000000} Acc:{1:0.0000}", loss, acc);
        }
    }
}
