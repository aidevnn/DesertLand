using System;
using System.Diagnostics;
using DesertLand.Activations;
using DesertLand.Layers;
using DesertLand.Losses;
using DesertLand.Optimizers;
using NDarrayLib;

namespace DesertLand
{
    class MainClass
    {
        static void Test1()
        {
            var a = ND.Arange(15).Reshape(5, 3);
            var b = ND.Arange(4, 3);
            var c = a + 2 * b;
            Console.WriteLine(a);
            Console.WriteLine(b);
            Console.WriteLine(c);
        }

        static void Test2()
        {
            var a = ND.Uniform<double>(0, 1, 4, 1);
            var b = ND.Uniform<double>(0, 1, 4, 1);

            Func<NDview<double>, NDview<double>> f = x => -a * ND.Log(x) + (1 - a) * ND.Log(1 - x);
            Func<NDview<double>, double, NDview<double>> df = (x, h) => (f(x + h) - f(x)) / h;

            var c = f(b);
            var d = -a / b - (1 - a) / (1 - b);
            var e = df(b, 1e-12);

            Console.WriteLine(a);
            Console.WriteLine(b);
            Console.WriteLine(c);
            Console.WriteLine(d);
            Console.WriteLine(e);
            Console.WriteLine(ND.Abs(d - e));
        }

        static void Test3()
        {
            var a = ND.Uniform(1, 10, 2, 3, 4).CastCopy<double>();
            Console.WriteLine(a);

            Console.WriteLine(a.Sum());
            Console.WriteLine(a.Sum(0));
            Console.WriteLine(a.Sum(1));
            Console.WriteLine(a.Sum(2));

            Console.WriteLine(a.Prod());
            Console.WriteLine(a.Prod(0));
            Console.WriteLine(a.Prod(1));
            Console.WriteLine(a.Prod(2));

            Console.WriteLine(a.Mean());
            Console.WriteLine(a.Mean(0));
            Console.WriteLine(a.Mean(1));
            Console.WriteLine(a.Mean(2));

            //var a = ND.Uniform(1, 10, 4, 2, 2);
            //Console.WriteLine(a);
            //Console.WriteLine(a.Prod(2));
            //Console.WriteLine(a.Prod(2).Prod(1, true));
            //Console.WriteLine(a.Reshape(4, -1).Prod(1, true));
        }

        static void Test4()
        {
            var a = ND.Uniform(1, 10, 4, 4);
            var b = ND.Uniform(1, 10, 2, 4, 2);
            Console.WriteLine(a);
            Console.WriteLine(b);
            Console.WriteLine(ND.TensorDot<int>(a, b));
        }

        static void Test5()
        {
            var a = ND.Arange(0, 24).Reshape(2, -1, 3);
            var b = ND.Arange(24, 12).Reshape(2, -1, 3);

            Console.WriteLine(a);
            Console.WriteLine(b);
            Console.WriteLine(ND.Concatene(a, b, 1));

            //var a = ND.Arange(1, 8).Reshape(4, -1).Copy;
            //var b = ND.Arange(1, 4).Reshape(4, -1).Copy;

            //Console.WriteLine(a);
            //Console.WriteLine(b);

            //Utils.DebugNumpy = Utils.DbgNo;
            //var allBatch = ND.BatchIterator(a, b, 2, true);
            //foreach (var batch in allBatch)
                //Console.WriteLine(ND.HConcat<int>(batch.Item1, batch.Item2));
        }

        static void TestXor<Type>()
        {
            Console.WriteLine($"Hello World! Xor MLP. Backend NDarray<{typeof(Type).Name}>");

            Utils.DebugNumpy = Utils.DbgNo;

            var Xdata = ND.CreateNDarray(new double[4, 2] { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } }).CastCopy<Type>();
            var Ydata = ND.CreateNDarray(new double[4, 1] { { 0 }, { 1 }, { 1 }, { 0 } }).CastCopy<Type>();

            var net = new Network<Type>(new SGD<Type>(0.2), new SquareLoss<Type>(), new RoundAccuracy<Type>());
            net.AddLayer(new DenseLayer<Type>(2, 8, new TanhActivation<Type>()));
            net.AddLayer(new DenseLayer<Type>(1, new SigmoidActivation<Type>()));

            //net.AddLayer(new DenseLayer<Type>(2, 8));
            //net.AddLayer(new TanhLayer<Type>());
            //net.AddLayer(new DenseLayer<Type>(1));
            //net.AddLayer(new SigmoidLayer<Type>());

            net.Summary();

            var sw = Stopwatch.StartNew();
            net.Fit(Xdata, Ydata, 1000, 4, 500);
            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");

            Console.WriteLine("Prediction");
            NDarray<Type> pred = net.Predict(Xdata).Round(6);
            for (int k = 0; k < Xdata.Shape[0]; ++k)
            {
                Console.WriteLine($"{Xdata[k]} = {Ydata[k]} -> {pred[k]}");
            }
        }

        static void TestDigits<Type>()
        {
            Console.WriteLine($"Hello World! Digits MLP. Network<{typeof(Type).Name}>");

            Utils.DebugNumpy = Utils.DbgNo;

            (var trainX, var trainY, var testX, var testY) = ImportData.DigitsDataset<Type>(ratio: 0.9);
            Console.WriteLine($"Train on {trainX.Shape[0]}; Test on {testX.Shape[0]}");

            var net = new Network<Type>(new SGD<Type>(0.05), new SquareLoss<Type>(), new ArgmaxAccuracy<Type>());
            net.AddLayer(new DenseLayer<Type>(64, 32, new TanhActivation<Type>()));
            net.AddLayer(new DenseLayer<Type>(10, new SigmoidActivation<Type>()));

            net.Summary();

            var sw = Stopwatch.StartNew();
            net.Fit(trainX, trainY, epochs: 100, batchSize: 100, displayEpochs: 10);
            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");

            net.Test(testX, testY);
        }

        static void TestIris<Type>()
        {
            Console.WriteLine($"Hello World! Iris MLP. Network<{typeof(Type).Name}>");

            Utils.DebugNumpy = Utils.DbgNo;

            (var trainX, var trainY, var testX, var testY) = ImportData.IrisDataset<Type>(ratio: 0.8);
            Console.WriteLine($"Train on {trainX.Shape[0]}; Test on {testX.Shape[0]}");

            var net = new Network<Type>(new SGD<Type>(0.05), new SquareLoss<Type>(), new ArgmaxAccuracy<Type>());
            net.AddLayer(new DenseLayer<Type>(4, 5, new TanhActivation<Type>()));
            net.AddLayer(new DenseLayer<Type>(3, new SigmoidActivation<Type>()));

            net.Summary();

            var sw = Stopwatch.StartNew();
            net.Fit(trainX, trainY, epochs: 50, batchSize: 10, displayEpochs: 5);
            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");

            net.Test(testX, testY);
        }

        public static void Main(string[] args)
        {
            Utils.DebugNumpy = Utils.DbgLvlAll;
            //Test1();

            //TestXor<double>();
            TestDigits<double>();
            //TestIris<double>();

        }
    }
}
