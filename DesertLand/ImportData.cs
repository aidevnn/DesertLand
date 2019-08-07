using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NDarrayLib;

namespace DesertLand
{
    public static class ImportData
    {

        public static (NDarray<Type>, NDarray<Type>, NDarray<Type>, NDarray<Type>) DigitsDataset<Type>(double ratio)
        {
            Func<int, IEnumerable<double>> func0 = v => Enumerable.Range(0, 10).Select(v0 => v == v0 ? 1.0 : 0.0);
            Func<double, int, IEnumerable<double>> func1 = (v, i) => i % 65 != 64 ? new double[] { v } : func0((int)v);

            var raw = File.ReadAllLines("datasets/digits.csv").ToArray();

            var data = raw.SelectMany(l => l.Split(',')).Select(double.Parse).ToArray();
            data = data.SelectMany(func1).ToArray();

            var nDarray = ND.CreateNDarray(data: data, -1, 74);
            var idx0 = (int)(nDarray.Shape[0] * ratio);
            (var train, var test) = ND.Split(nDarray, 0, idx0);

            (var trainX, var trainY) = ND.Split(train, 1, 64);
            (var testX, var testY) = ND.Split(test, 1, 64);

            trainX = trainX / 16.0;
            testX = testX / 16.0;

            return (trainX.CastCopy<Type>(), trainY.CastCopy<Type>(), testX.CastCopy<Type>(), testY.CastCopy<Type>());
        }

        public static (NDarray<Type>, NDarray<Type>, NDarray<Type>, NDarray<Type>) IrisDataset<Type>(double ratio)
        {
            var raw = File.ReadAllLines("datasets/iris.csv").ToArray();
            var data = raw.SelectMany(l => l.Split(',')).Select(double.Parse).ToArray();

            var nDarray = ND.CreateNDarray(data: data, -1, 7);
            var idx0 = (int)(nDarray.Shape[0] * ratio);
            (var train, var test) = ND.Split(nDarray, 0, idx0);

            (var trainX, var trainY) = ND.Split(train, 1, 4);
            (var testX, var testY) = ND.Split(test, 1, 4);

            var vmax = ND.Max(trainX.Max(0, true), testX.Max(0, true)).Copy;
            trainX = trainX / vmax;
            testX = testX / vmax;

            return (trainX.CastCopy<Type>(), trainY.CastCopy<Type>(), testX.CastCopy<Type>(), testY.CastCopy<Type>());
        }
    }
}
