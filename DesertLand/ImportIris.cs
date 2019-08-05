using System;
using System.IO;
using System.Linq;
using NDarrayLib;

namespace DesertLand
{
    public static class ImportIris
    {

        static (double[], double[]) ImportDataset(string[] raw)
        {
            var data = raw.Select(l => l.Split(',').Select(double.Parse).ToArray()).ToArray();
            var dataX = data.SelectMany(l => l.Take(4)).ToArray();
            var dataY = data.SelectMany(l => l.Skip(4)).ToArray();

            return (dataX, dataY);
        }

        public static (NDarray<Type>, NDarray<Type>, NDarray<Type>, NDarray<Type>) SplittedDatasets<Type>(double ratio)
        {
            var raw = File.ReadAllLines("iris.csv").ToArray();
            (var dataX, var dataY) = ImportDataset(raw);

            var mx = dataX.Max();
            var mn = dataX.Min();

            int lX = dataX.Length / 4;
            int lY = dataY.Length / 3;
            int splitX = (int)(lX * ratio);
            int splitY = (int)(lY * ratio);

            double[] dataTrainX = dataX.Take(splitX * 4).Select(i => 0.1 + 0.8 * (i - mn) / (mx - mn)).ToArray();
            double[] dataTestX = dataX.Skip(splitX * 4).Select(i => 0.1 + 0.8 * (i - mn) / (mx - mn)).ToArray();
            double[] dataTrainY = dataY.Take(splitY * 3).ToArray();
            double[] dataTestY = dataY.Skip(splitY * 3).ToArray();

            var trainX = ND.CreateNDarray(dataTrainX, splitX, 4).CastCopy<Type>();
            var trainY = ND.CreateNDarray(dataTrainY, splitY, 3).CastCopy<Type>();
            var testX = ND.CreateNDarray(dataTestX, lX - splitX, 4).CastCopy<Type>();
            var testY = ND.CreateNDarray(dataTestY, lY - splitY, 3).CastCopy<Type>();

            return (trainX, trainY, testX, testY);
        }
    }
}
