using System;
using System.IO;
using System.Linq;

using NDarrayLib;

namespace DesertLand
{
    public static class ImportDigits
    {

        static float[] Categorical(int[] x)
        {
            var data = x.SelectMany(i => Enumerable.Range(0, 10).Select((v, k) => k == i ? 1f : 0f)).ToArray();
            return data;
        }

        static (float[], float[]) ImportDataset(string[] raw)
        {
            var data = raw.Select(l => l.Split(',').Select(float.Parse).ToArray()).ToArray();
            var dataX = data.SelectMany(l => l.Take(64)).ToArray();
            var dataY = Categorical(data.Select(l => (int)l[64]).ToArray());

            return (dataX, dataY);
        }

        public static (NDarray<Type>, NDarray<Type>, NDarray<Type>, NDarray<Type>) SplittedDatasets<Type>(double ratio)
        {
            var raw = File.ReadAllLines("digits.csv").ToArray();
            (var dataX, var dataY) = ImportDataset(raw);

            int lX = dataX.Length / 64;
            int lY = dataY.Length / 10;
            int splitX = (int)(lX * ratio);
            int splitY = (int)(lY * ratio);

            float[] dataTrainX = dataX.Take(splitX * 64).Select(i => i / 16.0f).ToArray();
            float[] dataTestX = dataX.Skip(splitX * 64).Select(i => i / 16.0f).ToArray();
            float[] dataTrainY = dataY.Take(splitY * 10).ToArray();
            float[] dataTestY = dataY.Skip(splitY * 10).ToArray();

            var trainX = ND.CreateNDarray(dataTrainX, splitX, 64).CastCopy<Type>();
            var trainY = ND.CreateNDarray(dataTrainY, splitY, 10).CastCopy<Type>();
            var testX = ND.CreateNDarray(dataTestX, lX - splitX, 64).CastCopy<Type>();
            var testY = ND.CreateNDarray(dataTestY, lY - splitY, 10).CastCopy<Type>();

            return (trainX, trainY, testX, testY);
        }
    }
}
