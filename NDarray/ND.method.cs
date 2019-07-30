using System;
using System.Collections.Generic;
using System.Linq;

namespace NDarrayLib
{
    public static partial class ND
    {
        public static NDarray<Type> Scalar<Type>(Type v, params int[] shape) => new NDarray<Type>(v0: v, shape: shape);
        public static NDarray<Type> Zeros<Type>(params int[] shape) => Scalar(NDarray<Type>.OpsT.Zero, shape);
        public static NDarray<Type> Ones<Type>(params int[] shape) => Scalar(NDarray<Type>.OpsT.One, shape);

        public static NDarray<int> Arange(int start, int length, int step = 1)
        {
            int[] data = Enumerable.Range(0, length).Select(i => start + i * step).ToArray();
            return new NDarray<int>(data: data, shape: new int[] { length });
        }

        public static NDarray<int> Arange(int length) => Arange(0, length, 1);

        public static NDarray<Type> Uniform<Type>(Type min, Type max, params int[] shape)
        {
            int count = Utils.ArrMul(shape);
            Type[] data = Enumerable.Range(0, count).Select(i => NDarray<Type>.OpsT.Rand(min, max)).ToArray();
            return new NDarray<Type>(data: data, shape: shape);
        }

        public static NDarray<Type> CreateNDarray<Type>(Type[] data, params int[] shape) => new NDarray<Type>(data: data, shape: shape);
        public static NDarray<Type> CreateNDarray<Type>(Type[,] data)
        {
            int dim0 = data.GetLength(0);
            int dim1 = data.GetLength(1);

            Type[] data0 = new Type[dim0 * dim1];
            for (int i = 0; i < dim0; ++i)
                for (int j = 0; j < dim1; ++j)
                    data0[i * dim1 + j] = data[i, j];

            return CreateNDarray(data0, new int[] { dim0, dim1 });
        }

        public static List<(NDarray<Type>, NDarray<Type>)> BatchIterator<Type>(NDarray<Type> X, NDarray<Type> Y, int batchsize = 64, bool shuffle = true)
        {
            int dim0 = X.Shape[0];
            if (Y.Shape[0] != dim0)
                throw new ArgumentException();

            if (batchsize > dim0)
                batchsize = dim0;

            List<(NDarray<Type>, NDarray<Type>)> allBatch = new List<(NDarray<Type>, NDarray<Type>)>();
            int nb = dim0 / batchsize;

            var ltIdx = new Queue<int>(Enumerable.Range(0, dim0));
            if (shuffle)
                ltIdx = new Queue<int>(Enumerable.Range(0, dim0).OrderBy(t => Utils.GetRandom.NextDouble()));

            var xshape = X.Shape.ToArray();
            var yshape = Y.Shape.ToArray();
            xshape[0] = batchsize;
            yshape[0] = batchsize;

            for (int k = 0; k < nb; ++k)
            {
                var xarr = new NDarray<Type>(xshape);
                var yarr = new NDarray<Type>(yshape);
                var xdata = new Type[xarr.Count];
                var ydata = new Type[yarr.Count];
                xarr.SetData(xdata);
                yarr.SetData(ydata);
                for (int i = 0; i < batchsize; ++i)
                {
                    int idx = ltIdx.Dequeue();
                    X[idx].GetData.CopyTo(xdata, i * xarr.Strides[0]);
                    Y[idx].GetData.CopyTo(ydata, i * yarr.Strides[0]);
                }

                allBatch.Add((xarr, yarr));
            }

            return allBatch;
        }

    }
}
