using System;
using System.Collections.Generic;
using System.Linq;

namespace NDarrayLib
{
    public static class Utils
    {

        public static string Glue<T>(this IEnumerable<T> ts, string sep = " ", string format = "{0}") =>
            string.Join(sep, ts.Select(a => string.Format(format, a)));

        public const int DbgNo = 0, DbgLvl1 = 0b1, DbgLvl2 = 0b10, DbgLvlAll = 0b11;
        public static int DebugNumpy = DbgNo;

        public static bool IsDebugNo => DebugNumpy == DbgNo;
        public static bool IsDebugLvl1 => (DebugNumpy & DbgLvl1) == DbgLvl1;
        public static bool IsDebugLvl2 => (DebugNumpy & DbgLvl2) == DbgLvl2;

        //private static readonly Random random = new Random(123);
        private static readonly Random random = new Random((int)DateTime.Now.Ticks);
        
        public static Random GetRandom => random;

        public static int ArrMul(int[] shape, int start = 0) => shape.Skip(start).Aggregate(1, (a, i) => a * i);
        public static int[] Shape2Strides(int[] shape) => Enumerable.Range(0, shape.Length).Select(i => ArrMul(shape, i + 1)).ToArray();

        public static int Indices2Offset(int[] args, int[] shape, int[] strides)
        {
            int idx = 0;
            for (int k = 0; k < args.Length; ++k)
            {
                var v = args[k];
                if (v >= shape[k]) throw new ArgumentException();
                idx += v * strides[k];
            }

            return idx;
        }

        public static void InputIndicesFromIndex(int idx, int[] shape, int[] indices)
        {

            for (int k = shape.Length - 1; k >= 0; --k)
            {
                var sk = shape[k];
                indices[k] = idx % sk;
                idx = idx / sk;
            }
        }

        public static int GetNewIndex(int idx0, int[] shape, int[] strides)
        {
            int idx1 = 0;
            for (int k = shape.Length - 1; k >= 0; --k)
            {
                var sk = shape[k];
                idx1 += strides[k] * (idx0 % sk);
                idx0 = idx0 / sk;
            }

            return idx1;
        }

        public static int[] PrepareReshape(int[] baseShape, int[] shape)
        {
            int mone = shape.Count(i => i == -1);
            if (mone > 1)
                throw new ArgumentException("Only one dimension can be broadcasted");

            var dim0 = ArrMul(baseShape);
            if (mone == 1)
            {
                int idx = shape.ToList().FindIndex(i => i == -1);
                shape[idx] = 1;
                var dim2 = ArrMul(shape);
                shape[idx] = dim0 / dim2;
            }

            var dim1 = ArrMul(shape);

            if (dim0 != dim1)
                throw new ArgumentException($"Cannot reshape ({baseShape.Glue()}) to ({shape.Glue()})");

            return shape;
        }

        public static int[] PrepareTranspose(int rank) => Enumerable.Range(0, rank).Reverse().ToArray();
        public static int[] DoTranspose(int[] arr, int[] table) => Enumerable.Range(0, arr.Length).Select(i => arr[table[i]]).ToArray();

        public static int[] BroadCastShapes(int[] shape0, int[] shape1)
        {
            int sLength0 = shape0.Length;
            int sLength1 = shape1.Length;
            int mLength = Math.Max(sLength0, sLength1);

            int[] nshape = new int[mLength];
            for (int k = mLength - 1, i = sLength0 - 1, j = sLength1 - 1; k >= 0; --k, --i, --j)
            {
                int idx0 = i < 0 ? 1 : shape0[i];
                int idx1 = j < 0 ? 1 : shape1[j];
                if (idx0 != idx1 && idx0 != 1 && idx1 != 1)
                    throw new ArgumentException($"Cannot broadcast ({shape0.Glue()}) with ({shape1.Glue()})");

                nshape[k] = Math.Max(idx0, idx1);
            }

            return nshape;
        }

        public static int[] PrepareSumProd(int[] shape, int axis, bool keepdims)
        {
            List<int> nshape = new List<int>(shape);

            if (axis == -1)
                nshape = Enumerable.Repeat(1, shape.Length).ToList();
            else
                nshape[axis] = 1;

            if (!keepdims)
            {
                if (axis == -1)
                    nshape = new List<int>() { 1 };
                else
                    nshape.RemoveAt(axis);
            }

            return nshape.ToArray();
        }

        public static (int[], int[], int[], int[]) PrepareDot(int[] shape0, int[] shape1)
        {
            bool head = false, tail = false;
            int[] nshape;
            int[] lshape, rshape, idxInfos;

            if (head = shape0.Length == 1)
                lshape = new int[] { 1, shape0[0] };
            else
                lshape = shape0.ToArray();

            if (tail = shape1.Length == 1)
                rshape = new int[] { shape1[0], 1 };
            else
                rshape = shape1.ToArray();


            int length0 = lshape.Length;
            int length1 = rshape.Length;
            int piv = lshape.Last();

            if (piv != rshape[length1 - 2])
                throw new ArgumentException($"Cannot multiply ({shape0.Glue()}) and ({shape1.Glue()})");

            nshape = new int[length0 + length1 - 2];
            idxInfos = new int[length0 + length1 - 2];

            for (int k = 0, k0 = 0; k < length0 + length1; ++k)
            {
                if (k == length0 - 1 || k == length0 + length1 - 2) continue;
                if (k < length0 - 1) nshape[k] = lshape[idxInfos[k] = k];
                else nshape[k0] = rshape[idxInfos[k0] = k - length0];
                ++k0;
            }

            return (lshape, rshape, nshape, idxInfos);
        }

    }
}
