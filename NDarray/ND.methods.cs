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

        public static NDview<Type> Reshape<Type>(NDview<Type> nDview, params int[] shape)
        {
            Fnc<Type> fnc = () =>
            {
                if (Utils.IsDebugLvl2) Console.WriteLine("Reshape");

                var nDarray = nDview.fnc();
                var nshape = Utils.PrepareReshape(nDarray.Shape, shape);
                var nd0 = new NDarray<Type>(nshape)
                {
                    getAt = i => nDarray.GetAt(i),
                    setAt = (i, v) => nDarray.SetAt(i, v)
                };
                return nd0;
            };

            return new NDview<Type>(fnc);
        }

        public static NDview<Type> Transpose<Type>(NDview<Type> nDview, params int[] table)
        {
            Fnc<Type> fnc = () =>
            {
                if (Utils.IsDebugLvl2) Console.WriteLine("Transpose");

                var nDarray = nDview.fnc();
                if (table == null || table.Length == 0) table = Utils.PrepareTranspose(nDarray.Shape.Length);
                var nshape = Utils.DoTranspose(nDarray.Shape, table);
                var nstrides = Utils.DoTranspose(nDarray.Strides, table);
                var nd0 = new NDarray<Type>(shape: nshape, strides: nstrides)
                {
                    getAt = idx0 => nDarray.GetAt(Utils.GetNewIndex(idx0, nshape, nstrides)),
                    setAt = (idx0, v) => nDarray.SetAt(Utils.GetNewIndex(idx0, nshape, nstrides), v)
                };
                return nd0;
            };

            return new NDview<Type>(fnc);
        }

        public static NDview<V> ApplyOps<U, V>(NDview<U> nDview, Func<U, V> func)
        {
            Fnc<V> fnc = () =>
            {
                if (Utils.IsDebugLvl2) Console.WriteLine($"ApplyOps {func.Method.Name}");

                var nDarray = nDview.fnc();
                var nd0 = new NDarray<V>(shape: nDarray.Shape,strides: nDarray.Strides) { getAt = i => func(nDarray.GetAt(i))};
                return nd0;
            };
            return new NDview<V>(fnc);
        }

        internal static NDview<V> ApplyOpsLeft<U, V>(double left, Func<U, U, V> func, NDview<U> right)
            => ApplyOps(right, x => func(NDarray<U>.OpsT.Cast(left), x));

        internal static NDview<V> ApplyOpsRight<U, V>(NDview<U> left, Func<U, U, V> func, double right)
            => ApplyOps(left, x => func(x, NDarray<U>.OpsT.Cast(right)));

        public static NDview<V> ElementWiseOp<U, V>(NDview<U> fleft, NDview<U> fright, Func<U, U, V> func)
        {
            Fnc<V> fnc = () =>
            {
                if (Utils.IsDebugLvl2) Console.WriteLine($"ElementWiseOp {func.Method.Name}");

                var left = fleft.fnc();
                var right = fright.fnc();
                var nd0 = new NDarray<V>(shape: Utils.BroadCastShapes(left.Shape, right.Shape));
                nd0.getAt = index =>
                {
                    Utils.InputIndicesFromIndex(index, nd0.Shape, nd0.Indices);
                    for (int k = nd0.Indices.Length - 1, i = left.Shape.Length - 1, j = right.Shape.Length - 1; k >= 0; --k, --i, --j)
                    {
                        if (i >= 0) left.Indices[i] = nd0.Indices[k] % left.Shape[i];
                        if (j >= 0) right.Indices[j] = nd0.Indices[k] % right.Shape[j];
                    }

                    var v0 = left.GetAt(Utils.Indices2Offset(left.Indices, left.Shape, left.Strides));
                    var v1 = right.GetAt(Utils.Indices2Offset(right.Indices, right.Shape, right.Strides));
                    return func(v0, v1);
                };

                return nd0;
            };
            return new NDview<V>(fnc);
        }

        public static NDview<Type> AxisOps<Type>(NDview<Type> nDview, int axis, bool keepdims, Func<Type, Type, Type> func, Type neutre, bool mean = false)
        {
            Fnc<Type> fnc = () =>
            {
                if (Utils.IsDebugLvl2) Console.WriteLine($"AxisOps {func.Method.Name}");

                var nDarray = nDview.fnc();
                var nshape = Utils.PrepareSumProd(nDarray.Shape, axis, keepdims);
                var nd0 = new NDarray<Type>(nshape);
                if (axis == -1)
                {
                    Type res = neutre;
                    Type nb = mean ? NDarray<Type>.OpsT.Cast(nDarray.Count) : NDarray<Type>.OpsT.One;
                    for (int idx = 0; idx < nDarray.Count; ++idx)
                        res = func(res, nDarray.GetAt(idx));

                    res = NDarray<Type>.OpsT.Div(res, nb);
                    nd0.getAt = i => res;
                }
                else
                {
                    var NShape = Utils.PrepareSumProd(nDarray.Shape, axis, true);
                    var NIndices = new int[NShape.Length];
                    Type nb = mean ? NDarray<Type>.OpsT.Cast(nDarray.Shape[axis]) : NDarray<Type>.OpsT.One;
                    nd0.getAt = idx0 =>
                    {
                        Type res = neutre;
                        Utils.InputIndicesFromIndex(idx0, NShape, NIndices);

                        for (int k = 0; k < nDarray.Shape[axis]; ++k)
                        {
                            NIndices[axis] = k;
                            int idx1 = Utils.Indices2Offset(NIndices, nDarray.Shape, nDarray.Strides);
                            res = func(res, nDarray.GetAt(idx1));
                        }

                        res = NDarray<Type>.OpsT.Div(res, nb);
                        return res;
                    };
                }

                return nd0;
            };

            return new NDview<Type>(fnc);
        }

        public static NDview<Type> TensorDot<Type>(NDview<Type> a, NDview<Type> b)
        {
            Fnc<Type> fnc = () =>
            {
                if (Utils.IsDebugLvl2) Console.WriteLine("TensorDot");

                var left = a.fnc();
                var right = b.fnc();
                left.SetData(left.GetData);
                right.SetData(right.GetData);

                (int[] lshape, int[] rshape, int[] shape, int[] idxInfos) = Utils.PrepareDot(left.Shape, right.Shape);
                var nd0 = new NDarray<Type>(shape);

                var leftArr = new NDarray<Type>(lshape);
                var rightArr = new NDarray<Type>(rshape);
                leftArr.SetData(left.GetData);
                rightArr.SetData(right.GetData);

                int length0 = lshape.Length;
                int length1 = rshape.Length;
                int piv = lshape.Last();

                int[] indices = new int[shape.Length];
                Type[] Data = new Type[nd0.Count];
                for (int idx = 0; idx < nd0.Count; ++idx)
                {
                    Type sum = NDarray<Type>.OpsT.Zero;
                    Utils.InputIndicesFromIndex(idx, shape, indices);

                    for (int k = 0; k < shape.Length; ++k)
                    {
                        if (k < length0 - 1) leftArr.Indices[idxInfos[k]] = indices[k];
                        else rightArr.Indices[idxInfos[k]] = indices[k];
                    }

                    for (int i = 0; i < piv; ++i)
                    {
                        leftArr.Indices[length0 - 1] = rightArr.Indices[length1 - 2] = i;

                        int idxl = Utils.Indices2Offset(leftArr.Indices, leftArr.Shape, leftArr.Strides);
                        int idxr = Utils.Indices2Offset(rightArr.Indices, rightArr.Shape, rightArr.Strides);
                        var prod = NDarray<Type>.OpsT.Mul(leftArr.GetAt(idxl), rightArr.GetAt(idxr));
                        sum = NDarray<Type>.OpsT.Add(sum, prod);
                    }

                    Data[idx] = sum;
                }

                nd0.SetData(Data);
                return nd0;
            };

            return new NDview<Type>(fnc);
        }
    }
}
