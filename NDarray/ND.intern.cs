﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace NDarrayLib
{
    public static partial class ND
    {
        static NDarray<Type> reshape<Type>(NDarray<Type> nDarray, int[] shape)
        {
            if (Utils.IsDebugLvl2) Console.WriteLine("Reshape");

            var nshape = Utils.PrepareReshape(nDarray.Shape, shape);
            var nd0 = new NDarray<Type>(nshape)
            {
                getAt = i => nDarray.GetAt(i),
                setAt = (i, v) => nDarray.setAt(i, v),
                OwnData = false
            };

            return nd0;
        }

        static NDarray<Type> transpose<Type>(NDarray<Type> nDarray, int[] table)
        {
            if (Utils.IsDebugLvl2) Console.WriteLine("Transpose");

            if (table == null || table.Length == 0) table = Utils.PrepareTranspose(nDarray.Shape.Length);
            var nshape = Utils.DoTranspose(nDarray.Shape, table);
            var nstrides = Utils.DoTranspose(nDarray.Strides, table);
            var nd0 = new NDarray<Type>(shape: nshape, strides: nstrides)
            {
                getAt = idx0 => nDarray.GetAt(Utils.Int2IntIndex(idx0, nshape, nstrides)),
                setAt = (idx0, v) => nDarray.setAt(Utils.Int2IntIndex(idx0, nshape, nstrides), v),
                OwnData = false
            };

            return nd0;
        }

        static NDarray<V> applyOps<U, V>(NDarray<U> nDarray, Func<U, V> func)
        {
            if (Utils.IsDebugLvl2) Console.WriteLine($"ApplyOps {func.Method.Name}");

            var nd0 = new NDarray<V>(shape: nDarray.Shape, strides: nDarray.Strides)
            {
                getAt = i => func(nDarray.GetAt(i)),
                OwnData = false
            };

            return nd0;
        }

        static NDarray<V> elementWiseOp<U, V>(NDarray<U> left, NDarray<U> right, Func<U, U, V> func)
        {
            if (Utils.IsDebugLvl2) Console.WriteLine($"ElementWiseOp {func.Method.Name}");

            var nd0 = new NDarray<V>(shape: Utils.BroadCastShapes(left.Shape, right.Shape));
            nd0.OwnData = false;
            nd0.getAt = index =>
            {
                Utils.Int2ArrayIndex(index, nd0.Shape, nd0.Indices);
                for (int k = nd0.Indices.Length - 1, i = left.Shape.Length - 1, j = right.Shape.Length - 1; k >= 0; --k, --i, --j)
                {
                    if (i >= 0) left.Indices[i] = nd0.Indices[k] % left.Shape[i];
                    if (j >= 0) right.Indices[j] = nd0.Indices[k] % right.Shape[j];
                }

                var v0 = left.GetAt(Utils.Array2IntIndex(left.Indices, left.Shape, left.Strides));
                var v1 = right.GetAt(Utils.Array2IntIndex(right.Indices, right.Shape, right.Strides));
                return func(v0, v1);
            };

            return nd0;
        }

        static NDarray<int> argMinMax<Type>(NDarray<Type> nDarray, int axis, Func<Type, Type, Type> func, Type tmp)
        {
            if (Utils.IsDebugLvl2) Console.WriteLine($"ArgMinMax {func.Method.Name}");

            (int[] ishape, int[] nshape) = Utils.PrepareArgMinmax(nDarray.Shape, axis);
            int[] indices = new int[ishape.Length];
            var nd0 = new NDarray<int>(nshape);
            nd0.OwnData = false;
            int nb = nDarray.Shape[axis];

            nd0.getAt = idx =>
            {
                Type valBest = tmp;
                int idxBest = 0;
                Utils.Int2ArrayIndex(idx, ishape, indices);
                for (int k = 0; k < nb; ++k)
                {
                    indices[axis] = k;
                    var v = nDarray.GetAt(Utils.Array2IntIndex(indices, nDarray.Shape, nDarray.Strides));
                    var v0 = func(v, valBest);
                    if (!valBest.Equals(v0))
                    {
                        idxBest = k;
                        valBest = v0;
                    }
                }

                return idxBest;
            };

            return nd0;
        }

        static NDarray<Type> axisOps<Type>(NDarray<Type> nDarray, int axis, bool keepdims, Func<Type, Type, Type> func, Type start, bool mean = false)
        {
            if (Utils.IsDebugLvl2) Console.WriteLine($"AxisOps {func.Method.Name}");

            var nshape = Utils.PrepareSumProd(nDarray.Shape, axis, keepdims);
            var nd0 = new NDarray<Type>(nshape);
            if (axis == -1)
            {
                Type res = start;
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
                nd0.OwnData = false;
                nd0.getAt = idx0 =>
                {
                    Type res = start;
                    Utils.Int2ArrayIndex(idx0, NShape, NIndices);

                    for (int k = 0; k < nDarray.Shape[axis]; ++k)
                    {
                        NIndices[axis] = k;
                        int idx1 = Utils.Array2IntIndex(NIndices, nDarray.Shape, nDarray.Strides);
                        res = func(res, nDarray.GetAt(idx1));
                    }

                    res = NDarray<Type>.OpsT.Div(res, nb);
                    return res;
                };
            }

            return nd0;
        }

        // the worst case of for data access
        static NDarray<Type> axisCumOps<Type>(NDarray<Type> nDarray, int axis, Func<Type, Type, Type> func, Type start)
        {
            if (Utils.IsDebugLvl2) Console.WriteLine($"AxisCumOps {func.Method.Name}");

            var nshape = Utils.PrepareCumSumProd(nDarray.Shape, axis);
            var nd0 = new NDarray<Type>(nshape);
            nd0.OwnData = false;

            if (axis == -1)
            {
                nd0.getAt = idx0 =>
                {
                    Type res = start;
                    for (int idx1 = 0; idx1 <= idx0; ++idx1)
                        res = func(res, nDarray.GetAt(idx1));

                    return res;
                };
            }
            else
            {
                nd0.getAt = idx0 =>
                {
                    Type res = start;
                    Utils.Int2ArrayIndex(idx0, nDarray.Shape, nDarray.Indices);
                    int nb = nDarray.Indices[axis];

                    for (int k = 0; k <= nb; ++k)
                    {
                        nDarray.Indices[axis] = k;
                        int idx1 = Utils.Array2IntIndex(nDarray.Indices, nDarray.Shape, nDarray.Strides);
                        res = func(res, nDarray.GetAt(idx1));
                    }

                    return res;
                };
            }

            return nd0;
        }

        static NDarray<Type> tensorDot<Type>(NDarray<Type> left, NDarray<Type> right)
        {
            if (Utils.IsDebugLvl2) Console.WriteLine("TensorDot");

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
                Utils.Int2ArrayIndex(idx, shape, indices);

                for (int k = 0; k < shape.Length; ++k)
                {
                    if (k < length0 - 1) leftArr.Indices[idxInfos[k]] = indices[k];
                    else rightArr.Indices[idxInfos[k]] = indices[k];
                }

                for (int i = 0; i < piv; ++i)
                {
                    leftArr.Indices[length0 - 1] = rightArr.Indices[length1 - 2] = i;

                    int idxl = Utils.Array2IntIndex(leftArr.Indices, leftArr.Shape, leftArr.Strides);
                    int idxr = Utils.Array2IntIndex(rightArr.Indices, rightArr.Shape, rightArr.Strides);
                    var prod = NDarray<Type>.OpsT.Mul(leftArr.GetAt(idxl), rightArr.GetAt(idxr));
                    sum = NDarray<Type>.OpsT.Add(sum, prod);
                }

                Data[idx] = sum;
            }

            nd0.SetData(Data);
            return nd0;
        }

        static NDarray<Type> concatene<Type>(NDarray<Type> left, NDarray<Type> right, int axis = 0)
        {
            if (Utils.IsDebugLvl2) Console.WriteLine("Concatene");

            int[] nshape = Utils.PrepareConcatene(left.Shape, right.Shape, axis);
            var nd0 = new NDarray<Type>(shape: nshape);
            int dim = left.Shape[axis];

            nd0.getAt = idx =>
            {
                Utils.Int2ArrayIndex(idx, nd0.Shape, nd0.Indices);
                if (nd0.Indices[axis] < dim)
                {
                    int idx0 = Utils.Array2IntIndex(nd0.Indices, left.Shape, left.Strides);
                    return left.GetAt(idx0);
                }
                else
                {
                    nd0.Indices[axis] -= dim;
                    int idx0 = Utils.Array2IntIndex(nd0.Indices, right.Shape, right.Strides);
                    return right.GetAt(idx0);
                }
            };

            //nd0.SetData(nd0.GetData);
            return nd0;
        }

    }
}
