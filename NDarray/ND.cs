using System;
using System.Collections.Generic;
using System.Linq;

namespace NDarrayLib
{
    public static class ND
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

        internal static NDview<Type> Reshape<Type>(NDview<Type> nDview, params int[] shape)
        {
            Fnc<Type> fnc = () =>
            {
                if (Utils.DebugNumPy) Console.WriteLine("Reshape");

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

        internal static NDview<Type> Transpose<Type>(NDview<Type> nDview, params int[] table)
        {
            Fnc<Type> fnc = () =>
            {
                if (Utils.DebugNumPy) Console.WriteLine("Transpose");

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

        internal static NDview<V> ApplyOps<U, V>(NDview<U> nDview, Func<U, V> func)
        {
            Fnc<V> fnc = () =>
            {
                if (Utils.DebugNumPy) Console.WriteLine($"ApplyOps {func.Method.Name}");

                var nDarray = nDview.fnc();
                var nd0 = new NDarray<V>(shape: nDarray.Shape,strides: nDarray.Strides) { getAt = i => func(nDarray.GetAt(i))};
                return nd0;
            };
            return new NDview<V>(fnc);
        }

        internal static NDview<V> ApplyOpsLeft<U, V>(double left, Func<U, U, V> func, NDview<U> right)
            => ApplyOps(right, x => func(NDarray<U>.OpsT.Cast(left), x));

        public static NDview<V> ApplyOpsRight<U, V>(NDview<U> left, Func<U, U, V> func, double right)
            => ApplyOps(left, x => func(x, NDarray<U>.OpsT.Cast(right)));

        public static NDview<V> ElementWiseOp<U, V>(NDview<U> fleft, NDview<U> fright, Func<U, U, V> func)
        {
            Fnc<V> fnc = () =>
            {
                if (Utils.DebugNumPy) Console.WriteLine($"ElementWiseOp {func.Method.Name}");

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

        public static NDview<Type> Neg<Type>(NDview<Type> nDview) => ApplyOps(nDview, NDarray<Type>.OpsT.Neg);
        public static NDview<Type> Abs<Type>(NDview<Type> nDview) => ApplyOps(nDview, NDarray<Type>.OpsT.Abs);
        public static NDview<Type> Exp<Type>(NDview<Type> nDview) => ApplyOps(nDview, NDarray<Type>.OpsT.Exp);
        public static NDview<Type> Log<Type>(NDview<Type> nDview) => ApplyOps(nDview, NDarray<Type>.OpsT.Log);
        public static NDview<Type> Sq<Type>(NDview<Type> nDview) => ApplyOps(nDview, NDarray<Type>.OpsT.Sq);
        public static NDview<Type> Sqrt<Type>(NDview<Type> nDview) => ApplyOps(nDview, NDarray<Type>.OpsT.Sqrt);
        public static NDview<Type> Tanh<Type>(NDview<Type> nDview) => ApplyOps(nDview, NDarray<Type>.OpsT.Tanh);
        public static NDview<Type> DTanh<Type>(NDview<Type> nDview) => ApplyOps(nDview, NDarray<Type>.OpsT.DTanh);
        public static NDview<Type> Sigmoid<Type>(NDview<Type> nDview) => ApplyOps(nDview, NDarray<Type>.OpsT.Sigmoid);
        public static NDview<Type> DSigmoid<Type>(NDview<Type> nDview) => ApplyOps(nDview, NDarray<Type>.OpsT.DSigmoid);
        public static NDview<Type> Round<Type>(NDview<Type> nDview, int dec = 0) => ApplyOps(nDview, x => NDarray<Type>.OpsT.Round(x, dec));
        public static NDview<Type> Clamp<Type>(NDview<Type> nDview, double min, double max) => ApplyOps(nDview, x => NDarray<Type>.OpsT.Clamp(x, min, max));

        public static NDview<V> Cast<U, V>(NDview<U> nDview) => ApplyOps(nDview, NDarray<V>.OpsT.Cast);

        public static NDview<Type> Add<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Add);
        public static NDview<Type> Add<Type>(double left, NDview<Type> right) => ApplyOpsLeft(left, NDarray<Type>.OpsT.Add, right);
        public static NDview<Type> Add<Type>(NDview<Type> left, double right) => ApplyOpsRight(left, NDarray<Type>.OpsT.Add, right);

        public static NDview<Type> Sub<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Sub);
        public static NDview<Type> Sub<Type>(double left, NDview<Type> right) => ApplyOpsLeft(left, NDarray<Type>.OpsT.Sub, right);
        public static NDview<Type> Sub<Type>(NDview<Type> left, double right) => ApplyOpsRight(left, NDarray<Type>.OpsT.Sub, right);

        public static NDview<Type> Mul<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Mul);
        public static NDview<Type> Mul<Type>(double left, NDview<Type> right) => ApplyOpsLeft(left, NDarray<Type>.OpsT.Mul, right);
        public static NDview<Type> Mul<Type>(NDview<Type> left, double right) => ApplyOpsRight(left, NDarray<Type>.OpsT.Mul, right);

        public static NDview<Type> Div<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Div);
        public static NDview<Type> Div<Type>(double left, NDview<Type> right) => ApplyOpsLeft(left, NDarray<Type>.OpsT.Div, right);
        public static NDview<Type> Div<Type>(NDview<Type> left, double right) => ApplyOpsRight(left, NDarray<Type>.OpsT.Div, right);

        public static NDview<Type> Min<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Min);
        public static NDview<Type> Max<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Max);

        public static NDview<double> Eq<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Eq);
        public static NDview<double> Neq<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Neq);
        public static NDview<double> Lt<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Lt);
        public static NDview<double> Lte<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Lte);
        public static NDview<double> Gt<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Gt);
        public static NDview<double> Gte<Type>(NDview<Type> left, NDview<Type> right) => ElementWiseOp(left, right, NDarray<Type>.OpsT.Gte);
    }
}
