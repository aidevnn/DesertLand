using System;
using System.Collections.Generic;
using System.Linq;

namespace NDarrayLib
{
    public static partial class ND
    {
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

        public static NDview<Type> SumAxis<Type>(NDview<Type> nDview, int axis = -1, bool keepdims = false)
            => AxisOps(nDview, axis, keepdims, NDarray<Type>.OpsT.Add, NDarray<Type>.OpsT.Zero);
        public static NDview<Type> ProdAxis<Type>(NDview<Type> nDview, int axis = -1, bool keepdims = false)
            => AxisOps(nDview, axis, keepdims, NDarray<Type>.OpsT.Mul, NDarray<Type>.OpsT.One);
        public static NDview<Type> MeanAxis<Type>(NDview<Type> nDview, int axis = -1, bool keepdims = false)
            => AxisOps(nDview, axis, keepdims, NDarray<Type>.OpsT.Add, NDarray<Type>.OpsT.Zero, true);
    }
}
