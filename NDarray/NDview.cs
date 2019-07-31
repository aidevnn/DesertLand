using System;
namespace NDarrayLib
{
    internal delegate NDarray<Type> Fnc<Type>();

    // Lazy evaluation
    public struct NDview<Type>
    {
        internal NDview(Fnc<Type> fnc0)
        {
            fnc = fnc0;
        }

        internal readonly Fnc<Type> fnc;

        public NDarray<Type> BaseArray => fnc();

        public NDarray<Type> Copy => fnc().Copy;

        private Type GetAt(int idx) => BaseArray.getAt(idx);

        public override string ToString() => BaseArray.ToString();

        public NDview<Type> Reshape(params int[] shape) => ND.Reshape(this, shape);
        public NDview<Type> Transpose(int[] table) => ND.Transpose(this, table);
        public NDview<Type> T => ND.Transpose(this);

        public NDview<U> Cast<U>() => ND.Cast<Type, U>(this);

        public NDview<Type> Sum(int axis = -1, bool keepdims = false) => ND.SumAxis(this, axis, keepdims);
        public NDview<Type> Prod(int axis = -1, bool keepdims = false) => ND.ProdAxis(this, axis, keepdims);
        public NDview<Type> Mean(int axis = -1, bool keepdims = false) => ND.MeanAxis(this, axis, keepdims);

        public double SumAll() => NDarray<double>.OpsT.Cast(Sum().GetAt(0));
        public double ProdAll() => NDarray<double>.OpsT.Cast(Prod().GetAt(0));
        public double MeanAll() => NDarray<double>.OpsT.Cast(Mean().GetAt(0));

        public static implicit operator NDarray<Type>(NDview<Type> nDview) => nDview.Copy;

        public static NDview<Type> operator +(NDview<Type> a, NDview<Type> b) => ND.Add(a, b);
        public static NDview<Type> operator +(NDarray<Type> a, NDview<Type> b) => ND.Add(a.View, b);
        public static NDview<Type> operator +(NDview<Type> a, NDarray<Type> b) => ND.Add(a, b.View);
        public static NDview<Type> operator +(double a, NDview<Type> b) => ND.Add(a, b);
        public static NDview<Type> operator +(NDview<Type> a, double b) => ND.Add(a, b);

        public static NDview<Type> operator -(NDview<Type> a) => ND.Neg(a);
        public static NDview<Type> operator -(NDview<Type> a, NDview<Type> b) => ND.Sub(a, b);
        public static NDview<Type> operator -(NDarray<Type> a, NDview<Type> b) => ND.Sub(a.View, b);
        public static NDview<Type> operator -(NDview<Type> a, NDarray<Type> b) => ND.Sub(a, b.View);
        public static NDview<Type> operator -(double a, NDview<Type> b) => ND.Sub(a, b);
        public static NDview<Type> operator -(NDview<Type> a, double b) => ND.Sub(a, b);

        public static NDview<Type> operator *(NDview<Type> a, NDview<Type> b) => ND.Mul(a, b);
        public static NDview<Type> operator *(NDarray<Type> a, NDview<Type> b) => ND.Mul(a.View, b);
        public static NDview<Type> operator *(NDview<Type> a, NDarray<Type> b) => ND.Mul(a, b.View);
        public static NDview<Type> operator *(double a, NDview<Type> b) => ND.Mul(a, b);
        public static NDview<Type> operator *(NDview<Type> a, double b) => ND.Mul(a, b);

        public static NDview<Type> operator /(NDview<Type> a, NDview<Type> b) => ND.Div(a, b);
        public static NDview<Type> operator /(NDarray<Type> a, NDview<Type> b) => ND.Div(a.View, b);
        public static NDview<Type> operator /(NDview<Type> a, NDarray<Type> b) => ND.Div(a, b.View);
        public static NDview<Type> operator /(double a, NDview<Type> b) => ND.Div(a, b);
        public static NDview<Type> operator /(NDview<Type> a, double b) => ND.Div(a, b);


    }
}
