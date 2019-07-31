using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NDarrayLib
{
    public class NDarray<Type>
    {
        public static Operations<Type> OpsT;

        static NDarray()
        {
            if (typeof(Type) == typeof(int))
                OpsT = new OpsInt() as Operations<Type>;
            else if (typeof(Type) == typeof(float))
                OpsT = new OpsFloat() as Operations<Type>;
            else if (typeof(Type) == typeof(double))
                OpsT = new OpsDouble() as Operations<Type>;
            else
                throw new ArgumentException($"{typeof(Type).Name} is not supported. Only int, float or double");
        }

        public int[] Shape { get; set; }
        public int[] Strides { get; set; }
        public int[] Indices { get; set; }
        public int Count { get; set; }
        public bool OwnData { get; set; } = true;

        internal Func<int, Type> getAt;
        internal Action<int, Type> setAt;

        internal NDarray(Type v0, int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            Shape = shape.ToArray();
            Strides = Utils.Shape2Strides(Shape);
            Indices = new int[Shape.Length];
            Count = Utils.ArrMul(Shape);

            getAt = idx => v0;
            setAt = (idx, v) => { getAt = i => v; };
        }

        internal NDarray(Type[] data, int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            Shape = shape.ToArray();
            Strides = Utils.Shape2Strides(Shape);
            Indices = new int[Shape.Length];
            Count = Utils.ArrMul(Shape);

            SetData(data);
        }

        internal NDarray(int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            Shape = shape.ToArray();
            Strides = Utils.Shape2Strides(Shape);
            Indices = new int[Shape.Length];
            Count = Utils.ArrMul(Shape);

            getAt = idx => OpsT.Zero;
            setAt = (idx, v) => { getAt = i => v; };
        }

        internal NDarray(int[] shape, int[] strides)
        {
            if (shape.Length == 0)
            {
                shape = new int[] { 1 };
                strides = new int[] { 1 };
            }

            Shape = shape.ToArray();
            Strides = strides.ToArray();
            Indices = new int[Shape.Length];
            Count = Utils.ArrMul(Shape);

            getAt = idx => OpsT.Zero;
            setAt = (idx, v) => { getAt = i => v; };
        }

        internal NDarray(NDarray<Type> nDarray)
        {
            Shape = nDarray.Shape.ToArray();
            Strides = nDarray.Strides.ToArray();
            Indices = new int[Shape.Length];
            Count = Utils.ArrMul(Shape);

            Type[] data = Enumerable.Repeat(OpsT.Zero, Count).ToArray();
            SetData(data);
        }

        public void SetData(Type[] data)
        {
            if (Count != data.Length)
                throw new Exception();

            OwnData = true;
            getAt = idx => data[idx];
            setAt = (idx, v) => data[idx] = v;
        }

        public Type[] GetData => Enumerable.Range(0, Count).Select(getAt).ToArray();

        public NDarray<Type> this[int k]
        {
            get
            {
                var nd0 = new NDarray<Type>(Shape.Skip(1).ToArray());
                int offset = k * Strides[0];
                nd0.getAt = i => getAt(i + offset);
                nd0.setAt = (i, v) => setAt(i + offset, v);
                return nd0;
            }
        }

        public NDarray<Type> Copy
        {
            get
            {
                var nd0 = new NDarray<Type>(Shape);
                nd0.SetData(GetData);
                return nd0;
            }
        }

        public NDview<Type> View => new NDview<Type>(this);

        public NDview<U> Cast<U>() => View.Cast<U>();
        public NDarray<U> CastCopy<U>() => View.Cast<U>().Copy;

        public NDview<Type> Round(int dec = 0) => ND.Round(View, dec);
        public NDview<Type> Reshape(params int[] shape) => ND.Reshape(View, shape);
        public NDview<Type> Transpose(int[] table) => ND.Transpose(View, table);
        public NDview<Type> T => ND.Transpose<Type>(this);

        public NDview<Type> Sum(int axis = -1, bool keepdims = false) => View.Sum(axis, keepdims);
        public NDview<Type> Prod(int axis = -1, bool keepdims = false) => View.Prod(axis, keepdims);
        public NDview<Type> Mean(int axis = -1, bool keepdims = false) => View.Mean(axis, keepdims);

        public double SumAll() => View.SumAll();
        public double ProdAll() => View.ProdAll();
        public double MeanAll() => View.MeanAll();

        public static implicit operator NDview<Type>(NDarray<Type> nDarray) => nDarray.View;

        public static NDview<Type> operator +(NDarray<Type> a, NDarray<Type> b) => ND.Add<Type>(a, b);
        public static NDview<Type> operator +(double a, NDarray<Type> b) => ND.Add<Type>(a, b);
        public static NDview<Type> operator +(NDarray<Type> a, double b) => ND.Add<Type>(a, b);

        public static NDview<Type> operator -(NDarray<Type> a) => ND.Neg<Type>(a);
        public static NDview<Type> operator -(NDarray<Type> a, NDarray<Type> b) => ND.Sub<Type>(a, b);
        public static NDview<Type> operator -(double a, NDarray<Type> b) => ND.Sub<Type>(a, b);
        public static NDview<Type> operator -(NDarray<Type> a, double b) => ND.Sub<Type>(a, b);

        public static NDview<Type> operator *(NDarray<Type> a, NDarray<Type> b) => ND.Mul<Type>(a, b);
        public static NDview<Type> operator *(double a, NDarray<Type> b) => ND.Mul<Type>(a, b);
        public static NDview<Type> operator *(NDarray<Type> a, double b) => ND.Mul<Type>(a, b);

        public static NDview<Type> operator /(NDarray<Type> a, NDarray<Type> b) => ND.Div<Type>(a, b);
        public static NDview<Type> operator /(double a, NDarray<Type> b) => ND.Div<Type>(a, b);
        public static NDview<Type> operator /(NDarray<Type> a, double b) => ND.Div<Type>(a, b);

        public override string ToString()
        {
            var nargs = new int[Shape.Length];
            var strides = Utils.Shape2Strides(Shape);

            string result = "";
            var last = strides.Length == 1 ? Count : strides[strides.Length - 2];
            string before, after;

            List<Type> listValues = GetData.ToList();

            if (Utils.IsDebugLvl1)
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"Class:{GetType().Name,-20}");
                if (Shape.Length > 1 || Shape[0] != 1)
                {
                    sb.AppendLine($"Shape:({Shape.Glue()}) Version:{GetHashCode(),10}");
                    sb.AppendLine($"Strides:({Strides.Glue()})");
                    sb.AppendLine($"OwnData:({OwnData})");
                }

                string dbg = $" : np.array([{listValues.Glue(",")}], dtype={OpsT.dtype}).reshape({Shape.Glue(",")})";
                var nd = $"NDArray<{typeof(Type).Name}>";
                sb.AppendLine($"{nd,-20} {Shape.Glue("x")}{dbg}");
                //sb.AppendLine($"NB Get:{NbGet.Data} / {NbGet.Call}");
                Console.WriteLine(sb);
            }

            var ml0 = listValues.Select(v => $"{v}").Max(v => v.Length);
            var ml1 = listValues.Select(v => $"{v:F8}").Max(v => v.Length);
            string fmt = $"{{0,{ml0 + 2}}}";
            if (ml0 > ml1 + 3)
                fmt = $"{{0,{ml1 + 2}:F8}}";

            for (int idx = 0; idx < Count; ++idx)
            {
                after = before = "";

                if (idx % last == 0 || idx % last == last - 1)
                {
                    before = idx != 0 ? " " : "[";
                    after = idx == Count - 1 ? "]" : "";
                    for (int l = strides.Length - 2; l >= 0; --l)
                    {
                        if (idx % strides[l] == 0) before += "[";
                        else before = " " + before;

                        if (idx % strides[l] == strides[l] - 1) after += "]";
                    }
                }

                result += idx % last == 0 ? before : "";
                var val = listValues[idx];
                result += string.Format(fmt, val);
                result += idx % last == last - 1 ? after + "\n" : "";
                result += after.Length > 1 && idx != Count - 1 ? "\n" : "";
            }

            if (Utils.IsDebugNo)
            {
                result = result.Substring(0, result.Length - 1);
            }

            return result;
        }
    }
}
