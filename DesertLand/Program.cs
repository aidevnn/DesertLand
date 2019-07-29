using System;

using NDarrayLib;

namespace DesertLand
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            Utils.DebugNumPy = true;

            //NDarray<int> a = ND.Arange(15).Reshape(5, 3);
            //var b = ND.Arange(4, 3);
            //Console.WriteLine(a);
            //Console.WriteLine(b);

            ////b.SetAt(4, 20);
            ////Console.WriteLine(a);
            ////Console.WriteLine(b);

            //var c = a + b;
            //Console.WriteLine(c);
            var a = ND.Uniform<double>(0, 1, 4, 1);
            var b = ND.Uniform<double>(0, 1, 4, 1);
            var b0 = b + 1e-12;

            Func<NDview<double>, NDview<double>> f = x => -a * ND.Log(x) + (1 - a) * ND.Log(1 - x);

            NDarray<double> c = f(b);
            NDarray<double> d = -a / b - (1 - a) / (1 - b);
            NDarray<double> e = (f(b0) - f(b)) / 1e-12;
            Console.WriteLine(a);
            Console.WriteLine(b);
            Console.WriteLine(c);
            Console.WriteLine(d);
            Console.WriteLine(e);
            Console.WriteLine(ND.Abs(d - e));
        }
    }
}
