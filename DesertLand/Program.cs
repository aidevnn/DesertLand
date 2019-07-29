using System;

using NDarrayLib;

namespace DesertLand
{
    class MainClass
    {
        static void Test1()
        {
            NDarray<int> a = ND.Arange(15).Reshape(5, 3);
            var b = ND.Arange(4, 3);
            Console.WriteLine(a);
            Console.WriteLine(b);
            var c = a + 2 * b;
            Console.WriteLine(c);
        }

        static void Test2()
        {
            var a = ND.Uniform<double>(0, 1, 4, 1);
            var b = ND.Uniform<double>(0, 1, 4, 1);

            Func<NDview<double>, NDview<double>> f = x => -a * ND.Log(x) + (1 - a) * ND.Log(1 - x);
            Func<NDview<double>, double, NDview<double>> df = (x, h) => (f(x + h) - f(x)) / h;

            var c = f(b);
            var d = -a / b - (1 - a) / (1 - b);
            var e = df(b, 1e-12);

            Console.WriteLine(a);
            Console.WriteLine(b);
            Console.WriteLine(c);
            Console.WriteLine(d);
            Console.WriteLine(e);
            Console.WriteLine(ND.Abs(d - e));
        }

        static void Test3()
        {
            var a = ND.Uniform(1, 10, 2, 3, 4).CastCopy<double>();
            Console.WriteLine(a);

            Console.WriteLine(a.Sum());
            Console.WriteLine(a.Sum(0));
            Console.WriteLine(a.Sum(1));
            Console.WriteLine(a.Sum(2));

            Console.WriteLine(a.Prod());
            Console.WriteLine(a.Prod(0));
            Console.WriteLine(a.Prod(1));
            Console.WriteLine(a.Prod(2));

            Console.WriteLine(a.Mean());
            Console.WriteLine(a.Mean(0));
            Console.WriteLine(a.Mean(1));
            Console.WriteLine(a.Mean(2));

            //var a = ND.Uniform(1, 10, 4, 2, 2);
            //Console.WriteLine(a);
            //Console.WriteLine(a.Prod(2));
            //Console.WriteLine(a.Prod(2).Prod(1, true));
            //Console.WriteLine(a.Reshape(4, -1).Prod(1, true));
        }

        static void Test4()
        {
            var a = ND.Uniform(1, 10, 4, 4);
            var b = ND.Uniform(1, 10, 2, 4, 2);
            Console.WriteLine(a);
            Console.WriteLine(b);
            Console.WriteLine(ND.TensorDot<int>(a, b));
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            //Utils.DebugNumpy = Utils.DbgLvl1;

        }
    }
}
