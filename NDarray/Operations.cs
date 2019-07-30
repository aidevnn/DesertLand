using System;

namespace NDarrayLib
{
    public abstract class Operations<Type>
    {
        public Type One, Zero, Epsilon, Minvalue, Maxvalue;
        public string dtype;
        public abstract Type Neg(Type a);
        public abstract Type Add(Type a, Type b);
        public abstract Type Sub(Type a, Type b);
        public abstract Type Mul(Type a, Type b);
        public abstract Type Div(Type a, Type b);

        public abstract Type Exp(Type x);
        public abstract Type Log(Type x);
        public abstract Type Abs(Type x);
        public Type Sq(Type x) => Mul(x, x);
        public abstract Type Sqrt(Type x);
        public abstract Type Tanh(Type x);
        public Type DTanh(Type x) => Sub(One, Sq(Tanh(x)));
        public Type Sigmoid(Type x) => Div(One, Add(One, Exp(Neg(x))));
        public Type DSigmoid(Type x) => Mul(Sigmoid(x), Sub(One, Sigmoid(x)));
        public abstract Type Round(Type x, int d = 0);

        public abstract Type Min(Type x, Type y);
        public abstract Type Max(Type x, Type y);
        public abstract Type Rand(Type min, Type max);

        public abstract double Eq(Type x, Type y);
        public abstract double Neq(Type x, Type y);

        public abstract double Gt(Type x, Type y);
        public abstract double Lt(Type x, Type y);

        public abstract double Gte(Type x, Type y);
        public abstract double Lte(Type x, Type y);

        public Type Clamp(Type x, double min, double max) => Max(Cast(min), Min(x, Cast(max)));
        public Type Cast<U>(U x) => (Type)Convert.ChangeType(x, typeof(Type));
    }

    public class OpsInt : Operations<int>
    {
        public OpsInt() { One = 1; Zero = 0; Epsilon = 0; dtype = "np.int64"; Minvalue = int.MinValue; Maxvalue = int.MaxValue; }
        public override int Neg(int a) => -a;
        public override int Add(int a, int b) => a + b;
        public override int Sub(int a, int b) => a - b;
        public override int Mul(int a, int b) => a * b;
        public override int Div(int a, int b) => a / b;

        public override int Exp(int x) => throw new NotImplementedException();
        public override int Log(int x) => throw new NotImplementedException();
        public override int Abs(int x) => Math.Abs(x);
        public override int Sqrt(int x) => (int)Math.Sqrt(x);
        public override int Tanh(int x) => throw new NotImplementedException();
        public override int Round(int x, int d = 0) => x;

        public override int Min(int x, int y) => Math.Min(x, y);
        public override int Max(int x, int y) => Math.Max(x, y);
        public override int Rand(int min, int max) => Utils.GetRandom.Next(min, max);

        public override double Eq(int x, int y) => x == y ? 1 : 0;
        public override double Neq(int x, int y) => x != y ? 1 : 0;
        public override double Gt(int x, int y) => x > y ? 1 : 0;
        public override double Gte(int x, int y) => x >= y ? 1 : 0;
        public override double Lt(int x, int y) => x < y ? 1 : 0;
        public override double Lte(int x, int y) => x <= y ? 1 : 0;

    }

    public class OpsFloat : Operations<float>
    {
        public OpsFloat() { One = 1; Zero = 0; Epsilon = 1e-6f; dtype = "np.float32"; Minvalue = float.MinValue; Maxvalue = float.MaxValue; }
        public override float Neg(float a) => -a;
        public override float Add(float a, float b) => a + b;
        public override float Sub(float a, float b) => a - b;
        public override float Mul(float a, float b) => a * b;
        public override float Div(float a, float b) => a / b;

        public override float Exp(float x) => (float)Math.Exp(x);
        public override float Log(float x) => (float)Math.Log(x);
        public override float Abs(float x) => Math.Abs(x);
        public override float Sqrt(float x) => (float)Math.Sqrt(x);
        public override float Tanh(float x) => (float)Math.Tanh(x);
        public override float Round(float x, int d = 0) => (float)Math.Round(x, d);

        public override float Min(float x, float y) => Math.Min(x, y);
        public override float Max(float x, float y) => Math.Max(x, y);
        public override float Rand(float min, float max) => (float)(min + (max - min) * Utils.GetRandom.NextDouble());

        public override double Eq(float x, float y) => Math.Abs(x - y) <= Epsilon ? 1 : 0;
        public override double Neq(float x, float y) => Math.Abs(x - y) > Epsilon ? 1 : 0;
        public override double Gt(float x, float y) => x > y ? 1 : 0;
        public override double Gte(float x, float y) => x >= y ? 1 : 0;
        public override double Lt(float x, float y) => x < y ? 1 : 0;
        public override double Lte(float x, float y) => x <= y ? 1 : 0;
    }

    public class OpsDouble : Operations<double>
    {
        public OpsDouble() { One = 1; Zero = 0; Epsilon = 1e-6; dtype = "np.float64"; Minvalue = double.MinValue; Maxvalue = double.MaxValue; }
        public override double Neg(double a) => -a;
        public override double Add(double a, double b) => a + b;
        public override double Sub(double a, double b) => a - b;
        public override double Mul(double a, double b) => a * b;
        public override double Div(double a, double b) => a / b;

        public override double Exp(double x) => Math.Exp(x);
        public override double Log(double x) => Math.Log(x);
        public override double Abs(double x) => Math.Abs(x);
        public override double Sqrt(double x) => Math.Sqrt(x);
        public override double Tanh(double x) => Math.Tanh(x);
        public override double Round(double x, int d = 0) => Math.Round(x, d);

        public override double Min(double x, double y) => Math.Min(x, y);
        public override double Max(double x, double y) => Math.Max(x, y);
        public override double Rand(double min, double max) => min + (max - min) * Utils.GetRandom.NextDouble();

        public override double Eq(double x, double y) => Math.Abs(x - y) <= Epsilon ? 1.0 : 0.0;
        public override double Neq(double x, double y) => Math.Abs(x - y) > Epsilon ? 1.0 : 0.0;
        public override double Gt(double x, double y) => x > y ? 1.0 : 0.0;
        public override double Gte(double x, double y) => x >= y ? 1.0 : 0.0;
        public override double Lt(double x, double y) => x < y ? 1.0 : 0.0;
        public override double Lte(double x, double y) => x <= y ? 1.0 : 0.0;
    }

}
