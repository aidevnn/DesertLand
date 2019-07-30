using System;
using NDarrayLib;

namespace DesertLand.Activations
{
    public class TanhActivation<Type> : IActivation<Type>
    {
        public string Name => "Tanh";
        public NDview<Type> Func(NDview<Type> X) => ND.Tanh(X);
        public NDview<Type> Grad(NDview<Type> X) => ND.DTanh(X);
    }
}
