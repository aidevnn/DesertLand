using System;
using NDarrayLib;

namespace DesertLand.Activations
{
    public class ReLuActivation<Type> : IActivation<Type>
    {
        public string Name => "ReLu";

        public NDview<Type> Func(NDview<Type> X) => ND.Gte(X, ND.Zeros<Type>(1)).Cast<Type>() * X;

        public NDview<Type> Grad(NDview<Type> X) => ND.Gte(X, ND.Zeros<Type>(1)).Cast<Type>();
    }
}
