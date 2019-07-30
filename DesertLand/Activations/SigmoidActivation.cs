using System;
using NDarrayLib;

namespace DesertLand.Activations
{
    public class SigmoidActivation<Type> : IActivation<Type>
    {
        public string Name => "Sigmoid";
        public NDview<Type> Func(NDview<Type> X) => ND.Sigmoid(X);
        public NDview<Type> Grad(NDview<Type> X) => ND.DSigmoid(X);
    }
}
