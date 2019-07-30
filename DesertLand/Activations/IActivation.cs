using System;
using NDarrayLib;
namespace DesertLand.Activations
{
    public interface IActivation<Type>
    {
        string Name { get; }
        NDview<Type> Func(NDview<Type> X);
        NDview<Type> Grad(NDview<Type> X);
    }
}
