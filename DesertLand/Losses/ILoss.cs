using System;
using NDarrayLib;
namespace DesertLand.Losses
{
    public interface ILoss<Type>
    {
        NDarray<Type> Loss(NDarray<Type> y, NDarray<Type> p);
        NDarray<Type> Grad(NDarray<Type> y, NDarray<Type> p);
    }
}
