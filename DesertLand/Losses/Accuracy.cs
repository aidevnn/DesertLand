using System;
using NDarrayLib;
namespace DesertLand.Losses
{
    public interface IAccuracy<Type>
    {
        NDarray<double> Acc(NDarray<Type> y, NDarray<Type> p);
    }
}
