using System;
using NDarrayLib;
namespace DesertLand.Losses
{
    public interface IAccuracy<Type>
    {
        NDarray<double> Acc(NDarray<Type> y, NDarray<Type> p);
    }

    public class RoundAccuracy<Type> : IAccuracy<Type>
    {
        public NDarray<double> Acc(NDarray<Type> y, NDarray<Type> p) => ND.Eq(ND.Round<Type>(y), ND.Round<Type>(p)).Prod(1).Mean(0);
    }

    public class ArgmaxAccuracy<Type> : IAccuracy<Type>
    {
        public NDarray<double> Acc(NDarray<Type> y, NDarray<Type> p) => ND.Eq(ND.Argmax<Type>(y, 1), ND.Argmax<Type>(p, 1)).Mean(0);
    }
}
