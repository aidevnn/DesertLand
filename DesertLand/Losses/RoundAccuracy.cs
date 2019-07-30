using System;
using NDarrayLib;

namespace DesertLand.Losses
{
    public class RoundAccuracy<Type> : IAccuracy<Type>
    {
        public NDarray<double> Acc(NDarray<Type> y, NDarray<Type> p) => ND.Eq(ND.Round<Type>(y), ND.Round<Type>(p)).Prod(1).Mean(0);
    }
}
