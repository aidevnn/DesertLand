using System;
using NDarrayLib;

namespace DesertLand.Losses
{
    public class ArgmaxAccuracy<Type> : IAccuracy<Type>
    {
        public NDarray<double> Acc(NDarray<Type> y, NDarray<Type> p) => ND.Eq(ND.Argmax<Type>(y, 1), ND.Argmax<Type>(p, 1)).Mean(0);
    }
}
