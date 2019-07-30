using System;
using NDarrayLib;
namespace DesertLand.Losses
{
    public class CrossEntropyLoss<Type> : ILoss<Type>
    {
        public NDarray<Type> Grad(NDarray<Type> y, NDarray<Type> p)
        {
            var p0 = ND.Clamp<Type>(p, 1e-6, 1 - 1e-6);
            return -y / p0 + (1 - y) / (1 - p0);
        }

        public NDarray<Type> Loss(NDarray<Type> y, NDarray<Type> p)
        {
            var p0 = ND.Clamp<Type>(p, 1e-6, 1 - 1e-6);
            return -y * ND.Log(p0) - (1 - y) * ND.Log(1 - p0);
        }
    }
}
