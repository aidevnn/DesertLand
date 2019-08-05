using System;
using NDarrayLib;

namespace DesertLand.Losses
{
    public class SquareLoss<Type> : ILoss<Type>
    {
        public NDarray<Type> Grad(NDarray<Type> y, NDarray<Type> p) => p - y;
        public NDarray<Type> Loss(NDarray<Type> y, NDarray<Type> p) => ND.Sq(p - y) / 2;
    }
}
