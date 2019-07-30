using System;
using NDarrayLib;

namespace DesertLand.Optimizers
{
    public interface IOptimizer<Type>
    {
        IOptimizer<Type> Clone();
        NDarray<Type> Update(NDarray<Type> w, NDarray<Type> g);
    }
}
