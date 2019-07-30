using System;
using NDarrayLib;
namespace DesertLand.Optimizers
{
    public class SGD<Type> : IOptimizer<Type>
    {
        public SGD(double lr = 0.01, double momentum = 0.0)
        {
            this.lr = lr;
            this.momentum = momentum;
        }

        readonly double lr, momentum;

        public IOptimizer<Type> Clone() => new SGD<Type>(lr, momentum);

        NDarray<Type> wUpdt;
        public NDarray<Type> Update(NDarray<Type> w, NDarray<Type> g)
        {
            if (wUpdt == null)
                wUpdt = ND.Zeros<Type>(w.Shape);

            wUpdt = momentum * wUpdt + (1 - momentum) * g;
            return w - lr * wUpdt;
        }
    }
}
