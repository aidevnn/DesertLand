using System;
using System.Linq;
using DesertLand.Activations;
using DesertLand.Optimizers;
using NDarrayLib;

namespace DesertLand.Layers
{
    public class ActivationLayer<Type> : ILayer<Type>
    {
        readonly IActivation<Type> activation;
        public ActivationLayer(IActivation<Type> activation)
        {
            Name = activation.Name;
            this.activation = activation;
        }

        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        public int Params => 0;

        public string Name { get; set; }
        public bool IsTraining { get; set; }
        public NDarray<Type> LayerInput { get; set; }

        public NDarray<Type> Backward(NDarray<Type> accumGrad)
        {
            return accumGrad * activation.Grad(LayerInput);
        }

        public NDarray<Type> Forward(NDarray<Type> X, bool isTraining = true)
        {
            LayerInput = X.Copy;
            return activation.Func(X);
        }

        public void Initialize(IOptimizer<Type> optimizer) { }

        public void SetInputShape(int[] inputShape)
        {
            InputShape = inputShape.ToArray();
            OutputShape = inputShape.ToArray();
        }
    }

    public class SigmoidLayer<Type> : ActivationLayer<Type>
    {
        public SigmoidLayer() : base(new SigmoidActivation<Type>()) { }
    }

    public class TanhLayer<Type> : ActivationLayer<Type>
    {
        public TanhLayer() : base(new TanhActivation<Type>()) { }
    }
}
