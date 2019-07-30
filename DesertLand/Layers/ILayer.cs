using System;
using DesertLand.Activations;
using DesertLand.Optimizers;
using NDarrayLib;

namespace DesertLand.Layers
{
    public interface ILayer<Type>
    {
        int[] InputShape { get; set; }
        int[] OutputShape { get; set; }
        int Params { get; }
        string Name { get; set; }
        bool IsTraining { get; set; }

        NDarray<Type> LayerInput { get; set; }

        void Initialize(IOptimizer<Type> optimizer);
        void SetInputShape(int[] inputShape);

        NDarray<Type> Forward(NDarray<Type> X, bool isTraining);
        NDarray<Type> Backward(NDarray<Type> accumGrad);
    }
}
