using System;
using System.Linq;
using DesertLand.Activations;
using DesertLand.Optimizers;
using NDarrayLib;

namespace DesertLand.Layers
{
    public class DenseLayer<Type> : ILayer<Type>
    {
        public DenseLayer(int outputNodes)
        {
            OutputShape = new int[] { outputNodes };
        }

        public DenseLayer(int inputNodes, int outputNodes) : this(outputNodes)
        {
            InputShape = new int[] { inputNodes };
        }

        public DenseLayer(int outputNodes, IActivation<Type> activation) : this(outputNodes)
        {
            Name = $"DenseLayer-{activation.Name}";
            activationLayer = new ActivationLayer<Type>(activation);
        }

        public DenseLayer(int inputNodes, int outputNodes, IActivation<Type> activation) : this(inputNodes, outputNodes)
        {
            Name = $"DenseLayer-{activation.Name}";
            activationLayer = new ActivationLayer<Type>(activation);
        }

        private ActivationLayer<Type> activationLayer;

        public NDarray<Type> weights, biases;
        IOptimizer<Type> weightsOptmz, biasesOptmz;

        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        public int Params => weights.Count + biases.Count;

        public string Name { get; set; } = "DenseLayer";
        public bool IsTraining { get; set; }
        public NDarray<Type> LayerInput { get; set; }

        public NDarray<Type> Backward(NDarray<Type> accumGrad)
        {
            if (activationLayer != null)
                accumGrad = activationLayer.Backward(accumGrad);

            NDarray<Type> W = weights.T;
            if (IsTraining)
            {
                var gW = ND.TensorDot(LayerInput.T, accumGrad);
                var gw0 = accumGrad.Sum(0, true);

                weights = weightsOptmz.Update(weights, gW);
                biases = biasesOptmz.Update(biases, gw0);
            }

            return ND.TensorDot<Type>(accumGrad, W);
        }

        public NDarray<Type> Forward(NDarray<Type> X, bool isTraining = true)
        {
            IsTraining = isTraining;
            LayerInput = X.Copy;

            NDarray<Type> X0 = ND.TensorDot<Type>(X, weights) + biases;
            if (activationLayer == null)
                return X0;

            return activationLayer.Forward(X0);
        }

        public void Initialize(IOptimizer<Type> optimizer)
        {
            weightsOptmz = optimizer.Clone();
            biasesOptmz = optimizer.Clone();

            double lim = 1.0 / Math.Sqrt(InputShape[0]);

            weights = ND.Uniform(-lim, lim, InputShape[0], OutputShape[0]).Cast<Type>();
            biases = ND.Zeros<Type>(1, OutputShape[0]);
        }

        public void SetInputShape(int[] inputShape)
        {
            InputShape = inputShape.ToArray();
        }
    }
}
