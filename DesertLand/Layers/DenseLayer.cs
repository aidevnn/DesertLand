using System;
using System.Linq;
using DesertLand.Activations;
using DesertLand.Optimizers;
using NDarrayLib;

namespace DesertLand.Layers
{
    public class DenseLayer<Type> : ILayer<Type>
    {
        public DenseLayer(int outputNodes, bool useBiases = true)
        {
            this.useBiases = useBiases;
            OutputShape = new int[] { outputNodes };
        }

        public DenseLayer(int inputNodes, int outputNodes, bool useBiases = true) : this(outputNodes, useBiases)
        {
            InputShape = new int[] { inputNodes };
        }

        public DenseLayer(int outputNodes, IActivation<Type> activation, bool useBiases = true) : this(outputNodes, useBiases)
        {
            Name = $"DenseLayer-{activation.Name}";
            activationLayer = new ActivationLayer<Type>(activation);
        }

        public DenseLayer(int inputNodes, int outputNodes, IActivation<Type> activation, bool useBiases = true) : this(inputNodes, outputNodes, useBiases)
        {
            Name = $"DenseLayer-{activation.Name}";
            activationLayer = new ActivationLayer<Type>(activation);
        }

        private ActivationLayer<Type> activationLayer;

        readonly bool useBiases = true;
        public NDarray<Type> weights, biases;
        IOptimizer<Type> weightsOptmz, biasesOptmz;

        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        public int Params => weights.Count + (useBiases ? biases.Count : 0);

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
                weights = weightsOptmz.Update(weights, gW);

                if (useBiases)
                {
                    var gw0 = accumGrad.Sum(0, true);
                    biases = biasesOptmz.Update(biases, gw0);
                }
            }

            return ND.TensorDot<Type>(accumGrad, W);
        }

        public NDarray<Type> Forward(NDarray<Type> X, bool isTraining = true)
        {
            IsTraining = isTraining;
            LayerInput = X.Copy;

            NDarray<Type> X0 = useBiases ? ND.TensorDot<Type>(X, weights) + biases : ND.TensorDot<Type>(X, weights);
            if (activationLayer == null)
                return X0;

            return activationLayer.Forward(X0);
        }

        public void Initialize(IOptimizer<Type> optimizer)
        {
            weightsOptmz = optimizer.Clone();

            double lim = 1.0 / Math.Sqrt(InputShape[0]);

            weights = ND.Uniform(-lim, lim, InputShape[0], OutputShape[0]).Cast<Type>();
            if (useBiases)
            {
                biasesOptmz = optimizer.Clone();
                biases = ND.Zeros<Type>(1, OutputShape[0]);
            }
        }

        public void SetInputShape(int[] inputShape)
        {
            InputShape = inputShape.ToArray();
        }
    }
}
