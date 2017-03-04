package machinelearning.neuralnetwork.XORGate;

import java.util.Random;

import machinelearning.neuralnetwork.utils.MathUtils;

public class NeuralNetwork {

	private Random random = new Random();

	private ActivationFunction activationFunction = ActivationFunction.LOGISTIC_SIGMOID;
	private ErrorFunction errorFunction = ErrorFunction.LEAST_MEAN_SQUARE;

	private double learningRate = 0.2;
	private int numOfLayers;

	private Layer[] layers;
	private double[] target;

	public NeuralNetwork(int... numUnits) {

		this.numOfLayers = numUnits.length;

		target = new double[numUnits[numOfLayers - 1]];

		layers = new Layer[numOfLayers];

		for (int i = 0; i < numOfLayers; ++i) {

			layers[i] = new Layer();
			layers[i].units = numUnits[i];
			layers[i].output = new double[numUnits[i] + 1];
			layers[i].error = new double[numUnits[i] + 1];
			layers[i].output[numUnits[i]] = 1.0;
		}
		for (int i = 1; i < numOfLayers; ++i) {

			layers[i].weight = new double[numUnits[i]][numUnits[i - 1] + 1];

			for (int j = 0; j < numUnits[i]; ++j)
				for (int k = 0; k < numUnits[i - 1] + 1; ++k) {
					layers[i].weight[j][k] = random.nextDouble();
				}
		}

	}

	public void learn(double[][] inputValue, double[] targetValue, double errorThreshold, int numOfIteration) {

		for (int iteration = 0; iteration < numOfIteration; ++iteration) {

			double error = 0.0;

			for (int i = 0; i < inputValue.length; ++i) {
				error += learn(inputValue[i], targetValue[i]);
			}
			
			System.out.println("Iteration: " + iteration + " Error: " + error);

			if (error < errorThreshold)
				break;
		}

	}

	public double predict(double[] x) {

		setInput(x);
		propagate();

		if (layers[numOfLayers - 1].units == 1)
			return layers[numOfLayers - 1].output[0];

		double maximumValue = Double.NEGATIVE_INFINITY;
		int label = -1;

		for (int i = 0; i < layers[numOfLayers - 1].units; ++i) {
			if (layers[numOfLayers - 1].output[i] > maximumValue) {
				maximumValue = layers[numOfLayers - 1].output[i];
				label = i;
			}
		}

		return label;
	}

	private double learn(double[] inputValue, double targetValue) {

		if (layers[numOfLayers - 1].units == 1)
			target[0] = targetValue;

		return learn(inputValue, target);
	}

	private double learn(double[] inputValue, double[] targetValue) {

		setInput(inputValue);
		propagate();

		double error = computeError(targetValue, layers[numOfLayers - 1].error);

		backpropagate();
		adjustWeight();
		return error;
	}

	private void setInput(double[] inputValue) {
		System.arraycopy(inputValue, 0, layers[0].output, 0, inputValue.length);
	}

	private void propagate() {

		for (int l = 0; l < layers.length - 1; ++l) {
			propagate(layers[l], layers[l + 1]);
		}
	}

	private void propagate(Layer lowerLayer, Layer upperLayer) {

		for (int i = 0; i < upperLayer.units; ++i) {

			double sum = 0.0;

			for (int j = 0; j <= lowerLayer.units; ++j) {
				sum += lowerLayer.output[j] * upperLayer.weight[i][j];
			}

			if (activationFunction == ActivationFunction.LOGISTIC_SIGMOID) {
				upperLayer.output[i] = MathUtils.calculateSigmoid(sum);
			}
		}
	}

	public double computeError(double[] target, double[] outputLayerError) {

		double error = 0.0;

		for (int i = 0; i < layers[numOfLayers - 1].units; ++i) {

			double output = layers[numOfLayers - 1].output[i];

			double gradient = target[i] - output;

			if (errorFunction == ErrorFunction.LEAST_MEAN_SQUARE)
				error += 0.5 * gradient * gradient;

			if (errorFunction == ErrorFunction.LEAST_MEAN_SQUARE
					&& activationFunction == ActivationFunction.LOGISTIC_SIGMOID)
				gradient *= output * (1.0 - output);

			outputLayerError[i] = gradient;

		}

		return error;
	}

	private void backpropagate() {

		for (int i = numOfLayers; --i > 1;) {

			backpropagate(layers[i], layers[i - 1]);
		}
		
	}

	private void backpropagate(Layer upperLayer, Layer lowerLayer) {

		for (int i = 0; i <= lowerLayer.units; ++i) {

			double output = lowerLayer.output[i];
			double error = 0.0;

			for (int j = 0; j < upperLayer.units; ++j)
				error += upperLayer.weight[j][i] * upperLayer.error[j];

			lowerLayer.error[i] = output * (1.0 - output) * error;

		}
	}

	private void adjustWeight() {

		for (int i = 1; i < numOfLayers; ++i) {
			for (int j = 0; j < layers[i].units; ++j) {
				for (int k = 0; k <= layers[i - 1].units; ++k) {

					double output = layers[i - 1].output[k];
					double error = layers[i].error[j];
					double delta = learningRate * error * output;
					layers[i].weight[j][k] += delta;
				}
			}
		}
	}

}
