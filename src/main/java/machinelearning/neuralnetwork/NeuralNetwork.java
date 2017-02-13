package machinelearning.neuralnetwork;

import machinelearning.utils.MathUtils;

public class NeuralNetwork {

	private ErrorFunction errorFunction;
	private ActivationFunction activationFunction;

	private Layer[] network;

	private Layer inputLayer;
	private Layer outputLayer;

	private double learningRate = 0.1;
	private double momentumFactor = 0.0;
	private double weightDecayFactor = 0.0;

	private double[] target;

	public NeuralNetwork(ErrorFunction errorFunction, int... numUnits) {

		int numLayers = numUnits.length;

		for (int i = 0; i < numLayers; i++) {
			if (numUnits[i] < 1) {
				throw new IllegalArgumentException(
						String.format("Invalid number of units of layer %d: %d", i + 1, numUnits[i]));
			}
		}

		if (numUnits[numLayers - 1] < 2)
			throw new IllegalArgumentException(
					"Network only support mulitclass classification: " + numUnits[numLayers - 1]);

		this.errorFunction = errorFunction;
		this.activationFunction = this.getActivationFunction(errorFunction);

		if (errorFunction == ErrorFunction.CROSS_ENTROPY) {
			this.momentumFactor = 0.0;
			this.weightDecayFactor = 0.0;
		}

		this.target = new double[numUnits[numLayers - 1]];

		network = new Layer[numLayers];

		for (int i = 0; i < numLayers; ++i) {

			network[i] = new Layer();
			network[i].units = numUnits[i];
			network[i].output = new double[numUnits[i] + 1];
			network[i].error = new double[numUnits[i] + 1];
			network[i].output[numUnits[i]] = 1.0;
		}

		inputLayer = network[0];
		outputLayer = network[numLayers - 1];

		// Initialize random weight

		for (int i = 1; i < numLayers; ++i) {
			network[i].weight = new double[numUnits[i]][numUnits[i - 1] + 1];
			network[i].delta = new double[numUnits[i]][numUnits[i - 1] + 1];

			double range = 1.0 / Math.sqrt(network[i - 1].units);

			for (int j = 0; j < network[i].units; ++j)
				for (int k = 0; k < network[i - 1].units; ++k)
					network[i].weight[j][k] = Math.random() * (range + range) - range;
		}
	}

	public void setLearningRate(double learningRate) {

		if (learningRate <= 0)
			throw new IllegalArgumentException("Invalid learning rate: " + learningRate);

		this.learningRate = learningRate;
	}

	public double getLearningRate() {

		return learningRate;
	}

	public void setMomentumFactor(double momentumFactor) {

		if (momentumFactor < 0.0 || momentumFactor >= 1.0)
			throw new IllegalArgumentException("Invalid momentum factor: " + momentumFactor);

		this.momentumFactor = momentumFactor;
	}

	public double getMomentumFactor() {

		return momentumFactor;
	}

	public void setWeightDecayFactor(double weightDecayFactor) {

		if (weightDecayFactor < 0.0 || weightDecayFactor > 0.1)
			throw new IllegalArgumentException("Invalid weight decay factor: " + weightDecayFactor);

		this.weightDecayFactor = weightDecayFactor;
	}

	public double getWeightDecayFactor() {

		return weightDecayFactor;
	}

	private ActivationFunction getActivationFunction(ErrorFunction errorFunction) {

		if (errorFunction == ErrorFunction.CROSS_ENTROPY)
			return ActivationFunction.SOFTMAX;
		else
			return ActivationFunction.LOGISTIC_SIGMOID;
	}

	public void learn(double[][] x, int[] y) {

		int lengthTraningInstance = x.length;

		for (int i = 0; i < lengthTraningInstance; ++i) {
			learn(x[i], y[i]);
		}
	}

	public void learn(double[] x, int y) {

		if (outputLayer.units > 1 && y >= outputLayer.units) {
			throw new IllegalArgumentException("Invalid class label: " + y);
		}

		if (errorFunction == ErrorFunction.CROSS_ENTROPY) {

			for (int i = 0; i < target.length; ++i)
				target[i] = 0.0;

			target[y] = 1.0;

		} else {

			for (int i = 0; i < target.length; ++i)
				target[i] = 0.1;

			target[y] = 0.9;
		}

		learn(x, target);
	}

	public double learn(double[] x, double[] target) {

		setInput(x);
		propagate();
		
		double error = computeOutputError(target, outputLayer.error);

		backpropagate();
		adjustWeight();
		
		return error;
	}
	

	private void adjustWeight(){
	
		for(int l=1; l<network.length; ++l){
			for(int i=0; i<network[l].units; ++i ){
				for(int j=0; j<network[l-1].units; ++j){
					double output = network[l-1].output[j];
					double error = network[l].error[i];
					double delta = (1.0- momentumFactor) * learningRate * error * output + momentumFactor * network[l].delta[i][j];
					network[l].delta[i][j] = delta;
					network[l].weight[i][j] += delta;
					
					if(weightDecayFactor != 0.0 && j<network[l-1].units){
						network[l].weight[i][j] *= (1.0 - learningRate * weightDecayFactor);
					}
				}
			}
		}
	}

	private void backpropagate(){
		for(int l = network.length; --l > 0;){
			backpropagate(network[l], network[l-1]);
		}
	}
	
	private void backpropagate(Layer upper, Layer lower){
	
		for(int i=0; i <= lower.units; ++i){
			
			double output = lower.output[i];
			double error = 0.0;
			
			for(int j=0; j< upper.units; ++j)
				error += upper.weight[j][i] * upper.error[j];
			
			lower.error[i] = output * (1.0 - output) * error;
		}
	}
	
	
	private double computeOutputError(double[] target, double[] gradient){

		if(target.length != outputLayer.units)
			throw new IllegalArgumentException(String.format("Invalid output vector size: %d, expected: %d", target.length, outputLayer.units));
		
		double error = 0.0;
		
		for(int i = 0; i<outputLayer.units; ++i){
	
			double g = target[i] - outputLayer.output[i];
					
			if(errorFunction == ErrorFunction.LEAST_MEAN_SQUARE)
				error += 0.5 * Math.pow(g, 2);
				
			else if(errorFunction == ErrorFunction.CROSS_ENTROPY)
				error -= target[i] * MathUtils.calculateNaturalLog(outputLayer.output[i]);
			
			if(errorFunction == ErrorFunction.LEAST_MEAN_SQUARE && activationFunction == ActivationFunction.LOGISTIC_SIGMOID)
				g *= outputLayer.output[i] * (1.0 - outputLayer.output[i]);
			
			gradient[i] = g;
		}

		return error;
	}
	
	private void setInput(double[] x) {

		if (x.length != inputLayer.units)
			throw new IllegalArgumentException(
					String.format("Invalid input vector size: %d, expected: %d", x.length, inputLayer.units));

		System.arraycopy(x, 0, inputLayer, 0, inputLayer.units);
	}

	private void propagate() {

		for (int l = 0; l < network.length - 1; ++l)
			propagate(network[l], network[l + 1]);
	}

	private void propagate(Layer lower, Layer upper) {

		for (int i = 0; i < upper.units; ++i) {

			double sum = 0.0;

			for (int j = 0; j <= lower.units; ++j)
				sum += upper.weight[i][j] * upper.output[j];

			if (upper != outputLayer || activationFunction == ActivationFunction.LOGISTIC_SIGMOID)
				upper.output[i] = MathUtils.calculateSigmoid(sum);
			else
				upper.output[i] = sum;
		}

		if (upper == outputLayer && activationFunction == ActivationFunction.SOFTMAX) 
			softmax();
	}

	private void softmax() {

		double max = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < outputLayer.units; ++i)
			if (outputLayer.output[i] > max)
				max = outputLayer.output[i];

		double sum = 0.00000000000000001;

		for (int i = 0; i < outputLayer.units; ++i) {

			double output = Math.exp(outputLayer.output[i] - max);
			outputLayer.output[i] = output;
			sum += output;

		}

		for (int i = 0; i < outputLayer.units; ++i)
			outputLayer.output[i] /= sum;

	}
	
	public int predict(double[] x){
		
		setInput(x);
		propagate();
	
		double max = Double.NEGATIVE_INFINITY;
		int label = -1;
		
		for(int i=0; i<outputLayer.units; ++i){
			if(outputLayer.output[i] > max){
				max = outputLayer.output[i];
				label = i;
			}
		}
		
		return label;
	}
	public int predict(double[] x, double[] y){
		
		setInput(x);
		propagate();
		getOutput(y);
	
		double max = Double.NEGATIVE_INFINITY;
		int label = -1;
		
		for(int i=0; i<outputLayer.units; ++i){
			if(outputLayer.output[i] > max){
				max = outputLayer.output[i];
				label = i;
			}
		}
		
		return label;
	}
	
	private void getOutput(double[] y){
		if(y.length != outputLayer.units){
			throw new IllegalArgumentException(String.format("Invalid output vector size: %d, expected: %d", y.length, outputLayer.units));
		}
		
		System.arraycopy(outputLayer.output, 0, y, 0, outputLayer.units);
	}
}