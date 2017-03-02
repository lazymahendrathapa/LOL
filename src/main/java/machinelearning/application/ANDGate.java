package machinelearning.application;

import machinelearning.neuralnetwork.ActivationFunction;
import machinelearning.neuralnetwork.NeuralNetwork;
import machinelearning.neuralnetwork.lossfunction.SimpleErrorLossFunction;

public class ANDGate {

	public static void main(String[] args) {

		NeuralNetwork andGate = new NeuralNetwork(SimpleErrorLossFunction.SIMPLE_ERROR,
				ActivationFunction.STEP_FUNCTION, 2, 1);

		double[][] inputValue = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		double[] targetValue = { 0, 0, 0, 1 };
		
		andGate.learn(inputValue, targetValue, 0.000001, 10000);
		
		System.out.println("Output: " + andGate.predict(new double[] { 1, 1 }));
		System.out.println("Output: " + andGate.predict(new double[] { 1, 0 }));
		System.out.println("Output: " + andGate.predict(new double[] { 0, 1 }));
		System.out.println("Output: " + andGate.predict(new double[] { 0, 0 }));
	}
}
