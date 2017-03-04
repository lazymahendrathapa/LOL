package machinelearning.neuralnetwork.XORGate;

public class XORGate {

	public static void main(String[] args) {
		
		NeuralNetwork xorGate = new NeuralNetwork(2, 2, 1);
	
		double[][] inputValue = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		double[] targetValue = { 0, 1, 1, 0 };
	
		xorGate.learn(inputValue, targetValue, 0.0001, 100000);
		
		System.out.println("Output: " + xorGate.predict(new double[] { 1, 1 }));
		System.out.println("Output: " + xorGate.predict(new double[] { 1, 0 }));
		System.out.println("Output: " + xorGate.predict(new double[] { 0, 1 }));
		System.out.println("Output: " + xorGate.predict(new double[] { 0, 0 }));
		
	
	}

}
