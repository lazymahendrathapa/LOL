package machinelearning.neuralnetwork;

import java.util.Arrays;

public class Layer {

	int units;
	double[] output;
	double[] error;
	double[][] weight;
	double[][] delta;
	
	public int getUnits() {
		return units;
	}
	public void setUnits(int units) {
		this.units = units;
	}
	public double[] getOutput() {
		return output;
	}
	public void setOutput(double[] output) {
		this.output = output;
	}
	public double[] getError() {
		return error;
	}
	public void setError(double[] error) {
		this.error = error;
	}
	public double[][] getWeight() {
		return weight;
	}
	public void setWeight(double[][] weight) {
		this.weight = weight;
	}
	public double[][] getDelta() {
		return delta;
	}
	public void setDelta(double[][] delta) {
		this.delta = delta;
	}
	
	@Override
	public String toString() {
		return "Layer [units=" + units + ", output=" + Arrays.toString(output) + ", error=" + Arrays.toString(error)
				+ ", weight=" + Arrays.toString(weight) + ", delta=" + Arrays.toString(delta) + "]";
	}
}
