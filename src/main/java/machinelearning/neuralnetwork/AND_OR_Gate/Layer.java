package machinelearning.neuralnetwork.AND_OR_Gate;

import java.util.Arrays;

public class Layer {

	int units;
	double[] output;
	double[] error;
	double[][] weight;

	public static void printLayers(Layer[] layers) {

		for (int i = 0; i < layers.length; ++i)
			System.out.println(layers[i].toString());
	}

	@Override
	public String toString() {
		return "Layer [units=" + units + ", output=" + Arrays.toString(output) + ", error=" + Arrays.toString(error)
				+ ", weight=" + Arrays.deepToString(weight) + "]";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(error);
		result = prime * result + Arrays.hashCode(output);
		result = prime * result + units;
		result = prime * result + Arrays.deepHashCode(weight);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Layer other = (Layer) obj;
		if (!Arrays.equals(error, other.error))
			return false;
		if (!Arrays.equals(output, other.output))
			return false;
		if (units != other.units)
			return false;
		if (!Arrays.deepEquals(weight, other.weight))
			return false;
		return true;
	}
}
