package machinelearning.neuralnetwork.AND_OR_Gate;

public enum ErrorFunction{

	SIMPLE_ERROR;

	public double getError(double target, double output) {
		return target - output;
	}

}
