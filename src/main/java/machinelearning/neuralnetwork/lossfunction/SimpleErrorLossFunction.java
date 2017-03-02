package machinelearning.neuralnetwork.lossfunction;

public enum SimpleErrorLossFunction implements LossFunction {

	SIMPLE_ERROR;

	@Override
	public double getError(double target, double output) {
		return target - output;
	}

}
