package machinelearning.neuralnetwork.lossfunction;

@FunctionalInterface
public interface LossFunction {

	public double getError(double target, double output);
}
