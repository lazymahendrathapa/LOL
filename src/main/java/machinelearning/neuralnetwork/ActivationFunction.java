package machinelearning.neuralnetwork;

public enum ActivationFunction {

	/*
	 * But such a function is not differentiable because there is a
	 * discontinuity at zero. So in practice, when training with
	 * back-propagation, this function is not used. Instead, the sigmoid, or
	 * some other smooth function, is used.
	 * 
	 */
	STEP_FUNCTION;
}
