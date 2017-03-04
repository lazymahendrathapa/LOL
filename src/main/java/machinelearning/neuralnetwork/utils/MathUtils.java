package machinelearning.neuralnetwork.utils;

public class MathUtils {

	private MathUtils(){
		
	}
	
	/**
	 * Return natural log without underflow.
	 * 
	 * @param value
	 * @return
	 */
	public static double calculateNaturalLog(double x){
		
		double y;

		if (x < 1E-300) {
			y = -690.7755;
		} else {
			y = Math.log(x);
		}
		return y;

	}

	/**
	 * Logistic sigmoid function.
	 * 
	 * @param x
	 * @return
	 */
	
	public static double calculateSigmoid(double x){

		return 1.0 / (1.0 + Math.exp(-1.0 * x));
	}
}
