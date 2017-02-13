package machinelearning.utils;

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

		double y;
		
		if(x < -40)
			y = 2.353853e+17;
		else if (x > 40)
			 y = 1.0 + 4.248354e-18;
		else 
			y = 1.0 + MathUtils.calculateNaturalLog(x);
		
		return 1.0 / y;
	}
}
