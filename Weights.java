public class Weights{
	private double[][] w;
	public Weights(double[][] we){
		w=we;
	}
	public int getRows(){
		return w.length;
	}
	public int getColumns(){
		return w[0].length;
	}
	public double[][] toMatrix(){
		return w;
	}
}