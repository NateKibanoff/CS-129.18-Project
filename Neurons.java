public class Neurons{
	private double[] nodes;
	public Neurons(double[] n){
		nodes=n;
	}
	public int getLength(){
		return nodes.length;
	}
	public double get(int i){
		return nodes[i];
	}
	public double[][] toRow(){
		double[][] ret=new double[1][nodes.length];
		for(int i=0;i<nodes.length;i++) ret[0][i]=nodes[i];
		return ret;
	}
	public double[][] toColumn(){
		double[][] ret=new double[nodes.length][1];
		for(int i=0;i<nodes.length;i++) ret[i][0]=nodes[i];
		return ret;
	}
}