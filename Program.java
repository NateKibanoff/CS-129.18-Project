import java.io.*;
import java.util.*;
public class Program{
	public static void main(String[] args) throws IOException{
		NeuralNetwork nn=new NeuralNetwork();
		BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
		PrintWriter pw=new PrintWriter("Error values.txt");
		PrintWriter pw2=new PrintWriter("Overall.txt");
		Scanner sc=new Scanner(System.in); //I got too lazy na
		int datapoints=0;
		Weights[] weights=null;
		System.out.print("Number of layers (including input and output): ");
		int layers=Integer.parseInt(br.readLine());
		int[] topology=new int[layers];
		System.out.println("Input topology (one integer per line): ");
		for(int i=0;i<layers;i++) topology[i]=Integer.parseInt(br.readLine());
		System.out.print("Filename: ");
		String filename=br.readLine();
		BufferedReader csv=new BufferedReader(new FileReader(filename));
		boolean[] activation=new boolean[layers-1];
		System.out.println("Activation functions starting from input layer (type either 'relu' or 'sigmoid' per line):");
		for(int i=0;i<activation.length;i++) activation[i]=br.readLine().equalsIgnoreCase("sigmoid");
		//csv.readLine(); //uncomment this if feature labels are included in the .csv file
		System.out.print("Will you input weights (type 'yes') or should they be randomized? (type 'no') ");
		if(br.readLine().equalsIgnoreCase("yes")){
			weights=new Weights[layers-1];
			for(int i=0;i<layers-1;i++){ //input everything in matrix form. One space between two values
				if(i==0) System.out.println("Weights between input layer (rows) and next layer (columns)");
				else if(i==layers-2) System.out.println("Final set of weights");
				else System.out.println("Weights between the next pair of hidden layers");
				double[][] bigat=new double[topology[i]][topology[i+1]];
				for(int j=0;j<topology[i];j++) for(int k=0;k<topology[i+1];k++) bigat[j][k]=sc.nextDouble();
				weights[i]=new Weights(bigat);
			}
		}
		System.out.print("How many rounds of training? ");
		int epochs=Integer.parseInt(br.readLine());
		for(int round=1;round<=epochs;round++){
			System.out.println("EPOCH "+round);
			pw.println("EPOCH "+round);
			pw2.print("EPOCH "+round+" ");
			double err_temp=0;
			int dp=0;
			while(csv.ready()){
				dp++;
				System.out.println("DATAPOINT "+dp);
				pw.println("DATAPOINT "+dp);
				StringTokenizer st=new StringTokenizer(csv.readLine(),",");
				double[][] input=new double[1][topology[0]];
				double[][] output=new double[1][topology[layers-1]];
				for(int i=0;i<topology[0];i++) input[0][i]=Double.parseDouble(st.nextToken());
				for(int i=0;i<topology[layers-1];i++) output[0][i]=Double.parseDouble(st.nextToken());
				double[] errors=nn.feedForward(input,output,topology,activation,weights);
				double another_temp=0;
				for(int i=0;i<errors.length;i++) another_temp+=errors[i];
				err_temp+=another_temp;
				System.out.print("Errors: ");
				for(int i=0;i<errors.length;i++){
					System.out.print(errors[i]+" ");
					pw.print(errors[i]+" ");
				}
				System.out.println();
				pw.println();
				System.out.println("Sum: "+another_temp);
				weights=nn.backProp(errors,activation);
			}
			System.out.println("Overall error: "+err_temp/dp);
			pw2.println("Overall error: "+err_temp/dp);
			csv=new BufferedReader(new FileReader(filename));
			System.out.println();
			pw.println();
		}
		pw.flush();
		pw2.flush();
	}
}