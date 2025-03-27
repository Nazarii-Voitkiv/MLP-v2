import java.util.Random;

public class Neuron {
	double [] wagi;
	int liczba_wejsc;
	private ActivationFunction activationFunction;
	
	// Перелік типів функцій активації
	public enum ActivationFunction {
		SIGMOID,
		RELU,
		SOFTMAX
	}

	public Neuron(){
		liczba_wejsc = 0;
		wagi = null;
		activationFunction = ActivationFunction.SIGMOID;
	}
	
	public Neuron(int liczba_wejsc){
		this(liczba_wejsc, ActivationFunction.SIGMOID);
	}
	
	public Neuron(int liczba_wejsc, ActivationFunction activationFunction){
		this.liczba_wejsc = liczba_wejsc;
		this.activationFunction = activationFunction;
		wagi = new double[liczba_wejsc+1];
		generuj();
	}
	
	private void generuj() {
		Random r = new Random();
		for(int i = 0; i <= liczba_wejsc; i++)
			//wagi[i] = (r.nextDouble()-0.5)*2.0*10;//do ogladania
			wagi[i] = (r.nextDouble()-0.5)*2.0*0.01;//do projektu
	}
	
	public double oblicz_wyjscie(double [] wejscia){
		double fi = wagi[0]; // bias
		for(int i = 1; i <= liczba_wejsc; i++)
			fi += wagi[i] * wejscia[i-1];
		
		return applyActivation(fi);
	}
	
	// Метод для застосування функції активації
	private double applyActivation(double value) {
		switch(activationFunction) {
			case SIGMOID:
				return sigmoid(value);
			case RELU:
				return relu(value);
			case SOFTMAX:
				// Softmax реалізується на рівні шару, тому тут просто повертаємо значення
				return value;
			default:
				return sigmoid(value);
		}
	}
	
	// Сигмоїдна функція активації
	private double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	// ReLU функція активації
	private double relu(double x) {
		return Math.max(0, x);
	}
	
	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}
}
