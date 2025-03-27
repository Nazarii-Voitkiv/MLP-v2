import java.io.Serializable;
import java.util.Random;

public class Neuron implements Serializable {
    private static final long serialVersionUID = 1L;
    
	double [] wagi;
	int liczba_wejsc;
	private ActivationFunction activationFunction;
	private double output; // Зберігаємо останній вихід нейрону
	private double[] lastInputs; // Зберігаємо останні входи для backpropagation
	private double delta; // Помилка для backpropagation
	
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
		// Зберігаємо вхідні дані для backpropagation
		lastInputs = new double[wejscia.length];
		System.arraycopy(wejscia, 0, lastInputs, 0, wejscia.length);
		
		double fi = wagi[0]; // bias
		for(int i = 1; i <= liczba_wejsc; i++)
			fi += wagi[i] * wejscia[i-1];
		
		output = applyActivation(fi);
		return output;
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
	
	// Метод для обчислення похідної функції активації
	public double derivativeOfActivation(double output) {
		switch(activationFunction) {
			case SIGMOID:
				return output * (1 - output); // Похідна sigmoid
			case RELU:
				return output > 0 ? 1 : 0; // Похідна ReLU
			case SOFTMAX:
				// Похідна softmax обчислюється в шарі
				return 1.0;
			default:
				return output * (1 - output);
		}
	}
	
	// Методи для backpropagation
	public void updateDelta(double error) {
		delta = error * derivativeOfActivation(output);
	}
	
	public void updateWeights(double learningRate) {
		// Оновлюємо bias
		wagi[0] += learningRate * delta;
		
		// Оновлюємо ваги
		for(int i = 1; i <= liczba_wejsc; i++) {
			wagi[i] += learningRate * delta * lastInputs[i-1];
		}
	}
	
	// Метод для оновлення конкретної ваги з урахуванням регуляризації
	// Покращений метод оновлення ваги з регуляризацією
	public void updateWeightWithIndex(int index, double deltaUpdate, double regularizationTerm) {
		// Застосування L2-регуляризації: w = w * (1 - η*λ) + η*δ*x
		double newWeight = wagi[index] * (1.0 - regularizationTerm) + deltaUpdate;
		
		// Перевірка на NaN та обмеження максимального значення ваги
		if (Double.isNaN(newWeight) || Double.isInfinite(newWeight)) {
			System.err.println("УВАГА: NaN або Infinity в оновленні ваг. Пропускаємо оновлення.");
			return;
		}
		
		// Обмеження максимального значення ваги для запобігання вибуху градієнтів
		double maxWeight = 10.0;
		if (Math.abs(newWeight) > maxWeight) {
			newWeight = Math.signum(newWeight) * maxWeight;
		}
		
		wagi[index] = newWeight;
	}

	// Додаємо метод для обмеження величини дельти
	public void clipDelta(double maxAbsValue) {
		if (delta > maxAbsValue) {
			delta = maxAbsValue;
		} else if (delta < -maxAbsValue) {
			delta = -maxAbsValue;
		}
	}

	// Метод для отримання конкретного входу
	public double getLastInput(int index) {
		return (lastInputs != null && index >= 0 && index < lastInputs.length) ? lastInputs[index] : 0.0;
	}

	// Getters and setters
	public double getOutput() {
		return output;
	}
	
	public double getDelta() {
		return delta;
	}
	
	public double[] getWeights() {
		return wagi;
	}
	
	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}
}
