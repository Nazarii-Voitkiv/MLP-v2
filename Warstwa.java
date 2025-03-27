public class Warstwa {
	Neuron [] neurony;
	int liczba_neuronow;
	private Neuron.ActivationFunction activationFunction;
	
	public Warstwa(){
		neurony = null;
		liczba_neuronow = 0;
		activationFunction = Neuron.ActivationFunction.SIGMOID;
	}
	
	public Warstwa(int liczba_wejsc, int liczba_neuronow){
		this(liczba_wejsc, liczba_neuronow, Neuron.ActivationFunction.SIGMOID);
	}
	
	public Warstwa(int liczba_wejsc, int liczba_neuronow, Neuron.ActivationFunction activationFunction){
		this.liczba_neuronow = liczba_neuronow;
		this.activationFunction = activationFunction;
		neurony = new Neuron[liczba_neuronow];
		for(int i = 0; i < liczba_neuronow; i++)
			neurony[i] = new Neuron(liczba_wejsc, activationFunction);
	}
	
	double [] oblicz_wyjscie(double [] wejscia){
		double [] wyjscie = new double[liczba_neuronow];
		
		// Розрахунок виходів нейронів
		for(int i = 0; i < liczba_neuronow; i++)
			wyjscie[i] = neurony[i].oblicz_wyjscie(wejscia);
		
		// Якщо це шар з Softmax активацією, застосовуємо Softmax до всього вектора
		if(activationFunction == Neuron.ActivationFunction.SOFTMAX) {
			applySoftmax(wyjscie);
		}
		
		return wyjscie;
	}
	
	// Метод для застосування функції Softmax до вектора
	private void applySoftmax(double[] vector) {
		// Знаходимо максимальне значення для чисельної стабільності
		double max = vector[0];
		for (int i = 1; i < vector.length; i++) {
			if (vector[i] > max) max = vector[i];
		}
		
		// Обчислюємо експоненти з вирахуванням максимального значення
		double sum = 0.0;
		for (int i = 0; i < vector.length; i++) {
			vector[i] = Math.exp(vector[i] - max);
			sum += vector[i];
		}
		
		// Нормалізуємо для отримання ймовірностей
		for (int i = 0; i < vector.length; i++) {
			vector[i] /= sum;
		}
	}
	
	public Neuron.ActivationFunction getActivationFunction() {
		return activationFunction;
	}
}
