public class Siec {
	Warstwa [] warstwy;
	int liczba_warstw;
	
	public Siec(){
		warstwy = null;
		this.liczba_warstw = 0;
	}
	
	public Siec(int liczba_wejsc, int liczba_warstw, int [] lnww){
		this(liczba_wejsc, liczba_warstw, lnww, null);
	}
	
	// Конструктор з можливістю вказати функції активації для шарів
	public Siec(int liczba_wejsc, int liczba_warstw, int [] lnww, Neuron.ActivationFunction[] activationFunctions){
		this.liczba_warstw = liczba_warstw;
		warstwy = new Warstwa[liczba_warstw];
		
		for(int i = 0; i < liczba_warstw; i++) {
			// Якщо передані функції активації, використовуємо їх
			if(activationFunctions != null && i < activationFunctions.length) {
				warstwy[i] = new Warstwa((i==0) ? liczba_wejsc : lnww[i-1], lnww[i], activationFunctions[i]);
			} else {
				// Інакше використовуємо Sigmoid за замовчуванням
				warstwy[i] = new Warstwa((i==0) ? liczba_wejsc : lnww[i-1], lnww[i]);
			}
		}
	}
	
	// Створює нейронну мережу з вхідним шаром 16384 нейронів, 3 прихованими шарами з ReLU
	// та вихідним шаром з 3 нейронів з Softmax
	public static Siec createLetterClassificationNetwork() {
		// Кількість нейронів у кожному шарі
		int[] neuronsInLayers = {1024, 512, 128, 3};
		
		// Функції активації для кожного шару
		Neuron.ActivationFunction[] activations = {
			Neuron.ActivationFunction.RELU,
			Neuron.ActivationFunction.RELU, 
			Neuron.ActivationFunction.RELU,
			Neuron.ActivationFunction.SOFTMAX
		};
		
		// Вхідний шар має 16384 входи (128x128 пікселів)
		return new Siec(16384, neuronsInLayers.length, neuronsInLayers, activations);
	}
	
	double [] oblicz_wyjscie(double [] wejscia){
		double [] wyjscie = null;
		for(int i = 0; i < liczba_warstw; i++)
			wejscia = wyjscie = warstwy[i].oblicz_wyjscie(wejscia);
		return wyjscie;
	}
	
	// Метод для визначення прогнозованого класу (індексу з найбільшим значенням)
	public int predict(double[] input) {
		double[] output = oblicz_wyjscie(input);
		int maxIndex = 0;
		
		for (int i = 1; i < output.length; i++) {
			if (output[i] > output[maxIndex]) {
				maxIndex = i;
			}
		}
		
		return maxIndex;
	}
}
