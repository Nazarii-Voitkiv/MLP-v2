public class Warstwa {
	Neuron [] neurony;
	int liczba_neuronow;
	private double[] ostatnieWejscia; // останні входи шару
	
	public Warstwa(){
		neurony=null;
		liczba_neuronow=0;
	}
	public Warstwa(int liczba_wejsc,int liczba_neuronow){
		this.liczba_neuronow=liczba_neuronow;
		neurony=new Neuron[liczba_neuronow];
		for(int i=0;i<liczba_neuronow;i++)
			neurony[i]=new Neuron(liczba_wejsc);
	}
	double [] oblicz_wyjscie(double [] wejscia){
		// Зберігаємо останні входи
		ostatnieWejscia = new double[wejscia.length];
		System.arraycopy(wejscia, 0, ostatnieWejscia, 0, wejscia.length);
		
		double [] wyjscie=new double[liczba_neuronow];
		for(int i=0;i<liczba_neuronow;i++)
			wyjscie[i]=neurony[i].oblicz_wyjscie(wejscia);
		return wyjscie;
	}
	
	// Обчислення дельт для вихідного шару - виправлена версія
	public void obliczDeltyWyjsciowe(double[] oczekiwane) {
		for (int i = 0; i < liczba_neuronow; i++) {
			double wyjscie = neurony[i].getOstatnieWyjscie();
			 // Використовуємо більш чіткий алгоритм
			double delta = (oczekiwane[i] - wyjscie) * wyjscie * (1.0 - wyjscie);
			neurony[i].setDelta(delta);
		}
	}
	
	// Обчислення дельт для прихованих шарів
	public void obliczDeltyUkryte(Warstwa nastepnaWarstwa) {
		for (int i = 0; i < liczba_neuronow; i++) {
			double suma = 0.0;
			// Сума зважених дельт з наступного шару
			for (int j = 0; j < nastepnaWarstwa.liczba_neuronow; j++) {
				suma += nastepnaWarstwa.neurony[j].getDelta() * nastepnaWarstwa.neurony[j].wagi[i+1];
			}
			
			double wyjscie = neurony[i].getOstatnieWyjscie();
			// Формула для прихованого шару: δ = y * (1 - y) * Σ(δ_j * w_ji)
			double delta = wyjscie * (1.0 - wyjscie) * suma;
			neurony[i].setDelta(delta);
		}
	}
	
	// Оновлення ваг нейронів у шарі
	public void aktualizujWagi(double wspolczynnikNauki) {
		for (int i = 0; i < liczba_neuronow; i++) {
			neurony[i].aktualizujWagi(wspolczynnikNauki);
		}
	}
}
