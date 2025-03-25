import java.util.Random;

public class Neuron {
	double [] wagi;
	int liczba_wejsc;
	private double ostatnieWyjscie; // останній вихід нейрона
	private double delta; // дельта для алгоритму зворотного поширення
	private double[] ostatnieWejscia; // останні входи для оновлення ваг

	public Neuron(){
		liczba_wejsc=0;
		wagi=null;
	}
	public Neuron(int liczba_wejsc){
		this.liczba_wejsc=liczba_wejsc;
		wagi=new double[liczba_wejsc+1];
		generuj();
	}
	private void generuj() {
		Random r=new Random();
		// Значно більша ініціалізація для уникнення плато при навчанні XOR
		for(int i=0;i<=liczba_wejsc;i++) {
			wagi[i] = (r.nextDouble()-0.5)*2.0; // Діапазон від -1 до 1
		}
	}
	public double oblicz_wyjscie(double [] wejscia){
		double fi=wagi[0];
		for(int i=1;i<=liczba_wejsc;i++)
			fi+=wagi[i]*wejscia[i-1];
		ostatnieWyjscie = 1.0/(1.0+Math.exp(-fi)); // функja aktywacji sigma -unip
		
		// Зберігаємо вхідні значення для оновлення ваг
		ostatnieWejscia = new double[wejscia.length];
		System.arraycopy(wejscia, 0, ostatnieWejscia, 0, wejscia.length);
		
		return ostatnieWyjscie;
	}
	
	// Отримання останнього значення виходу
	public double getOstatnieWyjscie() {
		return ostatnieWyjscie;
	}
	
	// Встановлення та отримання дельти
	public void setDelta(double delta) {
		this.delta = delta;
	}
	
	public double getDelta() {
		return delta;
	}
	
	// Виправлення методу оновлення ваг
	public void aktualizujWagi(double wspolczynnikNauki) {
		// Оновлення bias
		wagi[0] += wspolczynnikNauki * delta;
		
		// Оновлення інших ваг
		for (int i = 1; i <= liczba_wejsc; i++) {
			if (i-1 < ostatnieWejscia.length) {
				wagi[i] += wspolczynnikNauki * delta * ostatnieWejscia[i-1];
			}
		}
	}

    public void zapiszWagi(java.io.PrintWriter writer) {
        for (int i = 0; i <= liczba_wejsc; i++) {
            writer.println(wagi[i]);
        }
    }
    
    public void wczytajWagi(java.io.BufferedReader reader) throws java.io.IOException {
        for (int i = 0; i <= liczba_wejsc; i++) {
            wagi[i] = Double.parseDouble(reader.readLine());
        }
    }
}
