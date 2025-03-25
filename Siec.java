public class Siec {
	Warstwa [] warstwy;
	int liczba_warstw;
	
	public Siec(){
		warstwy=null;
		this.liczba_warstw=0;
	}
	public Siec(int liczba_wejsc,int liczba_warstw,int [] lnww){
		this.liczba_warstw=liczba_warstw;
		warstwy=new Warstwa[liczba_warstw];
		for(int i=0;i<liczba_warstw;i++)
			warstwy[i]=new Warstwa((i==0)?liczba_wejsc:lnww[i-1],lnww[i]);
	}
	double [] oblicz_wyjscie(double [] wejscia){
		double [] wyjscie=null;
		for(int i=0;i<liczba_warstw;i++)
			wejscia = wyjscie = warstwy[i].oblicz_wyjscie(wejscia);
		return wyjscie;
	}
	
	// Обчислення помилки мережі
	public double obliczBlad(double[] wejscia, double[] oczekiwane) {
		double[] wyjscie = oblicz_wyjscie(wejscia);
		double blad = 0.0;
		
		// Mean Squared Error
		for (int i = 0; i < wyjscie.length; i++) {
			blad += 0.5 * Math.pow(oczekiwane[i] - wyjscie[i], 2);
		}
		
		return blad;
	}
	
	// Метод для навчання мережі методом зворотного поширення
	public void backpropagation(double[] wejscia, double[] oczekiwane, double wspolczynnikNauki) {
		// Прямий прохід із збереженням входів для кожного шару
		double[] aktualneWejscia = wejscia;
		for (int i = 0; i < liczba_warstw; i++) {
			aktualneWejscia = warstwy[i].oblicz_wyjscie(aktualneWejscia);
		}
		
		// Зворотне поширення для останнього шару
		warstwy[liczba_warstw - 1].obliczDeltyWyjsciowe(oczekiwane);
		
		// Зворотне поширення для прихованих шарів
		for (int i = liczba_warstw - 2; i >= 0; i--) {
			warstwy[i].obliczDeltyUkryte(warstwy[i + 1]);
		}
		
		// Оновлення ваг усіх шарів
		for (int i = 0; i < liczba_warstw; i++) {
			warstwy[i].aktualizujWagi(wspolczynnikNauki);
		}
	}
	
	// Метод для тренування на одному прикладі
	public void ucz(double[] wejscia, double[] oczekiwane, double wspolczynnikNauki) {
		backpropagation(wejscia, oczekiwane, wspolczynnikNauki);
	}
	
	// Оновлений метод для тренування на наборі даних
	public void uczZbiorem(double[][] wejscia, double[][] oczekiwane, int liczbaEpok, double wspolczynnikNauki) {
		double bestError = Double.MAX_VALUE;
        int noImprovementCount = 0;
        
        for (int epoka = 0; epoka < liczbaEpok; epoka++) {
            double sredniBlad = 0.0;
            
            // Перемішуємо приклади
            java.util.List<Integer> indices = new java.util.ArrayList<>();
            for (int i = 0; i < wejscia.length; i++) {
                indices.add(i);
            }
            java.util.Collections.shuffle(indices);
            
            // Проходимо по всім прикладам
            for (int idx : indices) {
                ucz(wejscia[idx], oczekiwane[idx], wspolczynnikNauki);
                sredniBlad += obliczBlad(wejscia[idx], oczekiwane[idx]);
            }
            
            sredniBlad /= wejscia.length;
            
            // Виводимо статистику
            if (epoka % 100 == 0) {
                System.out.println("Epoka " + epoka + ": sredni blad = " + sredniBlad);
            }
            
            // Перевірка на покращення
            if (sredniBlad < bestError) {
                bestError = sredniBlad;
                noImprovementCount = 0;
            } else {
                noImprovementCount++;
                if (noImprovementCount > 1000) {
                    System.out.println("Зупинка через відсутність покращення");
                    break;
                }
            }
            
            // Зупинка, якщо помилка достатньо мала
            if (sredniBlad < 0.001) {
                System.out.println("Досягнуто цільової помилки");
                break;
            }
        }
	}
}
