public class TestBackprop {
    
    public static void main(String[] args) {
        // Тестування на XOR
        System.out.println("Тестування алгоритму зворотного поширення на прикладі XOR");
        
        // Дані для XOR
        double[][] wejscia = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        
        double[][] oczekiwane = {
            {0},
            {1},
            {1},
            {0}
        };
        
        // Більший прихований шар для кращого навчання
        int[] lnww = {10, 1};  // 10 нейронів у прихованому шарі
        Siec siec = new Siec(2, 2, lnww);
        
        // Виводимо початкові ваги першого нейрона
        System.out.println("Початкові ваги першого нейрона у прихованому шарі:");
        for (int i = 0; i <= 2; i++) {
            System.out.println("w" + i + " = " + siec.warstwy[0].neurony[0].wagi[i]);
        }
        
        // Значно збільшуємо швидкість навчання для XOR
        int liczbaEpok = 20000;
        double wspolczynnikNauki = 0.5;  // Більша швидкість навчання
        
        // Тренування мережі
        System.out.println("\nПочаток навчання...");
        double bestError = Double.MAX_VALUE;
        int noImprovementCount = 0;
        
        for (int epoka = 0; epoka < liczbaEpok; epoka++) {
            double totalError = 0;
            
            // Навчаємо на кожному прикладі окремо
            for (int i = 0; i < wejscia.length; i++) {
                siec.backpropagation(wejscia[i], oczekiwane[i], wspolczynnikNauki);
                totalError += siec.obliczBlad(wejscia[i], oczekiwane[i]);
            }
            
            double avgError = totalError / wejscia.length;
            
            if (epoka % 100 == 0) {
                System.out.println("Epoka " + epoka + ": sredni blad = " + avgError);
            }
            
            // Перевірка на покращення
            if (avgError < bestError) {
                bestError = avgError;
                noImprovementCount = 0;
            } else {
                noImprovementCount++;
                if (noImprovementCount > 2000) {  // Даємо більше часу для навчання
                    System.out.println("Зупинка через відсутність покращення протягом 2000 епох");
                    break;
                }
            }
            
            // Зменшуємо поріг помилки для більш точного навчання
            if (avgError < 0.0001) {
                System.out.println("Досягнуто цільової помилки на епосі " + epoka);
                break;
            }
            
            // Повільно зменшуємо швидкість навчання для кращої збіжності
            if (epoka % 5000 == 0 && epoka > 0) {
                wspolczynnikNauki *= 0.8;
                System.out.println("Змінено швидкість навчання на " + wspolczynnikNauki);
            }
        }
        
        // Виводимо кінцеві ваги першого нейрона у прихованому шарі
        System.out.println("\nКінцеві ваги першого нейрона у прихованому шарі:");
        for (int i = 0; i <= 2; i++) {
            System.out.println("w" + i + " = " + siec.warstwy[0].neurony[0].wagi[i]);
        }
        
        // Тестуємо навчену мережу з більш детальним виводом
        System.out.println("\nРезультати:");
        for (int i = 0; i < wejscia.length; i++) {
            double[] wynik = siec.oblicz_wyjscie(wejscia[i]);
            System.out.println(wejscia[i][0] + " XOR " + wejscia[i][1] + " = " + wynik[0] + 
                             " (очікувано: " + oczekiwane[i][0] + ")");
        }
        
        // Додаємо перевірку точності тестування
        double accuracy = calculateAccuracy(siec, wejscia, oczekiwane);
        System.out.println("\nТочність на тестових даних: " + (accuracy * 100) + "%");
    }
    
    // Перевірка точності - скільки класифіковано правильно
    private static double calculateAccuracy(Siec siec, double[][] inputs, double[][] targets) {
        int correct = 0;
        
        for (int i = 0; i < inputs.length; i++) {
            double[] output = siec.oblicz_wyjscie(inputs[i]);
            // Порівнюємо з порогом 0.5
            boolean predicted = output[0] >= 0.5;
            boolean expected = targets[i][0] >= 0.5;
            
            if (predicted == expected) {
                correct++;
            }
        }
        
        return (double) correct / inputs.length;
    }
}
