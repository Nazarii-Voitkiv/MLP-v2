import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        String csvPath = "dataset/emnist-letters-train.csv";
        // Збільшуємо кількість зразків до 1000 для кожної літери
        List<Sample> data = DataLoader.loadMONSamples(csvPath, 1000);
        System.out.println("Завантажено зразків: " + data.size());
        
        // Перемішуємо дані для кращого навчання
        Collections.shuffle(data);

        NeuralNetwork net = new NeuralNetwork(784, 64, 3);
        
        File model = new File("model.dat");
        if (model.exists()) {
            net.loadModel("model.dat");
            System.out.println("Модель завантажено!");
        } else {
            System.out.println("Початок тренування...");
            // Збільшуємо кількість епох до 50
            net.train(data, 50);
            net.saveModel("model.dat");
            System.out.println("Модель збережено!");
        }

        // Визначаємо набір тестових даних (можна взяти останні 10%)
        int testSize = data.size() / 10;
        List<Sample> testData = data.subList(data.size() - testSize, data.size());
        
        // Оцінюємо точність на тестових даних
        int correct = 0;
        for (Sample test : testData) {
            double[] result = net.predict(test.input);
            
            // Визначення найбільшої ймовірності
            int maxIndex = 0;
            for (int i = 1; i < result.length; i++) {
                if (result[i] > result[maxIndex]) maxIndex = i;
            }
            
            // Визначаємо, чи правильно мережа розпізнала літеру
            int targetIndex = 0;
            for (int i = 0; i < test.target.length; i++) {
                if (test.target[i] > 0.5) {
                    targetIndex = i;
                    break;
                }
            }
            
            if (maxIndex == targetIndex) {
                correct++;
            }
        }
        
        System.out.println("Точність на тестових даних: " + 
                           String.format("%.2f%%", 100.0 * correct / testData.size()));
        
        // Тест на одному прикладі
        Sample test = data.get(0);
        double[] result = net.predict(test.input);
        System.out.println("Літера: " + 
            (test.target[0] == 1.0 ? "M" : 
            (test.target[1] == 1.0 ? "N" : "O")));
        System.out.println("Передбачення: " + Arrays.toString(result));
        
        // Визначення найбільшої ймовірності
        int maxIndex = 0;
        for (int i = 1; i < result.length; i++) {
            if (result[i] > result[maxIndex]) maxIndex = i;
        }
        String predictedLetter = (maxIndex == 0) ? "M" : (maxIndex == 1 ? "N" : "O");
        System.out.println("Мережа розпізнала літеру: " + predictedLetter);
    }
}
