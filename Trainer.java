import java.io.File;
import java.io.IOException;
import java.util.List;

public class Trainer {
    
    private static final String MODEL_PATH = "model.dat";
    
    public static void main(String[] args) {
        System.out.println("Запуск тренування нейромережі для розпізнавання літер M, O, N");
        
        // 1. Завантаження зразків з MyDataLoader
        System.out.println("Завантаження зразків...");
        List<Sample> samples = MyDataLoader.loadSamples();
        
        // 2. Перевірка наявності зразків
        if (samples.isEmpty()) {
            System.err.println("Помилка: немає зразків для тренування. Перевірте папку data/");
            return;
        }
        
        // 3. Створення нової нейромережі
        System.out.println("Створення нейромережі...");
        NeuralNetwork net = new NeuralNetwork();
        
        // 4-5. Перевірка наявності збереженої моделі або тренування нової
        File modelFile = new File(MODEL_PATH);
        
        if (modelFile.exists()) {
            System.out.println("Знайдено збережену модель. Завантаження...");
            try {
                net.loadModel(MODEL_PATH);
            } catch (IOException | ClassNotFoundException e) {
                System.err.println("Помилка при завантаженні моделі: " + e.getMessage());
                System.out.println("Буде натреновано нову модель.");
                trainNewModel(net, samples);
            }
        } else {
            System.out.println("Збережену модель не знайдено. Початок навчання нової моделі...");
            trainNewModel(net, samples);
        }
        
        // 6. Обчислення точності на всій вибірці
        calculateAndPrintAccuracy(net, samples);
    }
    
    private static void trainNewModel(NeuralNetwork net, List<Sample> samples) {
        // Налаштування параметрів
        net.setDropoutRate(0.1);
        net.setPatience(10);
        net.setValidationSplit(0.2);
        
        // Тренування мережі (50 епох)
        System.out.println("Тренування нейромережі (50 епох)...");
        net.train(samples, 200);
        
        // Збереження моделі
        try {
            System.out.println("Збереження моделі...");
            net.saveModel(MODEL_PATH);
        } catch (IOException e) {
            System.err.println("Помилка при збереженні моделі: " + e.getMessage());
        }
    }
    
    private static void calculateAndPrintAccuracy(NeuralNetwork net, List<Sample> samples) {
        System.out.println("\nОцінка точності на всій вибірці:");
        
        int correctPredictions = 0;
        int totalSamples = samples.size();
        
        for (Sample sample : samples) {
            // Отримуємо прогноз
            double[] prediction = net.predict(sample.getInput());
            
            // Знаходимо індекс найбільшого значення (передбачений клас)
            int predictedIndex = findMaxIndex(prediction);
            
            // Знаходимо індекс найбільшого значення у цілі (правильний клас)
            int targetIndex = findMaxIndex(sample.getTarget());
            
            // Порівнюємо та рахуємо правильні передбачення
            if (predictedIndex == targetIndex) {
                correctPredictions++;
            }
        }
        
        // Обчислюємо та виводимо точність
        double accuracy = (double) correctPredictions / totalSamples * 100;
        System.out.printf("Точність: %.2f%% (%d/%d правильних прогнозів)%n", 
                         accuracy, correctPredictions, totalSamples);
        
        // Виводимо інформацію про класи
        System.out.println("Інформація про класи: 0 = M, 1 = O, 2 = N");
    }
    
    /**
     * Знаходить індекс елемента з максимальним значенням у масиві
     */
    private static int findMaxIndex(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];
        
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
}
