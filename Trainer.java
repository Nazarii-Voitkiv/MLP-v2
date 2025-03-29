import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

public class Trainer {
    
    private static final String MODEL_PATH = "model.dat";
    
    public static void main(String[] args) {
        System.out.println("Запуск тренування нейромережі для розпізнавання літер M, O, N");

        System.out.println("Завантаження зразків...");
        List<Sample> samples = MyDataLoader.loadSamples();

        if (samples.isEmpty()) {
            System.err.println("Помилка: немає зразків для тренування. Перевірте папку data/");
            return;
        }

        System.out.println("Створення нейромережі...");
        NeuralNetwork net = new NeuralNetwork();

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

        calculateAndPrintAccuracy(net, samples);
    }
    
    private static void trainNewModel(NeuralNetwork net, List<Sample> samples) {
        net.setDropoutRate(0.0);
        net.setPatience(10);
        net.setValidationSplit(0.2);

        System.out.println("Тренування нейромережі (50 епох)...");
        net.train(samples, 200);

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

        int[][] confusionMatrix = new int[3][3];

        String[] classNames = {"M", "O", "N"};

        int[] classCount = new int[3];
        
        for (Sample sample : samples) {
            double[] prediction = net.predict(sample.getInput());

            int predictedIndex = findMaxIndex(prediction);

            int targetIndex = findMaxIndex(sample.getTarget());

            classCount[targetIndex]++;

            confusionMatrix[targetIndex][predictedIndex]++;

            if (predictedIndex == targetIndex) {
                correctPredictions++;
            }
        }

        double accuracy = (double) correctPredictions / totalSamples * 100;
        System.out.printf("Точність: %.2f%% (%d/%d правильних прогнозів)%n", 
                         accuracy, correctPredictions, totalSamples);

        System.out.println("Інформація про класи: 0 = M, 1 = O, 2 = N");

        System.out.println("\nТочність по класах:");
        for (int i = 0; i < 3; i++) {
            double classAccuracy = (double) confusionMatrix[i][i] / classCount[i] * 100;
            System.out.printf("%s: %.2f%% (%d/%d)%n", 
                            classNames[i], classAccuracy, confusionMatrix[i][i], classCount[i]);
        }

        System.out.println("\nМатриця плутанини:");
        System.out.println("       | Predicted |");
        System.out.println("       |  M  |  O  |  N  |");
        System.out.println("-------|-----|-----|-----|");
        
        for (int i = 0; i < 3; i++) {
            System.out.printf("Actual %s | %3d | %3d | %3d |%n", 
                            classNames[i], confusionMatrix[i][0], confusionMatrix[i][1], confusionMatrix[i][2]);
        }

        System.out.println("\nНайбільш плутані пари класів:");
        Map<String, Integer> confusionPairs = new HashMap<>();
        
        for (int actual = 0; actual < 3; actual++) {
            for (int predicted = 0; predicted < 3; predicted++) {
                if (actual != predicted && confusionMatrix[actual][predicted] > 0) {
                    String pair = classNames[actual] + " → " + classNames[predicted];
                    confusionPairs.put(pair, confusionMatrix[actual][predicted]);
                }
            }
        }

        confusionPairs.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .forEach(entry -> System.out.printf("%s: %d випадків%n", entry.getKey(), entry.getValue()));

        System.out.println("\nРекомендації щодо аугментації даних:");
        int[] misclassified = new int[3];
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i != j) {
                    misclassified[i] += confusionMatrix[i][j];
                }
            }
        }
        
        for (int i = 0; i < 3; i++) {
            if (misclassified[i] > 0) {
                System.out.printf("- Літера %s: додати більше прикладів та варіацій (неправильно класифіковано: %d)%n", 
                                classNames[i], misclassified[i]);
            }
        }
    }
    
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
