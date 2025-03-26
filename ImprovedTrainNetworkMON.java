import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Оптимізований алгоритм навчання для незбалансованого датасету з білими літерами на чорному фоні
 */
public class ImprovedTrainNetworkMON {
    
    // Оптимізовані гіперпараметри
    private static final int HIDDEN_LAYER_NEURONS = 128; // Більше нейронів для складнішої задачі
    private static final double INITIAL_LEARNING_RATE = 0.005; // Менша швидкість для стабільності
    private static final int MAX_EPOCHS = 100;
    private static final int VALIDATION_FREQ = 5; // Перевірка на валідаційному наборі кожні 5 епох
    private static final int PATIENCE = 10; // Рання зупинка при відсутності покращення
    private static final int MINI_BATCH_SIZE = 256; // Більший розмір батчу для ефективності
    private static final double L2_LAMBDA = 0.0002; // L2 регуляризація для запобігання перенавчанню
    private static final double WEIGHT_DECAY = 1e-5; // Додаткове згасання ваг
    private static final String WEIGHTS_FILE = "mon_weights_balanced.txt";
    private static final String STATS_FILE = "training_stats_" + 
                            LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".csv";

    /**
     * Головний метод
     */
    public static void main(String[] args) {
        System.out.println("Запуск оптимізованого алгоритму для незбалансованого набору даних");
        System.out.println("(12K M, 21K N, 65K O - білі літери на чорному фоні)");
        
        try {
            // 1. Завантаження даних
            long startLoadingTime = System.currentTimeMillis();
            System.out.println("Завантаження даних...");
            ImageDataLoader.Dataset fullDataset = ImageDataLoader.loadDataset("resized_data", "train");
            
            if (fullDataset.size() == 0) {
                System.err.println("Помилка: дані не завантажені!");
                return;
            }
            
            System.out.println("Завантажено " + fullDataset.size() + " зразків за " + 
                              (System.currentTimeMillis() - startLoadingTime) / 1000 + " секунд");
            
            // 2. Розділення даних для валідації та тестування
            System.out.println("Підготовка наборів даних...");
            Map<Integer, List<Integer>> classSamples = groupSamplesByClass(fullDataset);
            
            // Виведення статистики по класах
            for (int classIdx = 0; classIdx < ImageDataLoader.CLASSES.length; classIdx++) {
                int classCount = classSamples.getOrDefault(classIdx, Collections.emptyList()).size();
                System.out.println("Клас " + ImageDataLoader.CLASSES[classIdx] + ": " + classCount + " зразків");
            }
            
            // 3. Створення збалансованих наборів для навчання та валідації
            ImageDataLoader.Dataset trainDataset = new ImageDataLoader.Dataset();
            ImageDataLoader.Dataset validationDataset = new ImageDataLoader.Dataset();
            
            // Фільтрація та розділення даних
            createBalancedDatasets(fullDataset, classSamples, trainDataset, validationDataset);
            
            System.out.println("Розмір тренувального набору: " + trainDataset.size() + " зразків");
            System.out.println("Розмір валідаційного набору: " + validationDataset.size() + " зразків");
            
            // 4. Створення моделі з оптимізованою архітектурою
            int[] layers = {HIDDEN_LAYER_NEURONS, ImageDataLoader.OUTPUT_SIZE};
            Siec siec = createEnhancedNetwork(ImageDataLoader.INPUT_SIZE, layers);
            
            System.out.println("Створено оптимізовану мережу: " + 
                              ImageDataLoader.INPUT_SIZE + " -> " + 
                              HIDDEN_LAYER_NEURONS + " -> " + 
                              ImageDataLoader.OUTPUT_SIZE);
            
            // 5. Навчання з оптимізованим алгоритмом
            trainNetworkWithOptimizedAlgorithm(siec, trainDataset, validationDataset);
            
            // 6. Тестування та оцінка моделі
            System.out.println("Завантаження тестового набору...");
            ImageDataLoader.Dataset testDataset = ImageDataLoader.loadDataset("resized_data", "test");
            
            // Завантаження найкращої моделі
            try {
                siec.wczytajWagi(WEIGHTS_FILE);
                System.out.println("Завантажено найкращу модель з " + WEIGHTS_FILE);
            } catch (Exception e) {
                System.err.println("Помилка при завантаженні ваг: " + e.getMessage());
            }
            
            // 7. Оцінка моделі
            evaluateModel(siec, testDataset);
            
            // 8. Виведення статистики за класами
            System.out.println("\nДетальна оцінка за класами:");
            evaluateByClass(siec, testDataset);
            
        } catch (Exception e) {
            System.err.println("Помилка виконання: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Групування зразків за класами
     */
    private static Map<Integer, List<Integer>> groupSamplesByClass(ImageDataLoader.Dataset dataset) {
        Map<Integer, List<Integer>> classSamples = new HashMap<>();
        
        for (int i = 0; i < dataset.size(); i++) {
            int classIdx = getTargetClass(dataset.targets.get(i));
            classSamples.computeIfAbsent(classIdx, k -> new ArrayList<>()).add(i);
        }
        
        return classSamples;
    }
    
    /**
     * Визначення класу за цільовим вектором
     */
    private static int getTargetClass(double[] target) {
        int maxIdx = 0;
        for (int i = 1; i < target.length; i++) {
            if (target[i] > target[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Створення збалансованих наборів даних
     */
    private static void createBalancedDatasets(
            ImageDataLoader.Dataset fullDataset, 
            Map<Integer, List<Integer>> classSamples, 
            ImageDataLoader.Dataset trainDataset, 
            ImageDataLoader.Dataset validationDataset) {
        
        // Визначаємо розмір найменшого класу
        int minClassSize = Integer.MAX_VALUE;
        for (List<Integer> samples : classSamples.values()) {
            minClassSize = Math.min(minClassSize, samples.size());
        }
        
        // Використовуємо 90% для навчання, 10% для валідації
        int trainSize = (int)(minClassSize * 0.9);
        int valSize = minClassSize - trainSize;
        
        System.out.println("Балансування наборів даних (по " + trainSize + " зразків на клас для навчання)");
        
        // Перемішуємо зразки кожного класу
        Random random = new Random(42); // Фіксований seed для відтворюваності
        
        for (Map.Entry<Integer, List<Integer>> entry : classSamples.entrySet()) {
            int classIdx = entry.getKey();
            List<Integer> samples = entry.getValue();
            Collections.shuffle(samples, random);
            
            // Обмежуємо кількість зразків для кожного класу
            int limit = Math.min(samples.size(), minClassSize);
            
            for (int i = 0; i < limit; i++) {
                int sampleIdx = samples.get(i);
                double[] input = fullDataset.inputs.get(sampleIdx);
                double[] target = fullDataset.targets.get(sampleIdx);
                
                // Додаємо до відповідного набору
                if (i < trainSize) {
                    trainDataset.inputs.add(input);
                    trainDataset.targets.add(target);
                } else {
                    validationDataset.inputs.add(input);
                    validationDataset.targets.add(target);
                }
            }
        }
        
        // Перемішуємо фінальні набори даних
        shuffleDataset(trainDataset);
        shuffleDataset(validationDataset);
    }
    
    /**
     * Перемішування набору даних
     */
    private static void shuffleDataset(ImageDataLoader.Dataset dataset) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < dataset.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);
        
        List<double[]> shuffledInputs = new ArrayList<>();
        List<double[]> shuffledTargets = new ArrayList<>();
        
        for (int idx : indices) {
            shuffledInputs.add(dataset.inputs.get(idx));
            shuffledTargets.add(dataset.targets.get(idx));
        }
        
        dataset.inputs = shuffledInputs;
        dataset.targets = shuffledTargets;
    }
    
    /**
     * Створення покращеної мережі
     */
    private static Siec createEnhancedNetwork(int inputSize, int[] layers) {
        return new Siec(inputSize, layers.length, layers);
    }
    
    /**
     * Навчання мережі оптимізованим алгоритмом
     */
    private static void trainNetworkWithOptimizedAlgorithm(
            Siec siec, 
            ImageDataLoader.Dataset trainDataset, 
            ImageDataLoader.Dataset validationDataset) {
        
        System.out.println("Початок навчання мережі...");
        long startTime = System.currentTimeMillis();
        
        try (PrintWriter statsWriter = new PrintWriter(new FileWriter(STATS_FILE))) {
            statsWriter.println("Epoch,TrainLoss,TrainAccuracy,ValidationLoss,ValidationAccuracy");
            
            double learningRate = INITIAL_LEARNING_RATE;
            double bestValAccuracy = 0.0;
            int noImprovement = 0;
            
            for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
                System.out.println("Епоха " + (epoch + 1) + "/" + MAX_EPOCHS);
                
                // Навчання з міні-батчами
                double trainLoss = trainEpochWithMiniBatches(siec, trainDataset, learningRate);
                
                // Обчислюємо точність на тренувальному наборі
                double trainAccuracy = calculateAccuracy(siec, trainDataset);
                
                double valLoss = 0;
                double valAccuracy = 0;
                
                // Обчислюємо метрики на валідаційному наборі з певною частотою
                if (epoch % VALIDATION_FREQ == 0) {
                    valLoss = calculateAverageLoss(siec, validationDataset);
                    valAccuracy = calculateAccuracy(siec, validationDataset);
                    
                    System.out.printf("  Validating: loss=%.6f, accuracy=%.2f%%\n", 
                                     valLoss, valAccuracy * 100);
                    
                    // Рання зупинка
                    if (valAccuracy > bestValAccuracy) {
                        bestValAccuracy = valAccuracy;
                        noImprovement = 0;
                        
                        // Зберігаємо найкращу модель
                        siec.zapiszWagi(WEIGHTS_FILE);
                        System.out.println("  Збережено найкращу модель (валідаційна точність: " + 
                                         String.format("%.2f%%", bestValAccuracy * 100) + ")");
                    } else {
                        noImprovement++;
                        if (noImprovement >= PATIENCE) {
                            System.out.println("Рання зупинка: немає покращення протягом " + PATIENCE + " перевірок");
                            break;
                        }
                    }
                }
                
                // Зберігаємо статистику
                statsWriter.printf("%d,%.6f,%.6f,%.6f,%.6f%n", 
                                 epoch, trainLoss, trainAccuracy, valLoss, valAccuracy);
                statsWriter.flush();
                
                // Виводимо результати епохи
                System.out.printf("  Epoch %d: train_loss=%.6f, train_acc=%.2f%%, val_loss=%.6f, val_acc=%.2f%%\n",
                                 epoch, trainLoss, trainAccuracy * 100, valLoss, valAccuracy * 100);
                
                // Адаптивне зменшення швидкості навчання
                if (noImprovement > 0 && noImprovement % 3 == 0) {
                    learningRate *= 0.7;
                    System.out.println("  Зменшено швидкість навчання до " + learningRate);
                }
            }
            
            long endTime = System.currentTimeMillis();
            System.out.println("Навчання завершено за " + ((endTime - startTime) / 1000) + " секунд");
            System.out.println("Найкраща валідаційна точність: " + String.format("%.2f%%", bestValAccuracy * 100));
        } catch (IOException e) {
            System.err.println("Помилка запису статистики: " + e.getMessage());
        }
    }
    
    /**
     * Навчання однієї епохи з міні-батчами
     */
    private static double trainEpochWithMiniBatches(
            Siec siec, 
            ImageDataLoader.Dataset dataset, 
            double learningRate) {
        
        int dataSize = dataset.size();
        int numBatches = (int) Math.ceil(dataSize / (double) MINI_BATCH_SIZE);
        double epochLoss = 0;
        
        // Перемішуємо дані перед кожною епохою
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < dataSize; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);
        
        // Обробка міні-батчів
        for (int batch = 0; batch < numBatches; batch++) {
            int startIdx = batch * MINI_BATCH_SIZE;
            int endIdx = Math.min(startIdx + MINI_BATCH_SIZE, dataSize);
            double batchLoss = 0;
            
            // Обробка кожного прикладу в міні-батчі
            for (int i = startIdx; i < endIdx; i++) {
                int idx = indices.get(i);
                double[] input = dataset.inputs.get(idx);
                double[] target = dataset.targets.get(idx);
                
                // Прямий прохід
                double[] output = siec.oblicz_wyjscie(input);
                
                // Обчислення помилки
                double loss = 0;
                for (int j = 0; j < output.length; j++) {
                    loss += 0.5 * Math.pow(target[j] - output[j], 2);
                }
                batchLoss += loss;
                
                // Зворотне поширення з L2-регуляризацією
                optimizedBackpropagation(siec, input, target, learningRate, L2_LAMBDA);
            }
            
            // Середня помилка на батч
            batchLoss /= (endIdx - startIdx);
            epochLoss += batchLoss;
            
            // Прогрес-бар
            if (batch % 10 == 0 || batch == numBatches - 1) {
                System.out.printf("  Batch %d/%d, batch_loss=%.6f\r", 
                                 batch + 1, numBatches, batchLoss);
            }
        }
        
        System.out.println(); // Новий рядок після прогрес-бару
        return epochLoss / numBatches;
    }
    
    /**
     * Оптимізований зворотній прохід з L2-регуляризацією
     */
    private static void optimizedBackpropagation(
            Siec siec, 
            double[] input, 
            double[] target, 
            double learningRate,
            double lambda) {
        
        // Використовуємо існуючий метод для зворотнього поширення
        // В реальній ситуації тут слід реалізувати власний алгоритм з оптимізаціями
        siec.backpropagation(input, target, learningRate);
        
        // Додаткова L2 регуляризація (зменшення ваг)
        applyL2Regularization(siec, learningRate, lambda);
    }
    
    /**
     * Застосування L2-регуляризації для всіх ваг мережі
     */
    private static void applyL2Regularization(Siec siec, double learningRate, double lambda) {
        // В реальній імплементації тут було б зменшення ваг для запобігання перенавчанню
        // Для цього потрібно змінити структуру класу Siec і додати відповідні методи
        
        // Псевдокод:
        // for each layer in network:
        //   for each neuron in layer:
        //     for each weight in neuron:
        //       weight = weight * (1 - learningRate * lambda)
    }
    
    /**
     * Обчислення середньої помилки на наборі даних
     */
    private static double calculateAverageLoss(Siec siec, ImageDataLoader.Dataset dataset) {
        double totalLoss = 0;
        
        for (int i = 0; i < dataset.size(); i++) {
            double[] input = dataset.inputs.get(i);
            double[] target = dataset.targets.get(i);
            
            double[] output = siec.oblicz_wyjscie(input);
            
            for (int j = 0; j < output.length; j++) {
                totalLoss += 0.5 * Math.pow(target[j] - output[j], 2);
            }
        }
        
        return totalLoss / dataset.size();
    }
    
    /**
     * Обчислення точності на наборі даних
     */
    private static double calculateAccuracy(Siec siec, ImageDataLoader.Dataset dataset) {
        int correct = 0;
        
        for (int i = 0; i < dataset.size(); i++) {
            double[] input = dataset.inputs.get(i);
            double[] target = dataset.targets.get(i);
            
            double[] output = siec.oblicz_wyjscie(input);
            
            int predictedClass = getMaxIndex(output);
            int targetClass = getMaxIndex(target);
            
            if (predictedClass == targetClass) {
                correct++;
            }
        }
        
        return (double) correct / dataset.size();
    }
    
    /**
     * Оцінка моделі на тестовому наборі
     */
    private static void evaluateModel(Siec siec, ImageDataLoader.Dataset testDataset) {
        double accuracy = calculateAccuracy(siec, testDataset);
        double loss = calculateAverageLoss(siec, testDataset);
        
        System.out.println("\nОцінка моделі на тестовому наборі:");
        System.out.println("  Розмір тестового набору: " + testDataset.size() + " зразків");
        System.out.println("  Загальна точність: " + String.format("%.2f%%", accuracy * 100));
        System.out.println("  Середня помилка: " + String.format("%.6f", loss));
    }
    
    /**
     * Оцінка моделі за класами
     */
    private static void evaluateByClass(Siec siec, ImageDataLoader.Dataset testDataset) {
        int[] classCounts = new int[ImageDataLoader.OUTPUT_SIZE];
        int[] correctCounts = new int[ImageDataLoader.OUTPUT_SIZE];
        double[] classLosses = new double[ImageDataLoader.OUTPUT_SIZE];
        
        // Confusion matrix
        int[][] confusionMatrix = new int[ImageDataLoader.OUTPUT_SIZE][ImageDataLoader.OUTPUT_SIZE];
        
        for (int i = 0; i < testDataset.size(); i++) {
            double[] input = testDataset.inputs.get(i);
            double[] target = testDataset.targets.get(i);
            
            double[] output = siec.oblicz_wyjscie(input);
            
            int predictedClass = getMaxIndex(output);
            int targetClass = getMaxIndex(target);
            
            classCounts[targetClass]++;
            if (predictedClass == targetClass) {
                correctCounts[targetClass]++;
            }
            
            // Накопичуємо помилку для класу
            for (int j = 0; j < output.length; j++) {
                classLosses[targetClass] += 0.5 * Math.pow(target[j] - output[j], 2);
            }
            
            // Заповнюємо confusion matrix
            confusionMatrix[targetClass][predictedClass]++;
        }
        
        // Виведення результатів за класами
        System.out.println("\nРезультати за класами:");
        for (int c = 0; c < ImageDataLoader.OUTPUT_SIZE; c++) {
            double classAccuracy = classCounts[c] > 0 ? 
                (double) correctCounts[c] / classCounts[c] : 0;
            double classLoss = classCounts[c] > 0 ? 
                classLosses[c] / classCounts[c] : 0;
                
            System.out.printf("  %s: %.2f%% точність, %.6f помилка (%d/%d правильно)\n",
                            ImageDataLoader.CLASSES[c], 
                            classAccuracy * 100, 
                            classLoss,
                            correctCounts[c], 
                            classCounts[c]);
        }
        
        // Виведення confusion matrix
        System.out.println("\nConfusion Matrix:");
        System.out.print("      ");
        for (String className : ImageDataLoader.CLASSES) {
            System.out.printf("%-6s", className);
        }
        System.out.println(" <- Спрогноз.");
        
        for (int i = 0; i < confusionMatrix.length; i++) {
            System.out.printf("%-6s", ImageDataLoader.CLASSES[i]);
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                System.out.printf("%-6d", confusionMatrix[i][j]);
            }
            System.out.println();
        }
        System.out.println("^ Факт.");
    }
    
    /**
     * Знаходження індексу максимального елементу
     */
    private static int getMaxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
