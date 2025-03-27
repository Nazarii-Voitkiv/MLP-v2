import java.io.IOException;
import java.util.Random;
import java.io.File;

public class Main {
    public static void main(String[] args) {
        try {
            System.out.println("Запуск програми класифікації літер...");

            // Шляхи до файлів
            String dataFilePath = "letters_dataset.csv";
            String modelFilePath = "letter_classification_model.model";

            // Перевірка наявності файлу даних
            File dataFile = new File(dataFilePath);
            if (!dataFile.exists()) {
                System.err.println("ПОМИЛКА: Файл даних не знайдено: " + dataFilePath);
                System.err.println("Будь ласка, переконайтеся, що файл розташований у директорії програми.");
                return;
            }

            // Перевірка наявності збереженої моделі
            File modelFile = new File(modelFilePath);
            Siec network = null;

            if (modelFile.exists() && args.length > 0 && args[0].equals("--load-model")) {
                System.out.println("Завантаження існуючої моделі з " + modelFilePath);
                network = Siec.loadModel(modelFilePath);
            } else if (args.length > 0 && args[0].startsWith("--resume=")) {
                // Відновлення навчання з чекпойнта
                String checkpointPath = args[0].substring("--resume=".length());
                System.out.println("Відновлення навчання з чекпойнта: " + checkpointPath);

                try {
                    network = Siec.loadModel(checkpointPath);

                    // Завантажуємо дані для продовження навчання
                    DataProcessor dataProcessor = new DataProcessor();
                    System.out.println("Завантаження даних для продовження навчання...");
                    int samplesPerClass = 3000;
                    dataProcessor.loadBalancedDataFromCSV(dataFilePath, samplesPerClass);

                    double[][] allImages = dataProcessor.getImages();
                    double[][] allLabels = dataProcessor.getLabels();
                    shuffleData(allImages, allLabels);

                    // Розділення на тренувальну, валідаційну та тестову вибірки (70/15/15%)
                    int trainSize = (int)(allImages.length * 0.7);
                    int validSize = (int)(allImages.length * 0.15);
                    int testSize = allImages.length - trainSize - validSize;

                    double[][] trainImages = new double[trainSize][];
                    double[][] trainLabels = new double[trainSize][];
                    double[][] validImages = new double[validSize][];
                    double[][] validLabels = new double[validSize][];
                    double[][] testImages = new double[testSize][];
                    double[][] testLabels = new double[testSize][];

                    // Заповнюємо тренувальні дані
                    for(int i = 0; i < trainSize; i++) {
                        trainImages[i] = allImages[i];
                        trainLabels[i] = allLabels[i];
                    }

                    // Заповнюємо валідаційні дані
                    for(int i = 0; i < validSize; i++) {
                        validImages[i] = allImages[trainSize + i];
                        validLabels[i] = allLabels[trainSize + i];
                    }

                    // Заповнюємо тестові дані
                    for(int i = 0; i < testSize; i++) {
                        testImages[i] = allImages[trainSize + validSize + i];
                        testLabels[i] = allLabels[trainSize + validSize + i];
                    }

                    // Параметри навчання - додаємо визначення змінних
                    int remainingEpochs = 100; 
                    double learningRate = 0.001;
                    double regularizationLambda = 0.0001;
                    double dropoutRate = 0.2;
                    boolean useEarlyStop = true;

                    System.out.println("Продовження навчання на " + remainingEpochs + " епох...");
                    network.train(trainImages, trainLabels, validImages, validLabels,
                            remainingEpochs, learningRate, regularizationLambda, dropoutRate, useEarlyStop);

                    // Зберігаємо фінальну модель
                    network.saveModel(modelFilePath);
                } catch (Exception e) {
                    System.err.println("Помилка відновлення навчання: " + e.getMessage());
                    e.printStackTrace();
                    return;
                }
            } else {
                // Завантаження та підготовка даних
                DataProcessor dataProcessor = new DataProcessor();
                System.out.println("Завантаження даних з " + dataFilePath + "...");

                // Використання оптимальної кількості прикладів
                int samplesPerClass = 3000; // Достатня і збалансована кількість
                System.out.println("Використовуємо збалансоване завантаження - по " + samplesPerClass + " прикладів для кожної літери (M, O, N)");
                dataProcessor.loadBalancedDataFromCSV(dataFilePath, samplesPerClass);

                double[][] allImages = dataProcessor.getImages();
                double[][] allLabels = dataProcessor.getLabels();
                shuffleData(allImages, allLabels);

                // Розділення на тренувальну, валідаційну та тестову вибірки (70/15/15%)
                int trainSize = (int)(allImages.length * 0.7);
                int validSize = (int)(allImages.length * 0.15);
                int testSize = allImages.length - trainSize - validSize;

                double[][] trainImages = new double[trainSize][];
                double[][] trainLabels = new double[trainSize][];
                double[][] validImages = new double[validSize][];
                double[][] validLabels = new double[validSize][];
                double[][] testImages = new double[testSize][];
                double[][] testLabels = new double[testSize][];

                // Заповнюємо тренувальні дані
                for(int i = 0; i < trainSize; i++) {
                    trainImages[i] = allImages[i];
                    trainLabels[i] = allLabels[i];
                }

                // Заповнюємо валідаційні дані
                for(int i = 0; i < validSize; i++) {
                    validImages[i] = allImages[trainSize + i];
                    validLabels[i] = allLabels[trainSize + i];
                }

                // Заповнюємо тестові дані
                for(int i = 0; i < testSize; i++) {
                    testImages[i] = allImages[trainSize + validSize + i];
                    testLabels[i] = allLabels[trainSize + validSize + i];
                }

                System.out.println("Дані розділено: " + trainSize + " тренувальних, " +
                        validSize + " валідаційних і " + testSize + " тестових зразків");

                // Створення та навчання мережі
                System.out.println("\nСтворення нейронної мережі...");
                network = Siec.createLetterClassificationNetwork();

                // Налаштування автозбереження (включено за замовчуванням)
                // network.configureAutoSave(true, 1, "model_checkpoints"); // Це вже встановлено за замовчуванням

                // Параметри навчання
                int epochs = 100;
                double learningRate = 0.001;
                double regularizationLambda = 0.0001;
                double dropoutRate = 0.2;
                boolean useEarlyStop = true;

                System.out.println("\nПочаток навчання мережі з захистом від перенавчання...");
                network.train(trainImages, trainLabels, validImages, validLabels,
                        epochs, learningRate, regularizationLambda, dropoutRate, useEarlyStop);

                // Збереження навченої моделі
                network.saveModel(modelFilePath);
            }

            // Повторне завантаження тестових даних для оцінки
            DataProcessor testDataProcessor = new DataProcessor();
            testDataProcessor.loadDataFromCSV(dataFilePath, 200); // Завантажуємо тільки для тестування

            double[][] testImages = testDataProcessor.getImages();
            double[][] testLabels = testDataProcessor.getLabels();

            // Оцінка моделі
            System.out.println("\nОцінка моделі на тестових даних:");
            double testAccuracy = network.evaluateAccuracy(testImages, testLabels);
            System.out.printf("Точність на тестових даних: %.2f%%\n", testAccuracy * 100);

            // Виведення матриці помилок
            System.out.println("\nМатриця помилок (confusion matrix):");
            int[][] confusionMatrix = network.computeConfusionMatrix(testImages, testLabels, 3);
            printConfusionMatrix(confusionMatrix);

            // Приклад класифікації
            System.out.println("\nПриклад класифікації зображень з тестового набору:");

            for(int i = 0; i < Math.min(5, testImages.length); i++) {
                int predicted = network.predict(testImages[i]);
                String predictedClass = indexToClass(predicted);

                // Визначаємо справжній клас
                int actual = 0;
                for(int j = 1; j < testLabels[i].length; j++) {
                    if(testLabels[i][j] > testLabels[i][actual]) {
                        actual = j;
                    }
                }
                String actualClass = indexToClass(actual);

                System.out.printf("Зразок %d: прогноз = %s, справжній клас = %s\n",
                        i+1, predictedClass, actualClass);
            }

        } catch(IOException e) {
            System.err.println("Помилка при завантаженні даних: " + e.getMessage());
            e.printStackTrace();
        } catch(Exception e) {
            System.err.println("Несподівана помилка: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Метод для перемішування даних
    private static void shuffleData(double[][] images, double[][] labels) {
        Random random = new Random(42); // Встановлюємо seed для відтворюваності

        System.out.println("Перемішування даних...");

        for (int i = images.length - 1; i > 0; i--) {
            int index = random.nextInt(i + 1);

            // Міняємо місцями зображення
            double[] tempImage = images[index];
            images[index] = images[i];
            images[i] = tempImage;

            // Міняємо місцями відповідні мітки
            double[] tempLabel = labels[index];
            labels[index] = labels[i];
            labels[i] = tempLabel;
        }
    }

    // Метод для виведення матриці помилок
    private static void printConfusionMatrix(int[][] matrix) {
        String[] classes = {"M", "O", "N"};

        // Виводимо заголовок
        System.out.print("      ");
        for (String cls : classes) {
            System.out.printf("%-6s", cls);
        }
        System.out.println("\n      ------");

        // Виводимо рядки матриці
        for (int i = 0; i < matrix.length; i++) {
            System.out.printf("%-6s", classes[i]);
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.printf("%-6d", matrix[i][j]);
            }
            System.out.println();
        }
    }

    // Метод для перетворення індексу класу в літеру
    private static String indexToClass(int index) {
        switch(index) {
            case 0: return "M";
            case 1: return "O";
            case 2: return "N";
            default: return "Невідомий";
        }
    }
}
