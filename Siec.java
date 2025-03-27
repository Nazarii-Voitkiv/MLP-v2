import java.io.*;
import java.util.Random;

public class Siec implements Serializable {
    // Додаємо serialVersionUID для серіалізації
    private static final long serialVersionUID = 1L;
    
    Warstwa [] warstwy;
    int liczba_warstw;
    
    // Параметри навчання
    private double learningRate = 0.01;
    private int maxEpochs = 10;
    
    // Параметри для запобігання перенавчання
    private double regularizationLambda = 0.0001; // L2 регуляризація
    private double dropoutRate = 0.2; // Dropout rate
    private boolean useDropout = false; // Чи використовувати dropout
    private Random random = new Random(42); // Для відтворюваності
    private double bestValidationError = Double.MAX_VALUE; // Для раннього припинення
    private int patienceCounter = 0; // Лічильник patience для раннього припинення
    private int patience = 5; // Збільшуємо patience з 3 до 5 епох
    private Warstwa[] bestWeights; // Зберігати найкращі ваги для раннього припинення
    
    // Додаємо параметри для налаштування автозбереження
    private boolean autoSaveEnabled = true;  // Автозбереження включено за замовчуванням
    private int autoSaveFrequency = 1;       // Частота збереження (кожну епоху за замовчуванням)
    private String autoSaveDirectory = "model_checkpoints";
    
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
    
    // Метод для навчання мережі з розширеними опціями для запобігання перенавчання
    public void train(double[][] trainInputs, double[][] trainTargets, 
                      double[][] validInputs, double[][] validTargets,
                      int epochs, double learningRate, 
                      double lambda, double dropout, boolean earlyStop) {
        this.learningRate = Math.min(learningRate, 0.005);
        this.maxEpochs = epochs;
        this.regularizationLambda = lambda;
        this.dropoutRate = dropout;
        this.useDropout = dropout > 0;
        
        // Налаштування автозбереження
        File autoSaveFolder = new File(autoSaveDirectory);
        if (autoSaveEnabled) {
            if (!autoSaveFolder.exists() && !autoSaveFolder.mkdirs()) {
                System.err.println("ПОПЕРЕДЖЕННЯ: Не вдалося створити директорію для автозбереження: " + autoSaveDirectory);
                System.err.println("Автозбереження буде відключено.");
                autoSaveEnabled = false;
            } else {
                System.out.println("АВТОЗБЕРЕЖЕННЯ: Увімкнено. Моделі будуть зберігатися кожні " + 
                                  autoSaveFrequency + " епох(и) у директорію '" + autoSaveDirectory + "'");
            }
        }
        
        // Створюємо директорію для збереження моделей
        String checkpointDir = "model_checkpoints";
        boolean saveCheckpoints = true;
        File checkpointFolder = new File(checkpointDir);
        if (!checkpointFolder.exists() && saveCheckpoints) {
            if (!checkpointFolder.mkdir()) {
                System.err.println("Не вдалося створити директорію для чекпойнтів: " + checkpointDir);
                saveCheckpoints = false;
            }
        }
        
        System.out.println("--------------------------------------------------");
        System.out.println("ПОЧАТОК НАВЧАННЯ НЕЙРОННОЇ МЕРЕЖІ");
        System.out.println("--------------------------------------------------");
        System.out.println("Параметри навчання:");
        System.out.println("- Кількість епох: " + epochs);
        System.out.println("- Швидкість навчання: " + learningRate);
        System.out.println("- L2 регуляризація (lambda): " + lambda);
        System.out.println("- Dropout rate: " + dropout);
        System.out.println("- Early stopping: " + (earlyStop ? "включено" : "виключено"));
        System.out.println("- Розмір навчальної вибірки: " + trainInputs.length + " зразків");
        if (validInputs != null) {
            System.out.println("- Розмір валідаційної вибірки: " + validInputs.length + " зразків");
        }
        System.out.println("--------------------------------------------------");
        
        long startTime = System.currentTimeMillis();
        double bestTrainError = Double.MAX_VALUE;
        
        // Зберігаємо копії ваг для раннього припинення
        if (earlyStop && validInputs != null && validTargets != null) {
            saveBestWeights();
        }
        
        // Навчання на епохах
        for(int epoch = 0; epoch < maxEpochs; epoch++) {
            long epochStartTime = System.currentTimeMillis();
            double totalTrainError = 0.0;
            boolean hasNaN = false;
            
            // Перемішуємо тренувальні дані
            int[] indices = shuffleIndices(trainInputs.length);
            
            System.out.println("\nЕпоха " + (epoch + 1) + "/" + maxEpochs);
            System.out.println("Обробка тренувальних зразків...");
            
            // Проходимо всі приклади
            int batchSize = 100; // Виводимо прогрес кожні batchSize зразків
            for(int i = 0; i < trainInputs.length; i++) {
                int idx = indices[i]; // Випадковий індекс
                
                // Forward pass з dropout (якщо активований)
                double[] output = forwardPassWithDropout(trainInputs[idx]);
                
                // Перевірка на NaN у виході
                if (containsNaN(output)) {
                    System.err.println("УВАГА: NaN значення виявлено у виході мережі. Пропускаємо оновлення ваг.");
                    hasNaN = true;
                    continue; // Пропускаємо приклад з NaN
                }
                
                // Обчислюємо помилку (з регуляризацією)
                double error = calculateErrorWithRegularization(output, trainTargets[idx]);
                totalTrainError += error;
                
                // Backpropagation
                backpropagateWithRegularizationAndClipping(trainTargets[idx]);
                
                // Виводимо прогрес
                if ((i + 1) % batchSize == 0 || i == trainInputs.length - 1) {
                    System.out.printf("  Оброблено %d/%d зразків (%.1f%%)\r", 
                                     i + 1, trainInputs.length, (i + 1) * 100.0 / trainInputs.length);
                }
            }
            System.out.println(); // Новий рядок після завершення поточної епохи
            
            double avgTrainError = totalTrainError / trainInputs.length;
            if (avgTrainError < bestTrainError) {
                bestTrainError = avgTrainError;
            }
            
            long epochTime = System.currentTimeMillis() - epochStartTime;
            
            System.out.printf("Тренувальна помилка: %.6f", avgTrainError);
            
            // Перевіряємо на валідаційному наборі для раннього припинення
            if (earlyStop && validInputs != null && validTargets != null) {
                System.out.print(" | Перевірка на валідаційній вибірці...");
                double validationError = evaluateError(validInputs, validTargets);
                
                // Обчислюємо точність на валідаційному наборі
                double validAccuracy = evaluateAccuracy(validInputs, validTargets);
                System.out.printf(" помилка: %.6f, точність: %.2f%%", validationError, validAccuracy * 100);
                
                // Перевіряємо чи слід припинити навчання
                if (checkEarlyStop(validationError)) {
                    System.out.println("\nВиявлено переннавчання! Раннє припинення на епосі " + (epoch + 1));
                    System.out.println("Відновлення найкращих ваг з найнижчою валідаційною помилкою.");
                    restoreBestWeights();
                    break;
                }
            }
            
            System.out.printf(" | Час епохи: %d мс\n", epochTime);
            
            // Автозбереження моделі відповідно до налаштованої частоти
            if (autoSaveEnabled && (epoch + 1) % autoSaveFrequency == 0) {
                try {
                    // Формуємо ім'я файлу з номером епохи та значенням помилки
                    String checkpointPath = String.format("%s/epoch_%03d_error_%.4f.model", 
                                                       autoSaveDirectory, epoch + 1, avgTrainError);
                    saveModel(checkpointPath);
                    
                    // Відображаємо інформацію про збереження
                    System.out.println("АВТОЗБЕРЕЖЕННЯ: Модель збережено після епохи " + (epoch + 1) + ": " + checkpointPath);
                } catch (IOException e) {
                    System.err.println("ПОМИЛКА АВТОЗБЕРЕЖЕННЯ на епосі " + (epoch + 1) + ": " + e.getMessage());
                }
            }
            
            // Виводимо попередження, якщо виявлено NaN
            if (hasNaN) {
                System.err.println("ПОПЕРЕДЖЕННЯ: Виявлено NaN в цій епосі. Розгляньте зменшення швидкості навчання.");
            }
            
            // Автоматичне зменшення швидкості навчання, якщо помилка зростає
            if (epoch > 0 && avgTrainError > bestTrainError * 1.1) {
                learningRate *= 0.5;
                System.out.println("Зменшуємо швидкість навчання до " + learningRate);
            }
        }
        
        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("--------------------------------------------------");
        System.out.println("НАВЧАННЯ ЗАВЕРШЕНО");
        System.out.println("--------------------------------------------------");
        System.out.println("Загальний час навчання: " + (totalTime / 1000) + " сек.");
        System.out.println("Найкраща тренувальна помилка: " + bestTrainError);
        if (earlyStop && validInputs != null) {
            System.out.println("Найкраща валідаційна помилка: " + bestValidationError);
        }
        System.out.println("--------------------------------------------------");
    }
    
    // Стандартний метод для навчання (для зворотньої сумісності)
    public void train(double[][] inputs, double[][] targets, int epochs, double learningRate) {
        // Використовуємо 80% для тренування, 20% для валідації
        int trainSize = (int)(inputs.length * 0.8);
        int validSize = inputs.length - trainSize;
        
        double[][] trainInputs = new double[trainSize][];
        double[][] trainTargets = new double[trainSize][];
        double[][] validInputs = new double[validSize][];
        double[][] validTargets = new double[validSize][];
        
        // Розділяємо дані (припускаємо, що вони вже перемішані)
        for(int i = 0; i < trainSize; i++) {
            trainInputs[i] = inputs[i];
            trainTargets[i] = targets[i];
        }
        
        for(int i = 0; i < validSize; i++) {
            validInputs[i] = inputs[trainSize + i];
            validTargets[i] = targets[trainSize + i];
        }
        
        // Викликаємо розширений метод без регуляризації та dropout
        train(trainInputs, trainTargets, validInputs, validTargets,
              epochs, learningRate, 0, 0, false);
    }
    
    // Forward pass з dropout
    private double[] forwardPassWithDropout(double[] input) {
        double[] layerInput = input;
        double[] layerOutput = null;
        
        for(int i = 0; i < liczba_warstw; i++) {
            // Застосовуємо dropout тільки для прихованих шарів (не для вхідного та вихідного)
            if (useDropout && i > 0 && i < liczba_warstw - 1) {
                layerOutput = warstwy[i].obliczWyjscieZDropout(layerInput, dropoutRate);
            } else {
                layerOutput = warstwy[i].oblicz_wyjscie(layerInput);
            }
            layerInput = layerOutput;
        }
        
        return layerOutput;
    }
    
    // Обчислення помилки з регуляризацією
    private double calculateErrorWithRegularization(double[] output, double[] targets) {
        double error = calculateError(output, targets);
        
        // Додаємо L2 регуляризацію
        if (regularizationLambda > 0) {
            double sumOfSquaredWeights = 0.0;
            
            for (int i = 0; i < liczba_warstw; i++) {
                for (int j = 0; j < warstwy[i].liczba_neuronow; j++) {
                    double[] weights = warstwy[i].neurony[j].getWeights();
                    
                    // Підсумовуємо квадрати ваг (пропускаємо bias)
                    for (int k = 1; k < weights.length; k++) {
                        sumOfSquaredWeights += weights[k] * weights[k];
                    }
                }
            }
            
            // Додаємо регуляризаційний член (lambda * sum(w^2)) / 2
            error += (regularizationLambda * sumOfSquaredWeights) / 2.0;
        }
        
        return error;
    }
    
    // Backpropagation з регуляризацією
    private void backpropagateWithRegularization(double[] targets) {
        // Обчислюємо дельти як і раніше
        warstwy[liczba_warstw - 1].calculateOutputDeltas(targets);
        
        for(int i = liczba_warstw - 2; i >= 0; i--) {
            warstwy[i].calculateHiddenDeltas(warstwy[i + 1]);
        }
        
        // Оновлюємо ваги з регуляризацією
        for(int i = 0; i < liczba_warstw; i++) {
            warstwy[i].updateWeightsWithRegularization(learningRate, regularizationLambda);
        }
    }
    
    // Оцінка загальної помилки на наборі даних
    private double evaluateError(double[][] inputs, double[][] targets) {
        double totalError = 0.0;
        
        for (int i = 0; i < inputs.length; i++) {
            double[] output = oblicz_wyjscie(inputs[i]); // Без dropout під час оцінювання
            totalError += calculateError(output, targets[i]); // Без регуляризації
        }
        
        return totalError / inputs.length;
    }
    
    // Перевірка умов для раннього припинення
    private boolean checkEarlyStop(double validationError) {
        if (validationError < bestValidationError) {
            bestValidationError = validationError;
            saveBestWeights();
            patienceCounter = 0;
            return false;
        } else {
            patienceCounter++;
            return patienceCounter >= patience;
        }
    }
    
    // Збереження найкращих ваг
    private void saveBestWeights() {
        bestWeights = new Warstwa[liczba_warstw];
        
        for (int i = 0; i < liczba_warstw; i++) {
            bestWeights[i] = warstwy[i].deepCopy();
        }
    }
    
    // Відновлення найкращих ваг
    private void restoreBestWeights() {
        if (bestWeights != null) {
            for (int i = 0; i < liczba_warstw; i++) {
                warstwy[i] = bestWeights[i];
            }
        }
    }
    
    // Перемішування індексів для стохастичного градієнтного спуску
    private int[] shuffleIndices(int size) {
        int[] indices = new int[size];
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        
        for (int i = size - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        return indices;
    }
    
    // Методи для збереження та завантаження моделі
    
    // Збереження моделі у файл
    public void saveModel(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
            System.out.println("Модель збережено у файл: " + filePath);
        }
    }
    
    // Статичний метод для завантаження моделі з файлу
    public static Siec loadModel(String filePath) throws IOException, ClassNotFoundException {
        // Перевірка існування файлу
        File file = new File(filePath);
        if (!file.exists()) {
            throw new IOException("Файл моделі не знайдено: " + filePath);
        }
        
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            Siec model = (Siec) ois.readObject();
            System.out.println("Модель завантажено з файлу: " + filePath);
            return model;
        } catch (InvalidClassException e) {
            throw new IOException("Несумісна версія моделі або пошкоджений файл: " + e.getMessage(), e);
        } catch (ClassNotFoundException e) {
            throw new ClassNotFoundException("Не вдалося десеріалізувати модель: " + e.getMessage(), e);
        }
    }
    
    // Метод для обчислення помилки (cross-entropy для softmax)
    private double calculateError(double[] output, double[] targets) {
        double error = 0.0;
        
        // Додаємо захист від NaN та перевірку значень
        for(int i = 0; i < output.length; i++) {
            // Додаємо невелику константу epsilon для уникнення log(0)
            // і обмежуємо мінімальне значення для виходів, щоб уникнути надто малих значень
            double safeOutput = Math.max(output[i], 1e-15);
            
            if(targets[i] > 0) {
                error -= targets[i] * Math.log(safeOutput);
            }
        }
        
        // Перевірка на NaN або Infinity
        if (Double.isNaN(error) || Double.isInfinite(error)) {
            System.err.println("УВАГА: Виявлено NaN або Infinity у функції втрат. Повертаємо велике, але скінченне значення.");
            return 1000.0; // Використовуємо велике, але скінченне значення як запобіжник
        }
        
        return error;
    }
    
    // Метод для зворотного поширення помилки
    private void backpropagate(double[] targets) {
        // 1. Обчислюємо дельти для вихідного шару
        warstwy[liczba_warstw - 1].calculateOutputDeltas(targets);
        
        // 2. Обчислюємо дельти для прихованих шарів (від передостаннього до першого)
        for(int i = liczba_warstw - 2; i >= 0; i--) {
            warstwy[i].calculateHiddenDeltas(warstwy[i + 1]);
        }
        
        // 3. Оновлюємо ваги для всіх шарів
        for(int i = 0; i < liczba_warstw; i++) {
            warstwy[i].updateWeights(learningRate);
        }
    }
    
    // Метод для обчислення точності на тестовому наборі
    public double evaluateAccuracy(double[][] inputs, double[][] targets) {
        int correct = 0;
        
        for(int i = 0; i < inputs.length; i++) {
            int predictedClass = predict(inputs[i]);
            int actualClass = getActualClass(targets[i]);
            
            if(predictedClass == actualClass) {
                correct++;
            }
        }
        
        return (double) correct / inputs.length;
    }
    
    // Метод для обчислення матриці помилок
    public int[][] computeConfusionMatrix(double[][] inputs, double[][] targets, int numClasses) {
        int[][] confusionMatrix = new int[numClasses][numClasses];
        
        for(int i = 0; i < inputs.length; i++) {
            int predictedClass = predict(inputs[i]);
            int actualClass = getActualClass(targets[i]);
            
            // Інкрементуємо відповідний елемент матриці
            confusionMatrix[actualClass][predictedClass]++;
        }
        
        return confusionMatrix;
    }
    
    // Допоміжний метод для визначення справжнього класу
    private int getActualClass(double[] target) {
        int actualClass = 0;
        
        for(int j = 1; j < target.length; j++) {
            if(target[j] > target[actualClass]) {
                actualClass = j;
            }
        }
        
        return actualClass;
    }
    
    // Метод для перевірки наявності NaN у масиві
    private boolean containsNaN(double[] array) {
        for (double value : array) {
            if (Double.isNaN(value) || Double.isInfinite(value)) {
                return true;
            }
        }
        return false;
    }
    
    // Допоміжний метод для обмеження розміру градієнтів
    private void backpropagateWithRegularizationAndClipping(double[] targets) {
        // Обчислюємо дельти для вихідного шару
        warstwy[liczba_warstw - 1].calculateOutputDeltas(targets);
        
        // Обчислюємо дельти для прихованих шарів (від передостаннього до першого)
        for(int i = liczba_warstw - 2; i >= 0; i--) {
            warstwy[i].calculateHiddenDeltas(warstwy[i + 1]);
        }
        
        // Обмежуємо дельти для запобігання вибуху градієнтів
        double maxDelta = 1.0; // Максимальне значення дельти
        for(int i = 0; i < liczba_warstw; i++) {
            for(int j = 0; j < warstwy[i].liczba_neuronow; j++) {
                double delta = warstwy[i].neurony[j].getDelta();
                if (Math.abs(delta) > maxDelta) {
                    // Нормалізуємо дельту, зберігаючи знак
                    warstwy[i].neurony[j].clipDelta(maxDelta);
                }
            }
        }
        
        // Оновлюємо ваги з регуляризацією
        for(int i = 0; i < liczba_warstw; i++) {
            warstwy[i].updateWeightsWithRegularization(learningRate, regularizationLambda);
        }
    }
    
    // Метод для налаштування автозбереження
    public void configureAutoSave(boolean enabled, int frequency, String directory) {
        this.autoSaveEnabled = enabled;
        this.autoSaveFrequency = Math.max(1, frequency);  // Мінімум 1
        this.autoSaveDirectory = directory;
    }
}
