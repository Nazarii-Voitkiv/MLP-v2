public class LetterClassificationTest {
    public static void main(String[] args) {
        try {
            System.out.println("Створення нейронної мережі для класифікації літер...");
            
            // Створення мережі за заданою архітектурою
            Siec neuralNetwork = Siec.createLetterClassificationNetwork();
            
            System.out.println("Архітектура мережі:");
            System.out.println("Вхідний шар: 16384 нейронів (128x128 пікселів)");
            System.out.println("Прихований шар 1: 1024 нейронів з ReLU активацією");
            System.out.println("Прихований шар 2: 512 нейронів з ReLU активацією");
            System.out.println("Прихований шар 3: 128 нейронів з ReLU активацією");
            System.out.println("Вихідний шар: 3 нейрони з Softmax активацією (для класів M, O, N)");
            
            // Завантажуємо декілька зразків для тестування
            DataProcessor processor = new DataProcessor();
            String filePath = "letters_dataset.csv";
            
            System.out.println("\nЗавантаження 5 зразків для тестування...");
            processor.loadDataFromCSV(filePath, 5);
            
            double[][] images = processor.getImages();
            double[][] labels = processor.getLabels();
            
            // Перевірка прогнозу для кожного зразка
            if (images.length > 0) {
                System.out.println("\nПеревірка прогнозів (без навчання):");
                for (int i = 0; i < images.length; i++) {
                    double[] input = images[i];
                    
                    // Отримуємо прогноз мережі
                    int predictedClassIndex = neuralNetwork.predict(input);
                    
                    // Знаходимо справжній клас
                    int actualClassIndex = 0;
                    for (int j = 0; j < labels[i].length; j++) {
                        if (labels[i][j] == 1.0) {
                            actualClassIndex = j;
                            break;
                        }
                    }
                    
                    String predictedClass = indexToClass(predictedClassIndex);
                    String actualClass = indexToClass(actualClassIndex);
                    
                    System.out.printf("Зразок %d: Прогноз = %s, Справжній клас = %s%n", 
                                     i+1, predictedClass, actualClass);
                }
            }
            
            System.out.println("\nЗауваження: Мережа ще не навчена, тому прогнози випадкові.");
            System.out.println("Для отримання реальних результатів потрібно реалізувати навчання мережі.");
            
        } catch (Exception e) {
            System.err.println("Помилка при тестуванні мережі: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // Конвертація індексу класу в літеру
    private static String indexToClass(int index) {
        switch (index) {
            case 0: return "M";
            case 1: return "O";
            case 2: return "N";
            default: return "Невідомий";
        }
    }
}
