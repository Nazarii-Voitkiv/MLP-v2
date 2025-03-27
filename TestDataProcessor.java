import java.io.IOException;
import java.io.File;

public class TestDataProcessor {
    public static void main(String[] args) {
        try {
            // Створюємо екземпляр класу
            DataProcessor processor = new DataProcessor();
            
            // Вказуємо шлях до CSV-файлу
            String filePath = "letters_dataset.csv";
            
            // Перевіряємо чи існує файл
            File f = new File(filePath);
            if (!f.exists()) {
                System.out.println("УВАГА: Файл не знайдено: " + f.getAbsolutePath());
                System.out.println("Спробуйте вказати повний шлях до файлу.");
                return;
            }
            
            // Кількість зразків для тестового завантаження (10 для швидкого тестування)
            int testSamplesCount = 10;
            
            System.out.println("Спроба завантажити " + testSamplesCount + " зразків з: " + f.getAbsolutePath());
            processor.loadDataFromCSV(filePath, testSamplesCount);
            
            // Виводимо інформацію про завантажені дані
            System.out.println(processor.getDataInfo());
            
            // Отримуємо зображення і мітки
            double[][] images = processor.getImages();
            double[][] labels = processor.getLabels();
            
            // Виводимо кілька прикладів для перевірки
            if (images.length > 0) {
                System.out.println("\nПриклад першого зображення:");
                System.out.println("Кількість пікселів: " + images[0].length);
                System.out.println("Перші 10 пікселів:");
                for (int i = 0; i < 10 && i < images[0].length; i++) {
                    System.out.printf("%.4f ", images[0][i]);
                }
                System.out.println();
                
                System.out.println("\nВідповідна мітка (one-hot encoding):");
                System.out.printf("[%.0f, %.0f, %.0f] - це мітка %s\n", 
                    labels[0][0], labels[0][1], labels[0][2],
                    getClassFromOneHot(labels[0]));
            }
            
            // Додаємо повідомлення про успішне тестування
            System.out.println("\nЯкщо тест з 10 зразками успішний, спробуйте повноцінне завантаження:");
            System.out.println("1) Змінити testSamplesCount = -1 у TestDataProcessor.java");
            System.out.println("2) Запустити з більшим об'ємом пам'яті: java -Xmx4g TestDataProcessor");
            
        } catch (IOException e) {
            System.err.println("Помилка при завантаженні файлу: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("Несподівана помилка: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // Метод для перетворення one-hot encoding назад у літеру класу
    private static String getClassFromOneHot(double[] oneHot) {
        if (oneHot[0] == 1.0) return "M";
        if (oneHot[1] == 1.0) return "O";
        if (oneHot[2] == 1.0) return "N";
        return "Невідомий клас";
    }
}
