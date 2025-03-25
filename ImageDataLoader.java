import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ImageDataLoader {
    public static final String[] CLASSES = {"M", "N", "O"};
    public static final int IMAGE_SIZE = 32;
    public static final int INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE;
    public static final int OUTPUT_SIZE = CLASSES.length;
    
    public static class Dataset {
        public List<double[]> inputs;
        public List<double[]> targets;
        
        public Dataset() {
            inputs = new ArrayList<>();
            targets = new ArrayList<>();
        }
        
        public int size() {
            return inputs.size();
        }
    }
    
    /**
     * Завантажує зображення з каталогу та перетворює їх на вектори входів та цілей
     * 
     * @param baseDir Базовий каталог (resized_data)
     * @param setType Тип набору даних ("train" або "test")
     * @return Об'єкт Dataset, що містить вхідні та цільові вектори
     */
    public static Dataset loadDataset(String baseDir, String setType) {
        Dataset dataset = new Dataset();
        
        // Перевірка, чи існує базовий каталог
        File baseFolder = new File(baseDir, setType);
        if (!baseFolder.exists() || !baseFolder.isDirectory()) {
            System.err.println("Каталог не існує: " + baseFolder.getAbsolutePath());
            return dataset;
        }
        
        int totalFilesFound = 0;      // Загальна кількість знайдених файлів
        int successfullyLoaded = 0;   // Кількість успішно завантажених файлів
        
        // Обробка кожного класу (M, N, O)
        for (int classIndex = 0; classIndex < CLASSES.length; classIndex++) {
            String className = CLASSES[classIndex];
            File classFolder = new File(baseFolder, className);
            
            if (!classFolder.exists() || !classFolder.isDirectory()) {
                System.err.println("Каталог класу не існує: " + classFolder.getAbsolutePath());
                continue;
            }
            
            // Отримання всіх зображень у каталозі класу
            File[] imageFiles = classFolder.listFiles((dir, name) -> 
                name.toLowerCase().endsWith(".png") || 
                name.toLowerCase().endsWith(".jpg") || 
                name.toLowerCase().endsWith(".jpeg"));
            
            if (imageFiles == null || imageFiles.length == 0) {
                System.err.println("Немає зображень у каталозі: " + classFolder.getAbsolutePath());
                continue;
            }
            
            int filesInClass = imageFiles.length;
            totalFilesFound += filesInClass;
            int loadedInClass = 0;
            
            // Обробка кожного зображення
            for (File imageFile : imageFiles) {
                try {
                    BufferedImage img = ImageIO.read(imageFile);
                    if (img == null) {
                        System.err.println("Не вдалося прочитати зображення: " + imageFile.getAbsolutePath());
                        continue;
                    }
                    
                    // Перевірка розміру зображення
                    if (img.getWidth() != IMAGE_SIZE || img.getHeight() != IMAGE_SIZE) {
                        System.err.println("Розмір зображення не 32x32: " + imageFile.getAbsolutePath());
                        continue;
                    }
                    
                    // Конвертація зображення у вектор входів
                    double[] input = new double[INPUT_SIZE];
                    int index = 0;
                    for (int y = 0; y < IMAGE_SIZE; y++) {
                        for (int x = 0; x < IMAGE_SIZE; x++) {
                            int rgb = img.getRGB(x, y);
                            int r = (rgb >> 16) & 0xFF;
                            int g = (rgb >> 8) & 0xFF;
                            int b = rgb & 0xFF;
                            
                            // Перетворення в відтінки сірого та нормалізація [0..1]
                            double gray = (r + g + b) / (3.0 * 255.0);
                            // Інвертування значення (чорний = 1, білий = 0) для кращого розпізнавання
                            input[index++] = 1.0 - gray;
                        }
                    }
                    
                    // Створення цільового вектора (one-hot encoding)
                    double[] target = new double[OUTPUT_SIZE];
                    target[classIndex] = 1.0;
                    
                    // Додавання векторів до набору даних
                    dataset.inputs.add(input);
                    dataset.targets.add(target);
                    loadedInClass++;
                    successfullyLoaded++;
                    
                } catch (IOException e) {
                    System.err.println("Помилка при обробці зображення " + imageFile.getName() + ": " + e.getMessage());
                }
            }
            
            System.out.println("Клас " + className + ": знайдено " + filesInClass + 
                              " зображень, успішно завантажено " + loadedInClass);
        }
        
        System.out.println("Усього для набору " + setType + ": знайдено " + totalFilesFound + 
                          " зображень, успішно завантажено " + successfullyLoaded);
        
        // Перемішування даних
        if (!dataset.inputs.isEmpty()) {
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
        
        return dataset;
    }
    
    /**
     * Головний метод для демонстрації завантаження даних
     */
    public static void main(String[] args) {
        String baseDir = "resized_data";
        
        System.out.println("Завантаження навчальних даних...");
        Dataset trainDataset = loadDataset(baseDir, "train");
        System.out.println("Загалом у навчальному наборі: " + trainDataset.size() + " зразків");
        
        System.out.println("\nЗавантаження тестових даних...");
        Dataset testDataset = loadDataset(baseDir, "test");
        System.out.println("Загалом у тестовому наборі: " + testDataset.size() + " зразків");
        
        System.out.println("\nВикористовуємо всі " + trainDataset.size() + 
                          " зображень з папки train для навчання");
        System.out.println("Використовуємо всі " + testDataset.size() + 
                          " зображень з папки test для тестування");
    }
}
