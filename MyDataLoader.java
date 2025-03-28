import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

public class MyDataLoader {
    private static final String DATA_DIR = "data";
    
    public static List<Sample> loadSamples() {
        List<Sample> samples = new ArrayList<>();
        File dataDir = new File(DATA_DIR);
        
        if (!dataDir.exists() || !dataDir.isDirectory()) {
            System.err.println("Помилка: директорія " + DATA_DIR + " не існує");
            return samples;
        }
        
        File[] files = dataDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".csv"));
        
        if (files == null || files.length == 0) {
            System.out.println("Не знайдено CSV файлів в директорії " + DATA_DIR);
            return samples;
        }
        
        Pattern pattern = Pattern.compile("([MON])_(\\d+)\\.csv");
        
        for (File file : files) {
            try {
                // Визначаємо літеру з назви файлу
                Matcher matcher = pattern.matcher(file.getName());
                if (!matcher.matches()) {
                    System.out.println("Пропускаємо файл з неправильним форматом імені: " + file.getName());
                    continue;
                }
                
                char letter = matcher.group(1).charAt(0);
                
                // Створюємо цільовий масив відповідно до літери
                double[] target = createTargetArray(letter);
                
                // Зчитуємо вміст CSV файлу
                String content = new String(Files.readAllBytes(file.toPath()));
                double[] input = parseCSVContent(content);
                
                // Перевіряємо розмір масиву вхідних даних
                if (input.length != 784) {
                    System.out.println("Попередження: " + file.getName() + 
                                       " містить " + input.length + 
                                       " елементів замість 784. Пропускаємо файл.");
                    continue;
                }
                
                // Додаємо зразок до списку
                samples.add(new Sample(input, target));
                
            } catch (IOException e) {
                System.err.println("Помилка при читанні файлу " + file.getName() + ": " + e.getMessage());
            }
        }
        
        System.out.println("Завантажено " + samples.size() + " зразків з директорії " + DATA_DIR);
        return samples;
    }
    
    private static double[] createTargetArray(char letter) {
        double[] target = new double[3]; // [M, O, N]
        
        switch (letter) {
            case 'M':
                target[0] = 1.0; // [1, 0, 0]
                break;
            case 'O':
                target[1] = 1.0; // [0, 1, 0]
                break;
            case 'N':
                target[2] = 1.0; // [0, 0, 1]
                break;
        }
        
        return target;
    }
    
    private static double[] parseCSVContent(String content) {
        String[] values = content.trim().split(",");
        double[] result = new double[values.length];
        
        for (int i = 0; i < values.length; i++) {
            try {
                result[i] = Double.parseDouble(values[i]);
            } catch (NumberFormatException e) {
                System.err.println("Некоректний формат числа: " + values[i] + ". Використовуємо 0.0");
                result[i] = 0.0;
            }
        }
        
        return result;
    }
}
