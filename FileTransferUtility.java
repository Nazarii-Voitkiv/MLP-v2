import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

public class FileTransferUtility {
    private static final String SOURCE_DIR = "data 1";
    private static final String TARGET_DIR = "data";

    public static void main(String[] args) {
        System.out.println("Перенесення файлів з папки " + SOURCE_DIR + " до папки " + TARGET_DIR);
        
        // Перевірка існування директорій
        File sourceDir = new File(SOURCE_DIR);
        File targetDir = new File(TARGET_DIR);
        
        if (!sourceDir.exists() || !sourceDir.isDirectory()) {
            System.err.println("Помилка: Папка " + SOURCE_DIR + " не існує або не є директорією.");
            return;
        }
        
        if (!targetDir.exists()) {
            if (!targetDir.mkdirs()) {
                System.err.println("Помилка: Не вдалося створити папку " + TARGET_DIR);
                return;
            }
            System.out.println("Створено папку " + TARGET_DIR);
        }
        
        // Отримання списку файлів з вихідної директорії
        File[] sourceFiles = sourceDir.listFiles(file -> file.isFile() && file.getName().endsWith(".csv"));
        
        if (sourceFiles == null || sourceFiles.length == 0) {
            System.out.println("У папці " + SOURCE_DIR + " немає CSV-файлів для перенесення.");
            return;
        }
        
        System.out.println("Знайдено " + sourceFiles.length + " CSV-файлів для перенесення.");
        
        // Словники для зберігання максимальних індексів для кожної літери
        Map<Character, Integer> maxNumbers = new HashMap<>();
        
        // Знайти існуючі максимальні індекси в цільовій директорії
        File[] targetFiles = targetDir.listFiles(file -> file.isFile() && file.getName().endsWith(".csv"));
        if (targetFiles != null) {
            Pattern pattern = Pattern.compile("([MON])_(\\d+)\\.csv");
            
            for (File file : targetFiles) {
                Matcher matcher = pattern.matcher(file.getName());
                if (matcher.matches()) {
                    char letter = matcher.group(1).charAt(0);
                    int number = Integer.parseInt(matcher.group(2));
                    
                    maxNumbers.put(letter, Math.max(maxNumbers.getOrDefault(letter, 0), number));
                }
            }
        }
        
        // Перенесення та перейменування файлів
        int successCount = 0;
        
        for (File sourceFile : sourceFiles) {
            String fileName = sourceFile.getName();
            Pattern pattern = Pattern.compile("([MON])_(\\d+)\\.csv");
            Matcher matcher = pattern.matcher(fileName);
            
            if (matcher.matches()) {
                char letter = matcher.group(1).charAt(0);
                
                // Збільшуємо максимальний індекс для цієї літери
                int newNumber = maxNumbers.getOrDefault(letter, 0) + 1;
                maxNumbers.put(letter, newNumber);
                
                // Створюємо нову назву файлу
                String newFileName = String.format("%c_%02d.csv", letter, newNumber);
                File targetFile = new File(targetDir, newFileName);
                
                try {
                    // Копіюємо файл з новим іменем
                    Files.copy(sourceFile.toPath(), targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    successCount++;
                    System.out.println("Перенесено " + fileName + " -> " + newFileName);
                } catch (IOException e) {
                    System.err.println("Помилка перенесення файлу " + fileName + ": " + e.getMessage());
                }
            } else {
                System.out.println("Пропущено файл " + fileName + " (неправильний формат імені)");
            }
        }
        
        System.out.println("Перенесення завершено. Успішно перенесено " + successCount + " з " + sourceFiles.length + " файлів.");
    }
}
