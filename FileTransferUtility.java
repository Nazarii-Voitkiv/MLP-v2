import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/**
 * Narzędzie do transferu plików między katalogami
 */
public class FileTransferUtility {
    private static final String SOURCE_DIR = "data 1";
    private static final String TARGET_DIR = "data";

    public static void main(String[] args) {
        System.out.println("Przenoszenie plików z folderu " + SOURCE_DIR + " do folderu " + TARGET_DIR); // Translated
        
        // Sprawdzanie istnienia katalogów
        File sourceDir = new File(SOURCE_DIR);
        File targetDir = new File(TARGET_DIR);
        
        if (!sourceDir.exists() || !sourceDir.isDirectory()) {
            System.err.println("Błąd: Folder " + SOURCE_DIR + " nie istnieje lub nie jest katalogiem."); // Translated
            return;
        }
        
        if (!targetDir.exists()) {
            if (!targetDir.mkdirs()) {
                System.err.println("Błąd: Nie udało się utworzyć folderu " + TARGET_DIR); // Translated
                return;
            }
            System.out.println("Utworzono folder " + TARGET_DIR); // Translated
        }
        
        // Pobieranie listy plików z katalogu źródłowego
        File[] sourceFiles = sourceDir.listFiles(file -> file.isFile() && file.getName().endsWith(".csv"));
        
        if (sourceFiles == null || sourceFiles.length == 0) {
            System.out.println("W folderze " + SOURCE_DIR + " nie ma plików CSV do przeniesienia."); // Translated
            return;
        }
        
        System.out.println("Znaleziono " + sourceFiles.length + " plików CSV do przeniesienia."); // Translated
        
        // Słowniki do przechowywania maksymalnych indeksów dla każdej litery
        Map<Character, Integer> maxNumbers = new HashMap<>();
        
        // Znalezienie istniejących maksymalnych indeksów w katalogu docelowym
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
        
        // Przenoszenie i zmiana nazw plików
        int successCount = 0;
        
        for (File sourceFile : sourceFiles) {
            String fileName = sourceFile.getName();
            Pattern pattern = Pattern.compile("([MON])_(\\d+)\\.csv");
            Matcher matcher = pattern.matcher(fileName);
            
            if (matcher.matches()) {
                char letter = matcher.group(1).charAt(0);
                
                // Zwiększamy maksymalny indeks dla tej litery
                int newNumber = maxNumbers.getOrDefault(letter, 0) + 1;
                maxNumbers.put(letter, newNumber);
                
                // Tworzymy nową nazwę pliku
                String newFileName = String.format("%c_%02d.csv", letter, newNumber);
                File targetFile = new File(targetDir, newFileName);
                
                try {
                    // Kopiujemy plik z nową nazwą
                    Files.copy(sourceFile.toPath(), targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    successCount++;
                    System.out.println("Przeniesiono " + fileName + " -> " + newFileName); // Translated
                } catch (IOException e) {
                    System.err.println("Błąd przenoszenia pliku " + fileName + ": " + e.getMessage()); // Translated
                }
            } else {
                System.out.println("Pominięto plik " + fileName + " (nieprawidłowy format nazwy)"); // Translated
            }
        }
        
        System.out.println("Przenoszenie zakończone. Pomyślnie przeniesiono " + successCount + " z " + sourceFiles.length + " plików."); // Translated
    }
}
