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
        System.out.println("Przenoszenie plików z folderu " + SOURCE_DIR + " do folderu " + TARGET_DIR);

        if (!validateDirectories()) {
            return;
        }

        File[] sourceFiles = new File(SOURCE_DIR).listFiles(file -> file.isFile() && file.getName().endsWith(".csv"));

        if (sourceFiles == null || sourceFiles.length == 0) {
            System.out.println("W folderze " + SOURCE_DIR + " nie ma plików CSV do przeniesienia.");
            return;
        }

        System.out.println("Znaleziono " + sourceFiles.length + " plików CSV do przeniesienia.");

        Map<Character, Integer> maxNumbers = findMaxNumbers();
        transferFiles(sourceFiles, maxNumbers);
    }

    private static boolean validateDirectories() {
        File sourceDir = new File(SOURCE_DIR);
        File targetDir = new File(TARGET_DIR);

        if (!sourceDir.exists() || !sourceDir.isDirectory()) {
            System.err.println("Błąd: Folder " + SOURCE_DIR + " nie istnieje lub nie jest katalogiem.");
            return false;
        }

        if (!targetDir.exists()) {
            if (!targetDir.mkdirs()) {
                System.err.println("Błąd: Nie udało się utworzyć folderu " + TARGET_DIR);
                return false;
            }
            System.out.println("Utworzono folder " + TARGET_DIR);
        }

        return true;
    }

    private static Map<Character, Integer> findMaxNumbers() {
        Map<Character, Integer> maxNumbers = new HashMap<>();
        File[] targetFiles = new File(TARGET_DIR).listFiles(file -> file.isFile() && file.getName().endsWith(".csv"));

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

        return maxNumbers;
    }

    private static void transferFiles(File[] sourceFiles, Map<Character, Integer> maxNumbers) {
        int successCount = 0;
        Pattern pattern = Pattern.compile("([MON])_(\\d+)\\.csv");

        for (File sourceFile : sourceFiles) {
            String fileName = sourceFile.getName();
            Matcher matcher = pattern.matcher(fileName);

            if (matcher.matches()) {
                char letter = matcher.group(1).charAt(0);

                int newNumber = maxNumbers.getOrDefault(letter, 0) + 1;
                maxNumbers.put(letter, newNumber);

                String newFileName = String.format("%c_%02d.csv", letter, newNumber);
                File targetFile = new File(TARGET_DIR, newFileName);

                try {
                    Files.copy(sourceFile.toPath(), targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    successCount++;
                    System.out.println("Przeniesiono " + fileName + " -> " + newFileName);
                } catch (IOException e) {
                    System.err.println("Błąd przenoszenia pliku " + fileName + ": " + e.getMessage());
                }
            } else {
                System.out.println("Pominięto plik " + fileName + " (nieprawidłowy format nazwy)");
            }
        }

        System.out.println("Przenoszenie zakończone. Pomyślnie przeniesiono " + successCount + " z " + sourceFiles.length + " plików.");
    }
}
