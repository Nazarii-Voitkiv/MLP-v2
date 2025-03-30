import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

public class MyDataLoader {
    private static final String DATA_DIR = "data";
    private static final Pattern FILE_PATTERN = Pattern.compile("([MON])_(\\d+)\\.csv");
    
    public static List<Sample> loadSamples() {
        List<Sample> samples = new ArrayList<>();
        File dataDir = new File(DATA_DIR);
        
        if (!isValidDirectory(dataDir)) {
            return samples;
        }
        
        File[] files = dataDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".csv"));
        
        if (files == null || files.length == 0) {
            System.out.println("Nie znaleziono plików CSV w katalogu " + DATA_DIR);
            return samples;
        }
        
        for (File file : files) {
            processFile(file, samples);
        }
        
        System.out.println("Załadowano " + samples.size() + " próbek z katalogu " + DATA_DIR);
        return samples;
    }
    
    private static boolean isValidDirectory(File dir) {
        if (!dir.exists() || !dir.isDirectory()) {
            System.err.println("Błąd: katalog " + DATA_DIR + " nie istnieje");
            return false;
        }
        return true;
    }
    
    private static void processFile(File file, List<Sample> samples) {
        try {
            Matcher matcher = FILE_PATTERN.matcher(file.getName());
            if (!matcher.matches()) {
                System.out.println("Pomijamy plik z nieprawidłowym formatem nazwy: " + file.getName());
                return;
            }
            
            char letter = matcher.group(1).charAt(0);
            double[] target = createTargetArray(letter);
            double[] input = parseCSVContent(new String(Files.readAllBytes(file.toPath())));

            if (input.length != 784) {
                System.out.println("Ostrzeżenie: " + file.getName() + 
                                   " zawiera " + input.length + 
                                   " elementów zamiast 784. Pomijamy plik.");
                return;
            }

            samples.add(new Sample(input, target));
            
        } catch (IOException e) {
            System.err.println("Błąd odczytu pliku " + file.getName() + ": " + e.getMessage());
        }
    }
    
    private static double[] createTargetArray(char letter) {
        double[] target = new double[3];
        
        switch (letter) {
            case 'M': target[0] = 1.0; break;
            case 'O': target[1] = 1.0; break;
            case 'N': target[2] = 1.0; break;
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
                System.err.println("Nieprawidłowy format liczby: " + values[i] + ". Używamy 0.0");
                result[i] = 0.0;
            }
        }
        
        return result;
    }
}
