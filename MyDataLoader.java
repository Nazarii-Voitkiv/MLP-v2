import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

public class MyDataLoader {
    private static final String DATA_DIR = "data";
    private static final Pattern FILE_PATTERN = Pattern.compile("([MON])_(\\d+)\\.csv");
    
    public static List<Sample> loadSamples() {
        return loadSamplesFromDir(DATA_DIR);
    }
    
    public static List<Sample> loadSamplesFromDir(String dirPath) {
        List<Sample> samples = new ArrayList<>();
        File dataDir = new File(dirPath);
        
        if (!dataDir.exists() || !dataDir.isDirectory()) {
            return samples;
        }
        
        File[] files = dataDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".csv"));
        
        if (files == null || files.length == 0) {
            return samples;
        }
        
        for (File file : files) {
            processFile(file, samples);
        }
        
        return samples;
    }
    
    private static void processFile(File file, List<Sample> samples) {
        try {
            Matcher matcher = FILE_PATTERN.matcher(file.getName());
            if (!matcher.matches()) {
                return;
            }
            
            char letter = matcher.group(1).charAt(0);
            double[] target = createTargetArray(letter);
            double[] input = parseCSVContent(new String(Files.readAllBytes(file.toPath())));

            if (input.length != 784) {
                return;
            }

            samples.add(new Sample(input, target));
            
        } catch (IOException e) {

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
                result[i] = 0.0;
            }
        }
        
        return result;
    }
}
