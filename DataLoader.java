import java.io.*;
import java.util.*;

public class DataLoader {

    // Завантажує тільки букви M (13), N (14), O (15)
    public static List<Sample> loadMONSamples(String csvPath, int maxSamplesPerClass) throws IOException {
        List<Sample> samples = new ArrayList<>();
        Map<Integer, Integer> classCounter = new HashMap<>();
        classCounter.put(13, 0); // M
        classCounter.put(14, 0); // N
        classCounter.put(15, 0); // O

        BufferedReader reader = new BufferedReader(new FileReader(csvPath));
        String line;

        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            int label = Integer.parseInt(parts[0]);

            if (label >= 13 && label <= 15) {
                if (classCounter.get(label) >= maxSamplesPerClass) continue;

                double[] input = new double[784]; // 28x28
                for (int i = 0; i < 784; i++) {
                    input[i] = Integer.parseInt(parts[i + 1]) / 255.0;
                }

                double[] target = new double[3]; // M = [1,0,0], N = [0,1,0], O = [0,0,1]
                target[label - 13] = 1.0;

                samples.add(new Sample(input, target));
                classCounter.put(label, classCounter.get(label) + 1);
            }

            if (classCounter.get(13) >= maxSamplesPerClass &&
                classCounter.get(14) >= maxSamplesPerClass &&
                classCounter.get(15) >= maxSamplesPerClass) {
                break;
            }
        }

        reader.close();
        return samples;
    }
}
