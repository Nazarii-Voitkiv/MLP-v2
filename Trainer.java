import java.io.File;
import java.io.IOException;
import java.util.*;

public class Trainer {
    private static final String MODEL_PATH = "model.dat";
    private static final char[] LETTERS = {'M', 'O', 'N'};
    
    public static void main(String[] args) {
        List<Sample> samples = MyDataLoader.loadSamples();
        if (samples.isEmpty()) {
            System.err.println("Błąd: brak próbek do treningu. Sprawdź folder data/");
            return;
        }
        
        NeuralNetwork net = new NeuralNetwork();
        if (new File(MODEL_PATH).exists()) {
            try {
                net.loadModel(MODEL_PATH);
                System.out.println("Załadowano istniejący model.");
                System.out.println("Przeprowadzam trening istniejącego modelu...");
                configureNetworkForTraining(net);
                trainAndSaveModel(net, samples);
            } catch (IOException | ClassNotFoundException e) {
                System.out.println("Nie udało się załadować istniejącego modelu. Tworzę nowy model...");
                createAndTrainNewModel(net, samples);
            }
        } else {
            createAndTrainNewModel(net, samples);
        }
    }
    
    private static void configureNetworkForTraining(NeuralNetwork net) {
        net.setPatience(25);
        net.setValidationSplit(0.2);
        net.setDropoutRate(0.0);
        net.setInitialLearningRate(0.0001);  
        net.setPeakLearningRate(0.003);
        net.setWarmupEpochs(15);
    }
    
    private static void createAndTrainNewModel(NeuralNetwork net, List<Sample> samples) {
        System.out.println("Tworzę nowy model...");
        configureNetworkForTraining(net);
        trainAndSaveModel(net, samples);
    }
    
    private static void trainAndSaveModel(NeuralNetwork net, List<Sample> samples) {
        List<Sample> balancedSamples = balanceSamples(samples);
        net.train(balancedSamples, 300);
        
        try {
            net.saveModel(MODEL_PATH);
            System.out.println("Model został zapisany do " + MODEL_PATH);
        } catch (IOException e) {
            System.err.println("Błąd podczas zapisywania modelu: " + e.getMessage());
        }
    }
    
    private static List<Sample> balanceSamples(List<Sample> samples) {
        int[] countPerClass = new int[LETTERS.length];
        List<List<Sample>> samplesPerClass = new ArrayList<>();
        
        for (int i = 0; i < LETTERS.length; i++) {
            samplesPerClass.add(new ArrayList<>());
        }
        
        for (Sample sample : samples) {
            int classIndex = findMaxIndex(sample.getTarget());
            samplesPerClass.get(classIndex).add(sample);
            countPerClass[classIndex]++;
        }
        
        int maxCount = Arrays.stream(countPerClass).max().orElse(0);
        
        List<Sample> balancedSamples = new ArrayList<>();
        for (int i = 0; i < LETTERS.length; i++) {
            List<Sample> classSamples = samplesPerClass.get(i);
            balancedSamples.addAll(classSamples);
            
            if (classSamples.isEmpty() || countPerClass[i] >= maxCount) {
                continue;
            }
            
            int duplicatesNeeded = maxCount - countPerClass[i];
            int fullCopies = duplicatesNeeded / countPerClass[i];
            int remainder = duplicatesNeeded % countPerClass[i];
            
            for (int j = 0; j < fullCopies; j++) {
                balancedSamples.addAll(classSamples);
            }
            
            for (int j = 0; j < remainder; j++) {
                balancedSamples.add(classSamples.get(j % classSamples.size()));
            }
        }
        
        Collections.shuffle(balancedSamples);
        System.out.println("Zrównoważony zbiór treningowy: " + balancedSamples.size() + " próbek");
        
        return balancedSamples;
    }
    
    private static int findMaxIndex(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];
        
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
}
