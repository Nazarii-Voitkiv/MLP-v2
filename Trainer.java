import java.io.File;
import java.io.IOException;
import java.util.List;

public class Trainer {
    
    private static final String MODEL_PATH = "model.dat";
    
    public static void main(String[] args) {
        List<Sample> samples = MyDataLoader.loadSamples();

        if (samples.isEmpty()) {
            System.err.println("Błąd: brak próbek do treningu. Sprawdź folder data/");
            return;
        }

        NeuralNetwork net = new NeuralNetwork();
        File modelFile = new File(MODEL_PATH);
        
        if (modelFile.exists()) {
            try {
                net.loadModel(MODEL_PATH);
            } catch (IOException | ClassNotFoundException e) {
                trainNewModel(net, samples);
            }
        } else {
            trainNewModel(net, samples);
        }

        calculateAndPrintAccuracy(net, samples);
    }
    
    private static void trainNewModel(NeuralNetwork net, List<Sample> samples) {
        net.setDropoutRate(0.0);
        net.setPatience(10);
        net.setValidationSplit(0.2);
        net.train(samples, 200);

        try {
            net.saveModel(MODEL_PATH);
        } catch (IOException e) {
            System.err.println("Błąd podczas zapisywania modelu: " + e.getMessage());
        }
    }
    
    private static void calculateAndPrintAccuracy(NeuralNetwork net, List<Sample> samples) {
        int correctPredictions = 0;
        int totalSamples = samples.size();
        
        for (Sample sample : samples) {
            double[] prediction = net.predict(sample.getInput());
            int predictedIndex = findMaxIndex(prediction);
            int targetIndex = findMaxIndex(sample.getTarget());
            
            if (predictedIndex == targetIndex) {
                correctPredictions++;
            }
        }
        
        double accuracy = (double) correctPredictions / totalSamples * 100;
        System.out.printf("Dokładność: %.2f%% (%d/%d poprawnych prognoz)%n", 
                         accuracy, correctPredictions, totalSamples);
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
