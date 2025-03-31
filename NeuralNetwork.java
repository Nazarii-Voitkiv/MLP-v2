import java.io.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class NeuralNetwork {
    private int inputSize, hidden0Size, hidden1Size, hidden2Size, hidden3Size, hidden4Size, outputSize;
    private int[] layerSizes;
    private double[][][] weights;
    private double[][] biases;
    private double[][][] bestWeights;
    private double[][] bestBiases;
    
    private double learningRate;
    private double dropoutRate = 0.0;
    private boolean isTraining = false;
    
    private int patience = 25;
    private double bestValidationError = Double.MAX_VALUE;
    private int epochsSinceImprovement = 0;
    private double validationSplit = 0.2;
    private double initialLearningRate = 0.0001;
    private double peakLearningRate = 0.003;
    private int warmupEpochs = 15;

    public NeuralNetwork(int inputSize, int hidden0Size, int hidden1Size, int hidden2Size, 
                         int hidden3Size, int hidden4Size, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hidden0Size = hidden0Size;
        this.hidden1Size = hidden1Size;
        this.hidden2Size = hidden2Size;
        this.hidden3Size = hidden3Size;
        this.hidden4Size = hidden4Size;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        
        this.layerSizes = new int[]{inputSize, hidden0Size, hidden1Size, hidden2Size, hidden3Size, hidden4Size, outputSize};
        
        int numLayers = layerSizes.length - 1;
        this.weights = new double[numLayers][][];
        this.biases = new double[numLayers][];
        
        initializeWeightsAndBiases();
    }
    
    public NeuralNetwork() {
        // Use architecture: 784 → 512 → 256 → 128 → 32 → 16 → 3
        this(784, 512, 256, 128, 32, 16, 3, 0.0001);
    }
    
    public void setDropoutRate(double rate) {
        if (rate < 0.0 || rate >= 1.0) throw new IllegalArgumentException("Dropout must be between 0 and 1");
        this.dropoutRate = rate;
    }

    public void setPatience(int patience) {
        this.patience = patience;
    }

    public void setValidationSplit(double ratio) {
        if (ratio <= 0.0 || ratio >= 1.0) throw new IllegalArgumentException("Validation split must be between 0 and 1");
        this.validationSplit = ratio;
    }

    public void setInitialLearningRate(double rate) {
        this.initialLearningRate = rate;
    }

    public void setPeakLearningRate(double rate) {
        this.peakLearningRate = rate;
    }

    public void setWarmupEpochs(int epochs) {
        this.warmupEpochs = epochs;
    }
    
    private void initializeWeightsAndBiases() {
        int numLayers = layerSizes.length - 1;
        
        for (int layer = 0; layer < numLayers; layer++) {
            int inputNeurons = layerSizes[layer];
            int outputNeurons = layerSizes[layer + 1];
            
            weights[layer] = new double[inputNeurons][outputNeurons];
            biases[layer] = new double[outputNeurons];
            
            double limit = Math.sqrt(6.0 / (inputNeurons + outputNeurons));
            
            for (int i = 0; i < inputNeurons; i++) {
                for (int j = 0; j < outputNeurons; j++) {
                    weights[layer][i][j] = ThreadLocalRandom.current().nextDouble(-limit, limit);
                }
            }
            
            for (int j = 0; j < outputNeurons; j++) {
                biases[layer][j] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
            }
        }
    }
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    private double[][] forwardPass(double[] input) {
        int numLayers = layerSizes.length;
        double[][] layerOutputs = new double[numLayers][];
        
        layerOutputs[0] = input;
        
        for (int layer = 0; layer < numLayers - 1; layer++) {
            int currentLayerSize = layerSizes[layer];
            int nextLayerSize = layerSizes[layer + 1];
            layerOutputs[layer + 1] = new double[nextLayerSize];
            
            for (int j = 0; j < nextLayerSize; j++) {
                double sum = biases[layer][j];
                
                for (int i = 0; i < currentLayerSize; i++) {
                    sum += layerOutputs[layer][i] * weights[layer][i][j];
                }
                
                boolean isOutputLayer = (layer == numLayers - 2);
                
                if (!isOutputLayer) {
                    layerOutputs[layer + 1][j] = sigmoid(sum);
                    
                    if (isTraining && dropoutRate > 0) {
                        if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                            layerOutputs[layer + 1][j] = 0;
                        } else {
                            layerOutputs[layer + 1][j] /= (1.0 - dropoutRate);
                        }
                    }
                } else {
                    layerOutputs[layer + 1][j] = sum;
                }
            }
        }
        
        return layerOutputs;
    }
    
    private void saveModelState() {
        int numLayers = layerSizes.length - 1;
        bestWeights = new double[numLayers][][];
        bestBiases = new double[numLayers][];
        
        for (int layer = 0; layer < numLayers; layer++) {
            int inputNeurons = layerSizes[layer];
            int outputNeurons = layerSizes[layer + 1];
            
            bestWeights[layer] = new double[inputNeurons][outputNeurons];
            bestBiases[layer] = new double[outputNeurons];
            
            for (int i = 0; i < inputNeurons; i++) {
                for (int j = 0; j < outputNeurons; j++) {
                    bestWeights[layer][i][j] = weights[layer][i][j];
                }
            }
            
            for (int j = 0; j < outputNeurons; j++) {
                bestBiases[layer][j] = biases[layer][j];
            }
        }
    }
    
    private void restoreBestModel() {
        if (bestWeights == null) return;
        
        int numLayers = layerSizes.length - 1;
        
        for (int layer = 0; layer < numLayers; layer++) {
            int inputNeurons = layerSizes[layer];
            int outputNeurons = layerSizes[layer + 1];
            
            for (int i = 0; i < inputNeurons; i++) {
                for (int j = 0; j < outputNeurons; j++) {
                    weights[layer][i][j] = bestWeights[layer][i][j];
                }
            }
            
            for (int j = 0; j < outputNeurons; j++) {
                biases[layer][j] = bestBiases[layer][j];
            }
        }
    }
    
    private double evaluateError(List<Sample> samples) {
        double totalError = 0.0;
        
        for (Sample sample : samples) {
            double[][] outputs = forwardPass(sample.getInput());
            double[] finalOutputs = outputs[outputs.length - 1];

            for (int k = 0; k < outputSize; k++) {
                totalError += Math.pow(sample.getTarget()[k] - finalOutputs[k], 2); // MSE
            }
        }
        
        return totalError / (samples.size() * outputSize);
    }
    
    private Sample augmentSample(Sample sample) {
        double[] originalInput = sample.getInput();
        double[] augmentedInput = originalInput.clone();
        int pixelSize = 28;
        
        int transformCount = ThreadLocalRandom.current().nextInt(2, 4);
        for (int t = 0; t < transformCount; t++) {
            int transformType = ThreadLocalRandom.current().nextInt(5);
            
            switch (transformType) {
                case 0: augmentedInput = shiftImage(augmentedInput, pixelSize); break;
                case 1: augmentedInput = erasePatches(augmentedInput, pixelSize); break;
                case 2: augmentedInput = rotateImage(augmentedInput, pixelSize); break;
                case 3: augmentedInput = scaleImage(augmentedInput, pixelSize); break;
                case 4: augmentedInput = elasticDistortion(augmentedInput, pixelSize); break;
            }
        }
        
        augmentedInput = applyRandomNoise(augmentedInput);
        
        return new Sample(augmentedInput, sample.getTarget());
    }
    
    private double[] applyRandomNoise(double[] input) {
        for (int i = 0; i < input.length; i++) {
            input[i] += ThreadLocalRandom.current().nextDouble(-0.05, 0.05);
            input[i] = Math.min(1.0, Math.max(0.0, input[i]));
        }
        return input;
    }
    
    private double[] shiftImage(double[] input, int size) {
        double[] result = new double[input.length];
        int shiftX = ThreadLocalRandom.current().nextInt(-3, 4);
        int shiftY = ThreadLocalRandom.current().nextInt(-3, 4);
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int sourceX = x - shiftX;
                int sourceY = y - shiftY;
                
                if (sourceX >= 0 && sourceX < size && sourceY >= 0 && sourceY < size) {
                    result[y * size + x] = input[sourceY * size + sourceX];
                }
            }
        }
        
        return result;
    }
    
    private double[] erasePatches(double[] input, int size) {
        double[] result = input.clone();
        int numErasures = ThreadLocalRandom.current().nextInt(1, 4);
        
        for (int e = 0; e < numErasures; e++) {
            int eraseX = ThreadLocalRandom.current().nextInt(size);
            int eraseY = ThreadLocalRandom.current().nextInt(size);
            int eraseSize = ThreadLocalRandom.current().nextInt(1, 4);
            boolean eraseToWhite = ThreadLocalRandom.current().nextBoolean();
            
            for (int dy = -eraseSize; dy <= eraseSize; dy++) {
                for (int dx = -eraseSize; dx <= eraseSize; dx++) {
                    int y = eraseY + dy;
                    int x = eraseX + dx;
                    
                    if (y >= 0 && y < size && x >= 0 && x < size) {
                        result[y * size + x] = eraseToWhite ? 1.0 : 0.0;
                    }
                }
            }
        }
        
        return result;
    }
    
    private double[] rotateImage(double[] input, int size) {
        double[] result = new double[input.length];
        Arrays.fill(result, 0.0);
        
        double centerX = size / 2.0;
        double centerY = size / 2.0;
        double angle = ThreadLocalRandom.current().nextDouble(-0.25, 0.25);
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double xOrigin = x - centerX;
                double yOrigin = y - centerY;
                
                int srcX = (int)(xOrigin * Math.cos(angle) - yOrigin * Math.sin(angle) + centerX);
                int srcY = (int)(xOrigin * Math.sin(angle) + yOrigin * Math.cos(angle) + centerY);
                
                if (srcX >= 0 && srcX < size && srcY >= 0 && srcY < size) {
                    result[y * size + x] = input[srcY * size + srcX];
                }
            }
        }
        
        return result;
    }
    
    private double[] scaleImage(double[] input, int size) {
        double[] result = new double[input.length];
        Arrays.fill(result, 0.0);
        
        double centerX = size / 2.0;
        double centerY = size / 2.0;
        double scaleFactor = ThreadLocalRandom.current().nextDouble(0.8, 1.2);
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double srcX = (x - centerX) / scaleFactor + centerX;
                double srcY = (y - centerY) / scaleFactor + centerY;
                
                if (srcX >= 0 && srcX < size && srcY >= 0 && srcY < size) {
                    int x1 = (int)Math.floor(srcX);
                    int y1 = (int)Math.floor(srcY);
                    int x2 = Math.min(x1 + 1, size - 1);
                    int y2 = Math.min(y1 + 1, size - 1);
                    
                    double xWeight = srcX - x1;
                    double yWeight = srcY - y1;
                    
                    double val = input[y1 * size + x1] * (1 - xWeight) * (1 - yWeight) +
                                (x2 < size ? input[y1 * size + x2] : 0) * xWeight * (1 - yWeight) +
                                (y2 < size ? input[y2 * size + x1] : 0) * (1 - xWeight) * yWeight +
                                (x2 < size && y2 < size ? input[y2 * size + x2] : 0) * xWeight * yWeight;
                                
                    result[y * size + x] = val;
                }
            }
        }
        
        return result;
    }
    
    private double[] elasticDistortion(double[] input, int size) {
        double[] result = new double[input.length];
        double[][] displacementX = new double[size][size];
        double[][] displacementY = new double[size][size];
        
        int fieldSize = 7;
        double[][] smallFieldX = new double[fieldSize][fieldSize];
        double[][] smallFieldY = new double[fieldSize][fieldSize];
        double elasticScale = ThreadLocalRandom.current().nextDouble(3.0, 6.0);
        
        for (int i = 0; i < fieldSize; i++) {
            for (int j = 0; j < fieldSize; j++) {
                smallFieldX[i][j] = ThreadLocalRandom.current().nextDouble(-1, 1);
                smallFieldY[i][j] = ThreadLocalRandom.current().nextDouble(-1, 1);
            }
        }
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double fy = y * (fieldSize - 1.0) / size;
                double fx = x * (fieldSize - 1.0) / size;
                int y1 = (int)Math.floor(fy);
                int x1 = (int)Math.floor(fx);
                int y2 = Math.min(y1 + 1, fieldSize - 1);
                int x2 = Math.min(x1 + 1, fieldSize - 1);
                double yw = fy - y1;
                double xw = fx - x1;
                
                displacementX[y][x] = bilinearInterpolation(smallFieldX, x1, y1, x2, y2, xw, yw) * elasticScale;
                displacementY[y][x] = bilinearInterpolation(smallFieldY, x1, y1, x2, y2, xw, yw) * elasticScale;
            }
        }
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double srcX = Math.max(0, Math.min(size - 1, x + displacementX[y][x]));
                double srcY = Math.max(0, Math.min(size - 1, y + displacementY[y][x]));
                int sx = (int)Math.round(srcX);
                int sy = (int)Math.round(srcY);
                result[y * size + x] = input[sy * size + sx];
            }
        }
        
        return result;
    }
    
    private double bilinearInterpolation(double[][] field, int x1, int y1, int x2, int y2, double xw, double yw) {
        return field[y1][x1] * (1 - xw) * (1 - yw) +
               field[y1][x2] * xw * (1 - yw) +
               field[y2][x1] * (1 - xw) * yw +
               field[y2][x2] * xw * yw;
    }
    
    public void train(List<Sample> samples, int epochs) {
        if (samples.isEmpty()) {
            System.err.println("Brak danych do uczenia!");
            return;
        }

        learningRate = initialLearningRate;
        printTrainingConfiguration(samples.size(), epochs);
        
        List<Sample> trainingData = new ArrayList<>();
        List<Sample> validationData = new ArrayList<>();
        splitData(samples, trainingData, validationData);
        
        bestValidationError = Double.MAX_VALUE;
        epochsSinceImprovement = 0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            updateLearningRate(epoch);
            List<Sample> augmentedData = createAugmentedData(trainingData, epoch);
            double trainingError = trainEpoch(augmentedData) / (augmentedData.size() * outputSize);
            double validationError = evaluateError(validationData);

            System.out.printf("Epoka %d/%d, błąd (trening): %.6f, błąd (walidacja): %.6f%n", 
                             epoch + 1, epochs, trainingError, validationError);

            if (checkEarlyStopping(validationError, epoch)) {
                break;
            }
        }

        restoreBestModel();
        System.out.println("Uczenie zakończone! Najlepszy błąd walidacji: " + bestValidationError);
    }
    
    private void printTrainingConfiguration(int totalSamples, int epochs) {
        System.out.println("Rozpoczęcie uczenia sieci neuronowej...");
        System.out.println("Architektura: " + getArchitectureString());
        System.out.println("Liczba epok: " + epochs);
        System.out.println("Rozmiar zbioru uczącego: " + totalSamples);
        System.out.println("Learning rate: początkowy=" + initialLearningRate + ", maksymalny=" + peakLearningRate);
        System.out.println("Rozgrzewanie: " + warmupEpochs + " epok");
        System.out.println("Dropout rate: " + dropoutRate);
        System.out.println("Patience: " + patience + " epok");
    }
    
    private void splitData(List<Sample> samples, List<Sample> trainingData, List<Sample> validationData) {
        Collections.shuffle(samples);
        int validationSize = (int)(samples.size() * validationSplit);
        int trainingSize = samples.size() - validationSize;
        
        trainingData.addAll(samples.subList(0, trainingSize));
        validationData.addAll(samples.subList(trainingSize, samples.size()));
        
        System.out.println("Rozmiar zbioru treningowego: " + trainingData.size());
        System.out.println("Rozmiar zbioru walidacyjnego: " + validationData.size());
    }
    
    private void updateLearningRate(int epoch) {
        if (epoch < warmupEpochs) {
            learningRate = initialLearningRate + 
                          (peakLearningRate - initialLearningRate) * (epoch / (double)warmupEpochs);
            System.out.println("Rozgrzewanie: Learning rate zwiększony do: " + learningRate);
        } else if ((epoch - warmupEpochs) % 25 == 0 && epoch > warmupEpochs) {
            // More aggressive decay to help convergence
            learningRate *= 0.85;  // This is the 15% reduction (multiplying by 0.85)
            System.out.println("Learning rate zmniejszony do: " + learningRate);
        }
    }
    
    private List<Sample> createAugmentedData(List<Sample> trainingData, int epoch) {
        List<Sample> augmentedData = new ArrayList<>();
        
        for (Sample sample : trainingData) {
            augmentedData.add(sample);
            
            // Determine target class
            int classIndex = -1;
            double[] target = sample.getTarget();
            for (int i = 0; i < target.length; i++) {
                if (target[i] > 0.5) {
                    classIndex = i;
                    break;
                }
            }
            
            // Generate 10-15 augmentations per sample
            int numAugmentations = ThreadLocalRandom.current().nextInt(10, 16);
            
            for (int i = 0; i < numAugmentations; i++) {
                augmentedData.add(augmentSample(sample));
            }
        }

        if (epoch == 0) {
            System.out.println("Liczba próbek augmentowanych: " + augmentedData.size());
            System.out.println("Stosunek augmentacji: " + String.format("%.1f", 
                (double)augmentedData.size() / trainingData.size()) + "x");
        }
        
        Collections.shuffle(augmentedData);
        return augmentedData;
    }
    
    private double trainEpoch(List<Sample> augmentedData) {
        isTraining = true;
        double totalError = 0.0;
        
        for (Sample sample : augmentedData) {
            totalError += trainOnSample(sample);
        }
        
        isTraining = false;
        return totalError;
    }
    
    private double trainOnSample(Sample sample) {
        double[] input = sample.getInput();
        double[] target = sample.getTarget();
        double[][] layerOutputs = forwardPass(input);
        int numLayers = layerSizes.length;
        double[][] deltas = new double[numLayers - 1][];
        double totalError = 0.0;
        
        deltas[numLayers - 2] = new double[outputSize];
        for (int n = 0; n < outputSize; n++) {
            double error = target[n] - layerOutputs[numLayers - 1][n];
            totalError += Math.pow(error, 2);  // Tutaj jest używany MSE
            deltas[numLayers - 2][n] = error;
        }
        
        for (int layer = numLayers - 3; layer >= 0; layer--) {
            computeLayerDeltas(layer, layerOutputs, deltas);
        }
        
        for (int layer = 0; layer < numLayers - 1; layer++) {
            updateWeightsAndBiases(layer, layerOutputs, deltas);
        }
        
        return totalError;
    }
    
    private void computeLayerDeltas(int layer, double[][] layerOutputs, double[][] deltas) {
        int currentLayerSize = layerSizes[layer + 1];
        int nextLayerSize = layerSizes[layer + 2];
        
        deltas[layer] = new double[currentLayerSize];
        
        for (int j = 0; j < currentLayerSize; j++) {
            if (layerOutputs[layer + 1][j] == 0 && isTraining && dropoutRate > 0) {
                continue;
            }
            
            double error = 0.0;
            for (int k = 0; k < nextLayerSize; k++) {
                error += deltas[layer + 1][k] * weights[layer + 1][j][k];
            }
            
            double output = layerOutputs[layer + 1][j];
            deltas[layer][j] = error * output * (1 - output);
            
            if (isTraining && dropoutRate > 0) {
                deltas[layer][j] *= (1.0 - dropoutRate);
            }
        }
    }
    
    private void updateWeightsAndBiases(int layer, double[][] layerOutputs, double[][] deltas) {
        int fromSize = layerSizes[layer];
        int toSize = layerSizes[layer + 1];
        int numLayers = layerSizes.length;
        
        for (int to = 0; to < toSize; to++) {
            if (layer < numLayers - 2 && layerOutputs[layer + 1][to] == 0 && 
                isTraining && dropoutRate > 0) {
                continue;
            }
            
            biases[layer][to] += learningRate * deltas[layer][to];
            
            for (int from = 0; from < fromSize; from++) {
                weights[layer][from][to] += learningRate * deltas[layer][to] * layerOutputs[layer][from];
            }
        }
    }
    
    private boolean checkEarlyStopping(double validationError, int epoch) {
        if (validationError < bestValidationError) {
            bestValidationError = validationError;
            epochsSinceImprovement = 0;
            saveModelState();
            return false;
        } else {
            epochsSinceImprovement++;
            
            if (epochsSinceImprovement >= patience) {
                System.out.println("Wczesne zatrzymanie na epoce " + (epoch + 1) + 
                                  " (błąd walidacji nie poprawiał się przez " + patience + " epok)");
                return true;
            }
            
            return false;
        }
    }
    
    private String getArchitectureString() {
        return Arrays.stream(layerSizes)
               .mapToObj(String::valueOf)
               .reduce((a, b) -> a + " → " + b)
               .orElse("");
    }
    
    public double[] predict(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Nieprawidłowy rozmiar danych wejściowych: " + 
                                              input.length + " (oczekiwano " + inputSize + ")");
        }

        isTraining = false;
        double[][] outputs = forwardPass(input);
        return outputs[outputs.length - 1];
    }
    
    public void saveModel(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            writeModelToStream(oos);
        } catch (IOException e) {
            System.err.println("Błąd podczas zapisywania modelu: " + e.getMessage());
            throw e;
        }
    }
    
    private void writeModelToStream(ObjectOutputStream oos) throws IOException {
        oos.writeInt(inputSize);
        oos.writeInt(hidden0Size);
        oos.writeInt(hidden1Size);
        oos.writeInt(hidden2Size);
        oos.writeInt(hidden3Size);
        oos.writeInt(hidden4Size);
        oos.writeInt(outputSize);
        oos.writeDouble(learningRate);
        oos.writeDouble(dropoutRate);

        int numLayers = layerSizes.length - 1;
        for (int layer = 0; layer < numLayers; layer++) {
            oos.writeObject(weights[layer]);
            oos.writeObject(biases[layer]);
        }
    }
    
    public void loadModel(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            readModelFromStream(ois);
            System.out.println("Model został pomyślnie załadowany z pliku: " + path);
            System.out.println("Architektura: " + getArchitectureString());
        } catch (Exception e) {
            System.err.println("Błąd podczas ładowania modelu: " + e.getMessage());
            throw e;
        }
    }
    
    private void readModelFromStream(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        this.inputSize = ois.readInt();
        this.hidden0Size = ois.readInt();
        this.hidden1Size = ois.readInt();
        this.hidden2Size = ois.readInt();
        this.hidden3Size = ois.readInt();
        this.hidden4Size = ois.readInt();
        this.outputSize = ois.readInt();
        this.learningRate = ois.readDouble();
        this.dropoutRate = ois.readDouble();
        
        this.layerSizes = new int[]{inputSize, hidden0Size, hidden1Size, hidden2Size, hidden3Size, hidden4Size, outputSize};
        
        int numLayers = layerSizes.length - 1;
        this.weights = new double[numLayers][][];
        this.biases = new double[numLayers][];

        for (int layer = 0; layer < numLayers; layer++) {
            weights[layer] = (double[][]) ois.readObject();
            biases[layer] = (double[]) ois.readObject();
        }
    }
}
