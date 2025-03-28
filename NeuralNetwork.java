import java.io.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class NeuralNetwork {
    private int inputSize;
    private int hidden1Size;
    private int hidden2Size;
    private int outputSize;
    
    // Weights and biases for deeper architecture
    private double[][] weightsInputHidden1;    // 784x128
    private double[] biasesHidden1;            // 128
    private double[][] weightsHidden1Hidden2;  // 128x64
    private double[] biasesHidden2;            // 64
    private double[][] weightsHidden2Output;   // 64x3
    private double[] biasesOutput;             // 3
    
    // Learning rate
    private double learningRate;
    
    // Dropout rate
    private double dropoutRate = 0.1;
    private boolean isTraining = false;
    
    // Early stopping parameters
    private int patience = 10;
    private double bestValidationError = Double.MAX_VALUE;
    private int epochsSinceImprovement = 0;
    private double validationSplit = 0.2; // 20% for validation
    
    // Best model parameters for early stopping
    private double[][] bestWeightsInputHidden1;
    private double[] bestBiasesHidden1;
    private double[][] bestWeightsHidden1Hidden2;
    private double[] bestBiasesHidden2;
    private double[][] bestWeightsHidden2Output;
    private double[] bestBiasesOutput;
    
    /**
     * Creates a neural network with specified layer sizes for deeper architecture
     */
    public NeuralNetwork(int inputSize, int hidden1Size, int hidden2Size, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hidden1Size = hidden1Size;
        this.hidden2Size = hidden2Size;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        
        // Initialize weights and biases randomly
        initializeWeightsAndBiases();
    }
    
    /**
     * Creates a neural network with default architecture (784-128-64-3) and learning rate
     */
    public NeuralNetwork() {
        this(784, 128, 64, 3, 0.05);
    }
    
    // Setter for dropout rate
    public void setDropoutRate(double rate) {
        if (rate < 0.0 || rate >= 1.0) {
            throw new IllegalArgumentException("Dropout rate must be between 0 and 1");
        }
        this.dropoutRate = rate;
    }
    
    // Setter for early stopping patience
    public void setPatience(int patience) {
        this.patience = patience;
    }
    
    // Setter for validation split ratio
    public void setValidationSplit(double ratio) {
        if (ratio <= 0.0 || ratio >= 1.0) {
            throw new IllegalArgumentException("Validation split must be between 0 and 1");
        }
        this.validationSplit = ratio;
    }
    
    /**
     * Initializes weights and biases with random values for deeper architecture
     */
    private void initializeWeightsAndBiases() {
        // Initialize weights between input and first hidden layer
        weightsInputHidden1 = new double[inputSize][hidden1Size];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hidden1Size; j++) {
                // Xavier/Glorot initialization
                double limit = Math.sqrt(6.0 / (inputSize + hidden1Size));
                weightsInputHidden1[i][j] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }
        
        // Initialize biases for first hidden layer
        biasesHidden1 = new double[hidden1Size];
        for (int j = 0; j < hidden1Size; j++) {
            biasesHidden1[j] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }
        
        // Initialize weights between first and second hidden layer
        weightsHidden1Hidden2 = new double[hidden1Size][hidden2Size];
        for (int j = 0; j < hidden1Size; j++) {
            for (int k = 0; k < hidden2Size; k++) {
                double limit = Math.sqrt(6.0 / (hidden1Size + hidden2Size));
                weightsHidden1Hidden2[j][k] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }
        
        // Initialize biases for second hidden layer
        biasesHidden2 = new double[hidden2Size];
        for (int k = 0; k < hidden2Size; k++) {
            biasesHidden2[k] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }
        
        // Initialize weights between second hidden layer and output
        weightsHidden2Output = new double[hidden2Size][outputSize];
        for (int k = 0; k < hidden2Size; k++) {
            for (int l = 0; l < outputSize; l++) {
                double limit = Math.sqrt(6.0 / (hidden2Size + outputSize));
                weightsHidden2Output[k][l] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }
        
        // Initialize biases for output layer
        biasesOutput = new double[outputSize];
        for (int l = 0; l < outputSize; l++) {
            biasesOutput[l] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }
    }
    
    /**
     * Sigmoid activation function
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Forward pass through the network with deeper architecture
     * 
     * @param input Input values
     * @return Output from each layer [hidden1Outputs, hidden2Outputs, finalOutputs]
     */
    private double[][] forwardPass(double[] input) {
        // Calculate first hidden layer outputs
        double[] hidden1Inputs = new double[hidden1Size];
        double[] hidden1Outputs = new double[hidden1Size];
        
        for (int j = 0; j < hidden1Size; j++) {
            hidden1Inputs[j] = biasesHidden1[j];
            for (int i = 0; i < inputSize; i++) {
                hidden1Inputs[j] += input[i] * weightsInputHidden1[i][j];
            }
            hidden1Outputs[j] = sigmoid(hidden1Inputs[j]);
            
            // Apply dropout to first hidden layer during training
            if (isTraining && dropoutRate > 0) {
                if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                    hidden1Outputs[j] = 0; // Drop this neuron
                } else {
                    hidden1Outputs[j] /= (1.0 - dropoutRate); // Scale to maintain expected values
                }
            }
        }
        
        // Calculate second hidden layer outputs
        double[] hidden2Inputs = new double[hidden2Size];
        double[] hidden2Outputs = new double[hidden2Size];
        
        for (int k = 0; k < hidden2Size; k++) {
            hidden2Inputs[k] = biasesHidden2[k];
            for (int j = 0; j < hidden1Size; j++) {
                hidden2Inputs[k] += hidden1Outputs[j] * weightsHidden1Hidden2[j][k];
            }
            hidden2Outputs[k] = sigmoid(hidden2Inputs[k]);
            
            // Apply dropout to second hidden layer during training
            if (isTraining && dropoutRate > 0) {
                if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                    hidden2Outputs[k] = 0; // Drop this neuron
                } else {
                    hidden2Outputs[k] /= (1.0 - dropoutRate); // Scale to maintain expected values
                }
            }
        }
        
        // Calculate output layer inputs (raw logits)
        double[] outputInputs = new double[outputSize];
        for (int l = 0; l < outputSize; l++) {
            outputInputs[l] = biasesOutput[l];
            for (int k = 0; k < hidden2Size; k++) {
                outputInputs[l] += hidden2Outputs[k] * weightsHidden2Output[k][l];
            }
        }
        
        // Return raw outputs without softmax activation
        return new double[][] { hidden1Outputs, hidden2Outputs, outputInputs };
    }
    
    /**
     * Creates a deep copy of the current model weights and biases
     */
    private void saveModelState() {
        bestWeightsInputHidden1 = new double[inputSize][hidden1Size];
        bestBiasesHidden1 = new double[hidden1Size];
        bestWeightsHidden1Hidden2 = new double[hidden1Size][hidden2Size];
        bestBiasesHidden2 = new double[hidden2Size];
        bestWeightsHidden2Output = new double[hidden2Size][outputSize];
        bestBiasesOutput = new double[outputSize];
        
        // Copy weights and biases for input to hidden1 layer
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hidden1Size; j++) {
                bestWeightsInputHidden1[i][j] = weightsInputHidden1[i][j];
            }
        }
        
        // Copy biases for hidden1 and weights for hidden1 to hidden2
        for (int j = 0; j < hidden1Size; j++) {
            bestBiasesHidden1[j] = biasesHidden1[j];
            for (int k = 0; k < hidden2Size; k++) {
                bestWeightsHidden1Hidden2[j][k] = weightsHidden1Hidden2[j][k];
            }
        }
        
        // Copy biases for hidden2 and weights for hidden2 to output
        for (int k = 0; k < hidden2Size; k++) {
            bestBiasesHidden2[k] = biasesHidden2[k];
            for (int l = 0; l < outputSize; l++) {
                bestWeightsHidden2Output[k][l] = weightsHidden2Output[k][l];
            }
        }
        
        // Copy output biases
        for (int l = 0; l < outputSize; l++) {
            bestBiasesOutput[l] = biasesOutput[l];
        }
    }
    
    /**
     * Restores the best model weights and biases
     */
    private void restoreBestModel() {
        if (bestWeightsInputHidden1 == null) return;
        
        // Restore weights and biases for input to hidden1 layer
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hidden1Size; j++) {
                weightsInputHidden1[i][j] = bestWeightsInputHidden1[i][j];
            }
        }
        
        // Restore biases for hidden1 and weights for hidden1 to hidden2
        for (int j = 0; j < hidden1Size; j++) {
            biasesHidden1[j] = bestBiasesHidden1[j];
            for (int k = 0; k < hidden2Size; k++) {
                weightsHidden1Hidden2[j][k] = bestWeightsHidden1Hidden2[j][k];
            }
        }
        
        // Restore biases for hidden2 and weights for hidden2 to output
        for (int k = 0; k < hidden2Size; k++) {
            biasesHidden2[k] = bestBiasesHidden2[k];
            for (int l = 0; l < outputSize; l++) {
                weightsHidden2Output[k][l] = bestWeightsHidden2Output[k][l];
            }
        }
        
        // Restore output biases
        for (int l = 0; l < outputSize; l++) {
            biasesOutput[l] = bestBiasesOutput[l];
        }
    }
    
    /**
     * Evaluates the model on a set of samples
     */
    private double evaluateError(List<Sample> samples) {
        double totalError = 0.0;
        
        for (Sample sample : samples) {
            double[] input = sample.getInput();
            double[] target = sample.getTarget();
            
            double[][] outputs = forwardPass(input);
            double[] finalOutputs = outputs[2]; // Get raw outputs
            
            // Calculate squared error
            for (int k = 0; k < outputSize; k++) {
                totalError += Math.pow(target[k] - finalOutputs[k], 2);
            }
        }
        
        return totalError / (samples.size() * outputSize);
    }
    
    /**
     * Enhanced data augmentation method with more variations
     */
    private Sample augmentSample(Sample sample) {
        double[] originalInput = sample.getInput();
        double[] augmentedInput = new double[originalInput.length];
        
        // Apply random noise to each pixel (increased range from -0.05 to 0.05)
        for (int i = 0; i < originalInput.length; i++) {
            augmentedInput[i] = originalInput[i] + ThreadLocalRandom.current().nextDouble(-0.08, 0.08);
            augmentedInput[i] = Math.min(1.0, Math.max(0.0, augmentedInput[i])); // Clamp to [0, 1]
        }
        
        // Apply random transformation with higher probability (80%)
        if (ThreadLocalRandom.current().nextDouble() < 0.8) {
            int pixelSize = 28; // Assuming 28x28 images
            
            // Select a random transformation:
            int transformType = ThreadLocalRandom.current().nextInt(3);
            
            switch (transformType) {
                case 0: // Shift image
                    int shiftDirection = ThreadLocalRandom.current().nextInt(4); // 0: left, 1: right, 2: up, 3: down
                    int shiftAmount = ThreadLocalRandom.current().nextInt(1, 3); // 1 or 2 pixels
                    double[] shifted = new double[augmentedInput.length];
                    
                    for (int y = 0; y < pixelSize; y++) {
                        for (int x = 0; x < pixelSize; x++) {
                            int sourceX = x;
                            int sourceY = y;
                            
                            switch (shiftDirection) {
                                case 0: // left
                                    sourceX = Math.min(x + shiftAmount, pixelSize - 1);
                                    break;
                                case 1: // right
                                    sourceX = Math.max(x - shiftAmount, 0);
                                    break;
                                case 2: // up
                                    sourceY = Math.min(y + shiftAmount, pixelSize - 1);
                                    break;
                                case 3: // down
                                    sourceY = Math.max(y - shiftAmount, 0);
                                    break;
                            }
                            
                            shifted[y * pixelSize + x] = augmentedInput[sourceY * pixelSize + sourceX];
                        }
                    }
                    
                    augmentedInput = shifted;
                    break;
                    
                case 1: // Small random erasing (simulates noise/occlusion)
                    int eraseX = ThreadLocalRandom.current().nextInt(pixelSize);
                    int eraseY = ThreadLocalRandom.current().nextInt(pixelSize);
                    int eraseSize = ThreadLocalRandom.current().nextInt(1, 3);
                    
                    for (int dy = -eraseSize; dy <= eraseSize; dy++) {
                        for (int dx = -eraseSize; dx <= eraseSize; dx++) {
                            int y = eraseY + dy;
                            int x = eraseX + dx;
                            
                            if (y >= 0 && y < pixelSize && x >= 0 && x < pixelSize) {
                                // Randomly set to 0 or 1 (50/50 chance)
                                augmentedInput[y * pixelSize + x] = ThreadLocalRandom.current().nextBoolean() ? 1.0 : 0.0;
                            }
                        }
                    }
                    break;
                    
                case 2: // Slight rotation effect (approximated by shearing)
                    double[] rotated = new double[augmentedInput.length];
                    Arrays.fill(rotated, 0.0); // Initialize with zeros
                    
                    // Simple shear transform to approximate small rotation
                    double shearFactor = ThreadLocalRandom.current().nextDouble(-0.2, 0.2);
                    
                    for (int y = 0; y < pixelSize; y++) {
                        for (int x = 0; x < pixelSize; x++) {
                            int newX = (int)(x + shearFactor * (y - pixelSize/2));
                            
                            if (newX >= 0 && newX < pixelSize) {
                                rotated[y * pixelSize + newX] = augmentedInput[y * pixelSize + x];
                            }
                        }
                    }
                    
                    augmentedInput = rotated;
                    break;
            }
        }
        
        return new Sample(augmentedInput, sample.getTarget());
    }
    
    /**
     * Trains the neural network on a set of samples with enhanced augmentation
     * 
     * @param data List of training samples
     * @param epochs Number of training epochs
     */
    public void train(List<Sample> samples, int epochs) {
        int totalSamples = samples.size();
        if (totalSamples == 0) {
            System.err.println("Немає даних для навчання!");
            return;
        }
        
        System.out.println("Початок навчання нейромережі...");
        System.out.println("Архітектура: " + inputSize + " → " + hidden1Size + " → " + 
                           hidden2Size + " → " + outputSize);
        System.out.println("Кількість епох: " + epochs);
        System.out.println("Розмір навчальної вибірки: " + totalSamples);
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Dropout rate: " + dropoutRate);
        
        // Split data into training and validation sets
        Collections.shuffle(samples);
        int validationSize = (int)(totalSamples * validationSplit);
        int trainingSize = totalSamples - validationSize;
        
        List<Sample> trainingData = new ArrayList<>(samples.subList(0, trainingSize));
        List<Sample> validationData = new ArrayList<>(samples.subList(trainingSize, totalSamples));
        
        System.out.println("Розмір тренувальної вибірки: " + trainingData.size());
        System.out.println("Розмір валідаційної вибірки: " + validationData.size());
        
        // Reset early stopping parameters
        bestValidationError = Double.MAX_VALUE;
        epochsSinceImprovement = 0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Create augmented dataset for this epoch with enhanced augmentation
            List<Sample> augmentedData = new ArrayList<>();
            for (Sample s : trainingData) {
                augmentedData.add(s); // Add original sample
                
                // Add 2-3 augmented copies of each sample
                augmentedData.add(augmentSample(s));
                augmentedData.add(augmentSample(s));
                
                // Add a third augmented sample with 50% probability
                if (ThreadLocalRandom.current().nextBoolean()) {
                    augmentedData.add(augmentSample(s));
                }
            }
            
            // Print augmentation info only on first epoch
            if (epoch == 0) {
                System.out.println("Аугментованих прикладів: " + augmentedData.size());
            }
            
            double totalTrainingError = 0.0;
            
            // Set to training mode
            isTraining = true;
            
            // Shuffle the training data for each epoch
            Collections.shuffle(augmentedData);
            
            for (Sample sample : augmentedData) {
                double[] input = sample.getInput();
                double[] target = sample.getTarget();
                
                // Forward pass
                double[][] outputs = forwardPass(input);
                double[] hidden1Outputs = outputs[0];
                double[] hidden2Outputs = outputs[1];
                double[] finalOutputs = outputs[2];
                
                // Calculate error
                double[] outputErrors = new double[outputSize];
                for (int l = 0; l < outputSize; l++) {
                    outputErrors[l] = target[l] - finalOutputs[l];
                    totalTrainingError += Math.pow(outputErrors[l], 2);
                }
                
                // Calculate output layer gradient
                double[] outputDeltas = new double[outputSize];
                for (int l = 0; l < outputSize; l++) {
                    outputDeltas[l] = outputErrors[l]; // Simplified gradient for linear output
                }
                
                // Calculate hidden2 layer errors
                double[] hidden2Errors = new double[hidden2Size];
                for (int k = 0; k < hidden2Size; k++) {
                    // Skip if this neuron was dropped out
                    if (hidden2Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    double error = 0.0;
                    for (int l = 0; l < outputSize; l++) {
                        error += outputDeltas[l] * weightsHidden2Output[k][l];
                    }
                    hidden2Errors[k] = error;
                }
                
                // Calculate hidden2 layer delta
                double[] hidden2Deltas = new double[hidden2Size];
                for (int k = 0; k < hidden2Size; k++) {
                    // Skip if this neuron was dropped out
                    if (hidden2Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    hidden2Deltas[k] = hidden2Errors[k] * hidden2Outputs[k] * (1 - hidden2Outputs[k]);
                    if (isTraining && dropoutRate > 0) {
                        hidden2Deltas[k] *= (1.0 - dropoutRate); // Scale back the delta
                    }
                }
                
                // Calculate hidden1 layer errors
                double[] hidden1Errors = new double[hidden1Size];
                for (int j = 0; j < hidden1Size; j++) {
                    // Skip if this neuron was dropped out
                    if (hidden1Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    double error = 0.0;
                    for (int k = 0; k < hidden2Size; k++) {
                        error += hidden2Deltas[k] * weightsHidden1Hidden2[j][k];
                    }
                    hidden1Errors[j] = error;
                }
                
                // Calculate hidden1 layer delta
                double[] hidden1Deltas = new double[hidden1Size];
                for (int j = 0; j < hidden1Size; j++) {
                    // Skip if this neuron was dropped out
                    if (hidden1Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    hidden1Deltas[j] = hidden1Errors[j] * hidden1Outputs[j] * (1 - hidden1Outputs[j]);
                    if (isTraining && dropoutRate > 0) {
                        hidden1Deltas[j] *= (1.0 - dropoutRate); // Scale back the delta
                    }
                }
                
                // Update weights and biases for output layer
                for (int k = 0; k < hidden2Size; k++) {
                    // Skip if this neuron was dropped out
                    if (hidden2Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    for (int l = 0; l < outputSize; l++) {
                        weightsHidden2Output[k][l] += learningRate * outputDeltas[l] * hidden2Outputs[k];
                    }
                }
                
                for (int l = 0; l < outputSize; l++) {
                    biasesOutput[l] += learningRate * outputDeltas[l];
                }
                
                // Update weights and biases for hidden2 layer
                for (int j = 0; j < hidden1Size; j++) {
                    // Skip if this neuron was dropped out
                    if (hidden1Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    for (int k = 0; k < hidden2Size; k++) {
                        // Skip if this neuron was dropped out
                        if (hidden2Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                        
                        weightsHidden1Hidden2[j][k] += learningRate * hidden2Deltas[k] * hidden1Outputs[j];
                    }
                }
                
                for (int k = 0; k < hidden2Size; k++) {
                    // Skip if this neuron was dropped out
                    if (hidden2Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    biasesHidden2[k] += learningRate * hidden2Deltas[k];
                }
                
                // Update weights and biases for hidden1 layer
                for (int i = 0; i < inputSize; i++) {
                    for (int j = 0; j < hidden1Size; j++) {
                        // Skip if this neuron was dropped out
                        if (hidden1Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                        
                        weightsInputHidden1[i][j] += learningRate * hidden1Deltas[j] * input[i];
                    }
                }
                
                for (int j = 0; j < hidden1Size; j++) {
                    // Skip if this neuron was dropped out
                    if (hidden1Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    biasesHidden1[j] += learningRate * hidden1Deltas[j];
                }
            }
            
            // Switch to evaluation mode (no dropout)
            isTraining = false;
            
            // Evaluate on validation set
            double trainingError = totalTrainingError / (augmentedData.size() * outputSize);
            double validationError = evaluateError(validationData);
            
            // Print progress every 10 epochs or for the last epoch
            if (epoch % 5 == 0 || epoch == epochs - 1) {
                System.out.printf("Епоха %d/%d, помилка (тренування): %.6f, помилка (валідація): %.6f%n", 
                                 epoch + 1, epochs, trainingError, validationError);
            }
            
            // Early stopping check
            if (validationError < bestValidationError) {
                bestValidationError = validationError;
                epochsSinceImprovement = 0;
                saveModelState(); // Save the best model
            } else {
                epochsSinceImprovement++;
            }
            
            if (epochsSinceImprovement >= patience) {
                System.out.println("Early stopping на епосі " + (epoch + 1) + 
                                  " (валідаційна помилка не покращувалась " + patience + " епох)");
                break;
            }
        }
        
        // Restore the best model
        restoreBestModel();
        System.out.println("Навчання завершено! Найкраща валідаційна помилка: " + bestValidationError);
    }
    
    /**
     * Predicts the output for a given input
     * 
     * @param input Input data (784 values)
     * @return Array of logits (raw, unnormalized outputs) for each letter [M, O, N]
     */
    public double[] predict(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Неправильний розмір вхідних даних: " + input.length + 
                                              " (очікувалося " + inputSize + ")");
        }
        
        // Set to evaluation mode (no dropout)
        isTraining = false;
        
        double[][] outputs = forwardPass(input);
        return outputs[2]; // Return the final outputs (raw logits)
    }
    
    /**
     * Saves the model weights and biases to a file
     * 
     * @param path File path to save the model
     */
    public void saveModel(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            // Save network architecture
            oos.writeInt(inputSize);
            oos.writeInt(hidden1Size);
            oos.writeInt(hidden2Size);
            oos.writeInt(outputSize);
            oos.writeDouble(learningRate);
            oos.writeDouble(dropoutRate);
            
            // Save weights and biases
            oos.writeObject(weightsInputHidden1);
            oos.writeObject(biasesHidden1);
            oos.writeObject(weightsHidden1Hidden2);
            oos.writeObject(biasesHidden2);
            oos.writeObject(weightsHidden2Output);
            oos.writeObject(biasesOutput);
            
            System.out.println("Модель успішно збережено у файл: " + path);
        } catch (IOException e) {
            System.err.println("Помилка при збереженні моделі: " + e.getMessage());
            throw e;
        }
    }
    
    /**
     * Loads the model weights and biases from a file
     * 
     * @param path File path to load the model from
     */
    public void loadModel(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            // Load network architecture
            this.inputSize = ois.readInt();
            
            // Check if this is a new format with two hidden layers
            if (ois.available() > 0) {
                try {
                    this.hidden1Size = ois.readInt();
                    this.hidden2Size = ois.readInt();
                    this.outputSize = ois.readInt();
                    this.learningRate = ois.readDouble();
                    this.dropoutRate = ois.readDouble();
                    
                    // Load weights and biases for deeper architecture
                    this.weightsInputHidden1 = (double[][]) ois.readObject();
                    this.biasesHidden1 = (double[]) ois.readObject();
                    this.weightsHidden1Hidden2 = (double[][]) ois.readObject();
                    this.biasesHidden2 = (double[]) ois.readObject();
                    this.weightsHidden2Output = (double[][]) ois.readObject();
                    this.biasesOutput = (double[]) ois.readObject();
                    
                    System.out.println("Модель успішно завантажено з файлу: " + path);
                    System.out.println("Архітектура: " + inputSize + " → " + hidden1Size + " → " + 
                                      hidden2Size + " → " + outputSize);
                } catch (Exception e) {
                    System.err.println("Формат файлу моделі не відповідає очікуваному: " + e.getMessage());
                    throw e;
                }
            } else {
                // Legacy format (single hidden layer)
                System.err.println("Завантаження моделі зі старим форматом (один прихований шар) не підтримується");
                throw new IOException("Incompatible model format");
            }
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Помилка при завантаженні моделі: " + e.getMessage());
            throw e;
        }
    }
}
