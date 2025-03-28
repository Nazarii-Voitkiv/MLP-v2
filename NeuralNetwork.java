import java.io.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class NeuralNetwork {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    
    // Weights and biases
    private double[][] weightsInputHidden;  // 784x64
    private double[] biasesHidden;          // 64
    private double[][] weightsHiddenOutput; // 64x3
    private double[] biasesOutput;          // 3
    
    // Learning rate
    private double learningRate;
    
    // Dropout rate (0 = no dropout, 0.5 = drop half of neurons)
    private double dropoutRate = 0.2;
    private boolean isTraining = false;
    
    // Early stopping parameters
    private int patience = 10;
    private double bestValidationError = Double.MAX_VALUE;
    private int epochsSinceImprovement = 0;
    private double validationSplit = 0.2; // 20% for validation
    
    // Best model parameters for early stopping
    private double[][] bestWeightsInputHidden;
    private double[] bestBiasesHidden;
    private double[][] bestWeightsHiddenOutput;
    private double[] bestBiasesOutput;
    
    /**
     * Creates a neural network with specified layer sizes
     */
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        
        // Initialize weights and biases randomly
        initializeWeightsAndBiases();
    }
    
    /**
     * Creates a neural network with default architecture (784-64-3) and learning rate
     */
    public NeuralNetwork() {
        this(784, 64, 3, 0.1);
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
     * Initializes weights and biases with random values
     */
    private void initializeWeightsAndBiases() {
        // Initialize weights between input and hidden layer
        weightsInputHidden = new double[inputSize][hiddenSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                // Xavier/Glorot initialization for better convergence
                double limit = Math.sqrt(6.0 / (inputSize + hiddenSize));
                weightsInputHidden[i][j] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }
        
        // Initialize biases for hidden layer
        biasesHidden = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            biasesHidden[j] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }
        
        // Initialize weights between hidden and output layer
        weightsHiddenOutput = new double[hiddenSize][outputSize];
        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                double limit = Math.sqrt(6.0 / (hiddenSize + outputSize));
                weightsHiddenOutput[j][k] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }
        
        // Initialize biases for output layer
        biasesOutput = new double[outputSize];
        for (int k = 0; k < outputSize; k++) {
            biasesOutput[k] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }
    }
    
    /**
     * Sigmoid activation function
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Applies softmax activation to an array of values
     */
    private double[] softmax(double[] values) {
        // Find maximum value to avoid numerical overflow
        double max = Arrays.stream(values).max().orElse(0);
        
        // Calculate exponentials
        double[] exps = new double[values.length];
        double sum = 0.0;
        for (int i = 0; i < values.length; i++) {
            exps[i] = Math.exp(values[i] - max);
            sum += exps[i];
        }
        
        // Normalize to get probabilities
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = exps[i] / sum;
        }
        
        return result;
    }
    
    /**
     * Forward pass through the network
     * 
     * @param input Input values
     * @return Output from each layer [hiddenOutputs, finalOutputs]
     */
    private double[][] forwardPass(double[] input) {
        // Calculate hidden layer outputs
        double[] hiddenInputs = new double[hiddenSize];
        double[] hiddenOutputs = new double[hiddenSize];
        
        for (int j = 0; j < hiddenSize; j++) {
            hiddenInputs[j] = biasesHidden[j];
            for (int i = 0; i < inputSize; i++) {
                hiddenInputs[j] += input[i] * weightsInputHidden[i][j];
            }
            hiddenOutputs[j] = sigmoid(hiddenInputs[j]);
            
            // Apply dropout during training
            if (isTraining && dropoutRate > 0) {
                if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                    hiddenOutputs[j] = 0; // Drop this neuron
                } else {
                    hiddenOutputs[j] /= (1.0 - dropoutRate); // Scale to maintain expected values
                }
            }
        }
        
        // Calculate output layer inputs
        double[] outputInputs = new double[outputSize];
        for (int k = 0; k < outputSize; k++) {
            outputInputs[k] = biasesOutput[k];
            for (int j = 0; j < hiddenSize; j++) {
                outputInputs[k] += hiddenOutputs[j] * weightsHiddenOutput[j][k];
            }
        }
        
        // Apply softmax activation to output layer
        double[] finalOutputs = softmax(outputInputs);
        
        return new double[][] { hiddenOutputs, finalOutputs };
    }
    
    /**
     * Creates a deep copy of the current model weights and biases
     */
    private void saveModelState() {
        bestWeightsInputHidden = new double[inputSize][hiddenSize];
        bestBiasesHidden = new double[hiddenSize];
        bestWeightsHiddenOutput = new double[hiddenSize][outputSize];
        bestBiasesOutput = new double[outputSize];
        
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                bestWeightsInputHidden[i][j] = weightsInputHidden[i][j];
            }
        }
        
        for (int j = 0; j < hiddenSize; j++) {
            bestBiasesHidden[j] = biasesHidden[j];
            for (int k = 0; k < outputSize; k++) {
                bestWeightsHiddenOutput[j][k] = weightsHiddenOutput[j][k];
            }
        }
        
        for (int k = 0; k < outputSize; k++) {
            bestBiasesOutput[k] = biasesOutput[k];
        }
    }
    
    /**
     * Restores the best model weights and biases
     */
    private void restoreBestModel() {
        if (bestWeightsInputHidden == null) return;
        
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = bestWeightsInputHidden[i][j];
            }
        }
        
        for (int j = 0; j < hiddenSize; j++) {
            biasesHidden[j] = bestBiasesHidden[j];
            for (int k = 0; k < outputSize; k++) {
                weightsHiddenOutput[j][k] = bestWeightsHiddenOutput[j][k];
            }
        }
        
        for (int k = 0; k < outputSize; k++) {
            biasesOutput[k] = bestBiasesOutput[k];
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
            double[] finalOutputs = outputs[1];
            
            // Calculate squared error
            for (int k = 0; k < outputSize; k++) {
                totalError += Math.pow(target[k] - finalOutputs[k], 2);
            }
        }
        
        return totalError / (samples.size() * outputSize);
    }
    
    /**
     * Trains the neural network on a set of samples
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
        System.out.println("Архітектура: " + inputSize + " → " + hiddenSize + " → " + outputSize);
        System.out.println("Кількість епох: " + epochs);
        System.out.println("Розмір навчальної вибірки: " + totalSamples);
        System.out.println("Dropout rate: " + dropoutRate);
        
        // Split data into training and validation sets
        Collections.shuffle(samples);
        int validationSize = (int)(totalSamples * validationSplit);
        int trainingSize = totalSamples - validationSize;
        
        List<Sample> trainingData = samples.subList(0, trainingSize);
        List<Sample> validationData = samples.subList(trainingSize, totalSamples);
        
        System.out.println("Розмір тренувальної вибірки: " + trainingData.size());
        System.out.println("Розмір валідаційної вибірки: " + validationData.size());
        
        // Reset early stopping parameters
        bestValidationError = Double.MAX_VALUE;
        epochsSinceImprovement = 0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalTrainingError = 0.0;
            
            // Set to training mode
            isTraining = true;
            
            // Shuffle the training data for each epoch
            Collections.shuffle(trainingData);
            
            for (Sample sample : trainingData) {
                double[] input = sample.getInput();
                double[] target = sample.getTarget();
                
                // Forward pass
                double[][] outputs = forwardPass(input);
                double[] hiddenOutputs = outputs[0];
                double[] finalOutputs = outputs[1];
                
                // Calculate error
                double[] outputErrors = new double[outputSize];
                for (int k = 0; k < outputSize; k++) {
                    outputErrors[k] = target[k] - finalOutputs[k];
                    totalTrainingError += Math.pow(outputErrors[k], 2);
                }
                
                // Calculate output layer gradient (for softmax + cross-entropy this simplifies)
                double[] outputDeltas = new double[outputSize];
                for (int k = 0; k < outputSize; k++) {
                    outputDeltas[k] = outputErrors[k]; // Simplified gradient for softmax
                }
                
                // Calculate hidden layer errors
                double[] hiddenErrors = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    // Skip if this neuron was dropped out
                    if (hiddenOutputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    double error = 0.0;
                    for (int k = 0; k < outputSize; k++) {
                        error += outputDeltas[k] * weightsHiddenOutput[j][k];
                    }
                    hiddenErrors[j] = error;
                }
                
                // Calculate hidden layer delta
                double[] hiddenDeltas = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    // Skip if this neuron was dropped out
                    if (hiddenOutputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    hiddenDeltas[j] = hiddenErrors[j] * hiddenOutputs[j] * (1 - hiddenOutputs[j]);
                    if (isTraining && dropoutRate > 0) {
                        hiddenDeltas[j] *= (1.0 - dropoutRate); // Scale back the delta
                    }
                }
                
                // Update weights and biases for output layer
                for (int j = 0; j < hiddenSize; j++) {
                    // Skip if this neuron was dropped out
                    if (hiddenOutputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    for (int k = 0; k < outputSize; k++) {
                        weightsHiddenOutput[j][k] += learningRate * outputDeltas[k] * hiddenOutputs[j];
                    }
                }
                
                for (int k = 0; k < outputSize; k++) {
                    biasesOutput[k] += learningRate * outputDeltas[k];
                }
                
                // Update weights and biases for hidden layer
                for (int i = 0; i < inputSize; i++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        // Skip if this neuron was dropped out
                        if (hiddenOutputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                        
                        weightsInputHidden[i][j] += learningRate * hiddenDeltas[j] * input[i];
                    }
                }
                
                for (int j = 0; j < hiddenSize; j++) {
                    // Skip if this neuron was dropped out
                    if (hiddenOutputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    biasesHidden[j] += learningRate * hiddenDeltas[j];
                }
            }
            
            // Switch to evaluation mode (no dropout)
            isTraining = false;
            
            // Evaluate on validation set
            double trainingError = totalTrainingError / (trainingData.size() * outputSize);
            double validationError = evaluateError(validationData);
            
            // Print progress every 10 epochs or for the last epoch
            if (epoch % 10 == 0 || epoch == epochs - 1) {
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
     * @return Array of probabilities for each letter [M, O, N]
     */
    public double[] predict(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Неправильний розмір вхідних даних: " + input.length + 
                                              " (очікувалося " + inputSize + ")");
        }
        
        // Set to evaluation mode (no dropout)
        isTraining = false;
        
        double[][] outputs = forwardPass(input);
        return outputs[1]; // Return the final outputs
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
            oos.writeInt(hiddenSize);
            oos.writeInt(outputSize);
            oos.writeDouble(learningRate);
            oos.writeDouble(dropoutRate);
            
            // Save weights and biases
            oos.writeObject(weightsInputHidden);
            oos.writeObject(biasesHidden);
            oos.writeObject(weightsHiddenOutput);
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
            this.hiddenSize = ois.readInt();
            this.outputSize = ois.readInt();
            this.learningRate = ois.readDouble();
            
            try {
                this.dropoutRate = ois.readDouble(); // For backwards compatibility
            } catch (Exception e) {
                this.dropoutRate = 0.0;
                System.out.println("Завантажена модель не містить параметр dropout rate, використовується 0.0");
            }
            
            // Load weights and biases
            this.weightsInputHidden = (double[][]) ois.readObject();
            this.biasesHidden = (double[]) ois.readObject();
            this.weightsHiddenOutput = (double[][]) ois.readObject();
            this.biasesOutput = (double[]) ois.readObject();
            
            System.out.println("Модель успішно завантажено з файлу: " + path);
            System.out.println("Архітектура: " + inputSize + " → " + hiddenSize + " → " + outputSize);
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Помилка при завантаженні моделі: " + e.getMessage());
            throw e;
        }
    }
}
