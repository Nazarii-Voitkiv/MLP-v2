import java.io.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class NeuralNetwork {
    private int inputSize;
    private int hidden0Size; // First hidden layer (256 neurons)
    private int hidden1Size; // Second hidden layer (128 neurons)
    private int hidden2Size; // Third hidden layer (64 neurons)
    private int outputSize;
    
    // Weights and biases for deeper architecture
    private double[][] weightsInputHidden0;    // 784x256
    private double[] biasesHidden0;            // 256
    private double[][] weightsHidden0Hidden1;  // 256x128
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
    private double[][] bestWeightsInputHidden0;
    private double[] bestBiasesHidden0;
    private double[][] bestWeightsHidden0Hidden1;
    private double[] bestBiasesHidden1;
    private double[][] bestWeightsHidden1Hidden2;
    private double[] bestBiasesHidden2;
    private double[][] bestWeightsHidden2Output;
    private double[] bestBiasesOutput;
    
    /**
     * Creates a neural network with specified layer sizes for deeper architecture
     */
    public NeuralNetwork(int inputSize, int hidden0Size, int hidden1Size, int hidden2Size, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hidden0Size = hidden0Size;
        this.hidden1Size = hidden1Size;
        this.hidden2Size = hidden2Size;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        
        // Initialize weights and biases randomly
        initializeWeightsAndBiases();
    }
    
    /**
     * Creates a neural network with default architecture (784-256-128-64-3) and learning rate
     */
    public NeuralNetwork() {
        this(784, 256, 128, 64, 3, 0.005);
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
        weightsInputHidden0 = new double[inputSize][hidden0Size];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hidden0Size; j++) {
                // Xavier/Glorot initialization
                double limit = Math.sqrt(6.0 / (inputSize + hidden0Size));
                weightsInputHidden0[i][j] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }
        
        // Initialize biases for first hidden layer
        biasesHidden0 = new double[hidden0Size];
        for (int j = 0; j < hidden0Size; j++) {
            biasesHidden0[j] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }
        
        // Initialize weights between first and second hidden layer
        weightsHidden0Hidden1 = new double[hidden0Size][hidden1Size];
        for (int j = 0; j < hidden0Size; j++) {
            for (int k = 0; k < hidden1Size; k++) {
                double limit = Math.sqrt(6.0 / (hidden0Size + hidden1Size));
                weightsHidden0Hidden1[j][k] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }
        
        // Initialize biases for second hidden layer
        biasesHidden1 = new double[hidden1Size];
        for (int k = 0; k < hidden1Size; k++) {
            biasesHidden1[k] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }
        
        // Initialize weights between second and third hidden layer
        weightsHidden1Hidden2 = new double[hidden1Size][hidden2Size];
        for (int k = 0; k < hidden1Size; k++) {
            for (int l = 0; l < hidden2Size; l++) {
                double limit = Math.sqrt(6.0 / (hidden1Size + hidden2Size));
                weightsHidden1Hidden2[k][l] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }
        
        // Initialize biases for third hidden layer
        biasesHidden2 = new double[hidden2Size];
        for (int l = 0; l < hidden2Size; l++) {
            biasesHidden2[l] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }
        
        // Initialize weights between third hidden layer and output
        weightsHidden2Output = new double[hidden2Size][outputSize];
        for (int l = 0; l < hidden2Size; l++) {
            for (int m = 0; m < outputSize; m++) {
                double limit = Math.sqrt(6.0 / (hidden2Size + outputSize));
                weightsHidden2Output[l][m] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }
        
        // Initialize biases for output layer
        biasesOutput = new double[outputSize];
        for (int m = 0; m < outputSize; m++) {
            biasesOutput[m] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
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
     * @return Output from each layer [hidden0Outputs, hidden1Outputs, hidden2Outputs, finalOutputs]
     */
    private double[][] forwardPass(double[] input) {
        // Calculate first hidden layer outputs (hidden0)
        double[] hidden0Inputs = new double[hidden0Size];
        double[] hidden0Outputs = new double[hidden0Size];
        
        for (int j = 0; j < hidden0Size; j++) {
            hidden0Inputs[j] = biasesHidden0[j];
            for (int i = 0; i < inputSize; i++) {
                hidden0Inputs[j] += input[i] * weightsInputHidden0[i][j];
            }
            hidden0Outputs[j] = sigmoid(hidden0Inputs[j]);
            
            // Apply dropout during training
            if (isTraining && dropoutRate > 0) {
                if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                    hidden0Outputs[j] = 0; // Drop this neuron
                } else {
                    hidden0Outputs[j] /= (1.0 - dropoutRate); // Scale to maintain expected values
                }
            }
        }
        
        // Calculate second hidden layer outputs (hidden1)
        double[] hidden1Inputs = new double[hidden1Size];
        double[] hidden1Outputs = new double[hidden1Size];
        
        for (int k = 0; k < hidden1Size; k++) {
            hidden1Inputs[k] = biasesHidden1[k];
            for (int j = 0; j < hidden0Size; j++) {
                hidden1Inputs[k] += hidden0Outputs[j] * weightsHidden0Hidden1[j][k];
            }
            hidden1Outputs[k] = sigmoid(hidden1Inputs[k]);
            
            // Apply dropout during training
            if (isTraining && dropoutRate > 0) {
                if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                    hidden1Outputs[k] = 0; // Drop this neuron
                } else {
                    hidden1Outputs[k] /= (1.0 - dropoutRate); // Scale to maintain expected values
                }
            }
        }
        
        // Calculate third hidden layer outputs (hidden2)
        double[] hidden2Inputs = new double[hidden2Size];
        double[] hidden2Outputs = new double[hidden2Size];
        
        for (int l = 0; l < hidden2Size; l++) {
            hidden2Inputs[l] = biasesHidden2[l];
            for (int k = 0; k < hidden1Size; k++) {
                hidden2Inputs[l] += hidden1Outputs[k] * weightsHidden1Hidden2[k][l];
            }
            hidden2Outputs[l] = sigmoid(hidden2Inputs[l]);
            
            // Apply dropout during training
            if (isTraining && dropoutRate > 0) {
                if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                    hidden2Outputs[l] = 0; // Drop this neuron
                } else {
                    hidden2Outputs[l] /= (1.0 - dropoutRate); // Scale to maintain expected values
                }
            }
        }
        
        // Calculate output layer inputs (raw logits)
        double[] outputInputs = new double[outputSize];
        for (int m = 0; m < outputSize; m++) {
            outputInputs[m] = biasesOutput[m];
            for (int l = 0; l < hidden2Size; l++) {
                outputInputs[m] += hidden2Outputs[l] * weightsHidden2Output[l][m];
            }
        }
        
        // Return raw outputs without softmax activation
        return new double[][] { hidden0Outputs, hidden1Outputs, hidden2Outputs, outputInputs };
    }
    
    /**
     * Creates a deep copy of the current model weights and biases
     */
    private void saveModelState() {
        bestWeightsInputHidden0 = new double[inputSize][hidden0Size];
        bestBiasesHidden0 = new double[hidden0Size];
        bestWeightsHidden0Hidden1 = new double[hidden0Size][hidden1Size];
        bestBiasesHidden1 = new double[hidden1Size];
        bestWeightsHidden1Hidden2 = new double[hidden1Size][hidden2Size];
        bestBiasesHidden2 = new double[hidden2Size];
        bestWeightsHidden2Output = new double[hidden2Size][outputSize];
        bestBiasesOutput = new double[outputSize];
        
        // Copy weights and biases for input to hidden0 layer
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hidden0Size; j++) {
                bestWeightsInputHidden0[i][j] = weightsInputHidden0[i][j];
            }
        }
        
        // Copy biases for hidden0 and weights for hidden0 to hidden1
        for (int j = 0; j < hidden0Size; j++) {
            bestBiasesHidden0[j] = biasesHidden0[j];
            for (int k = 0; k < hidden1Size; k++) {
                bestWeightsHidden0Hidden1[j][k] = weightsHidden0Hidden1[j][k];
            }
        }
        
        // Copy biases for hidden1 and weights for hidden1 to hidden2
        for (int k = 0; k < hidden1Size; k++) {
            bestBiasesHidden1[k] = biasesHidden1[k];
            for (int l = 0; l < hidden2Size; l++) {
                bestWeightsHidden1Hidden2[k][l] = weightsHidden1Hidden2[k][l];
            }
        }
        
        // Copy biases for hidden2 and weights for hidden2 to output
        for (int l = 0; l < hidden2Size; l++) {
            bestBiasesHidden2[l] = biasesHidden2[l];
            for (int m = 0; m < outputSize; m++) {
                bestWeightsHidden2Output[l][m] = weightsHidden2Output[l][m];
            }
        }
        
        // Copy output biases
        for (int m = 0; m < outputSize; m++) {
            bestBiasesOutput[m] = biasesOutput[m];
        }
    }
    
    /**
     * Restores the best model weights and biases
     */
    private void restoreBestModel() {
        if (bestWeightsInputHidden0 == null) return;
        
        // Restore weights and biases for input to hidden0 layer
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hidden0Size; j++) {
                weightsInputHidden0[i][j] = bestWeightsInputHidden0[i][j];
            }
        }
        
        // Restore biases for hidden0 and weights for hidden0 to hidden1
        for (int j = 0; j < hidden0Size; j++) {
            biasesHidden0[j] = bestBiasesHidden0[j];
            for (int k = 0; k < hidden1Size; k++) {
                weightsHidden0Hidden1[j][k] = bestWeightsHidden0Hidden1[j][k];
            }
        }
        
        // Restore biases for hidden1 and weights for hidden1 to hidden2
        for (int k = 0; k < hidden1Size; k++) {
            biasesHidden1[k] = bestBiasesHidden1[k];
            for (int l = 0; l < hidden2Size; l++) {
                weightsHidden1Hidden2[k][l] = bestWeightsHidden1Hidden2[k][l];
            }
        }
        
        // Restore biases for hidden2 and weights for hidden2 to output
        for (int l = 0; l < hidden2Size; l++) {
            biasesHidden2[l] = bestBiasesHidden2[l];
            for (int m = 0; m < outputSize; m++) {
                weightsHidden2Output[l][m] = bestWeightsHidden2Output[l][m];
            }
        }
        
        // Restore output biases
        for (int m = 0; m < outputSize; m++) {
            biasesOutput[m] = bestBiasesOutput[m];
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
            double[] finalOutputs = outputs[3]; // Get raw outputs (index changed to 3 because we now have 4 layers of outputs)
            
            // Calculate squared error
            for (int k = 0; k < outputSize; k++) {
                totalError += Math.pow(target[k] - finalOutputs[k], 2);
            }
        }
        
        return totalError / (samples.size() * outputSize);
    }
    
    /**
     * Enhanced data augmentation method with more robust transformations
     */
    private Sample augmentSample(Sample sample) {
        double[] originalInput = sample.getInput();
        double[] augmentedInput = new double[originalInput.length];
        
        // Apply random noise to each pixel (slightly increased range)
        for (int i = 0; i < originalInput.length; i++) {
            augmentedInput[i] = originalInput[i] + ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
            augmentedInput[i] = Math.min(1.0, Math.max(0.0, augmentedInput[i])); // Clamp to [0, 1]
        }
        
        // Apply multiple transformations with high probability (90%)
        if (ThreadLocalRandom.current().nextDouble() < 0.9) {
            int pixelSize = 28; // Assuming 28x28 images
            double centerX = pixelSize / 2.0; // Define centerX here for all transformations
            double centerY = pixelSize / 2.0; // Define centerY here for all transformations
            
            // Select 1-3 random transformations to apply
            int numTransformations = ThreadLocalRandom.current().nextInt(1, 4);
            
            for (int t = 0; t < numTransformations; t++) {
                // Select a transformation type
                int transformType = ThreadLocalRandom.current().nextInt(5);
                
                switch (transformType) {
                    case 0: // Shift image (enhanced with diagonal shifts)
                        int shiftX = ThreadLocalRandom.current().nextInt(-2, 3); // -2 to 2 pixels
                        int shiftY = ThreadLocalRandom.current().nextInt(-2, 3); // -2 to 2 pixels
                        double[] shifted = new double[augmentedInput.length];
                        
                        for (int y = 0; y < pixelSize; y++) {
                            for (int x = 0; x < pixelSize; x++) {
                                int sourceX = x - shiftX;
                                int sourceY = y - shiftY;
                                
                                if (sourceX >= 0 && sourceX < pixelSize && 
                                    sourceY >= 0 && sourceY < pixelSize) {
                                    shifted[y * pixelSize + x] = augmentedInput[sourceY * pixelSize + sourceX];
                                } else {
                                    shifted[y * pixelSize + x] = 0.0; // Black background
                                }
                            }
                        }
                        
                        augmentedInput = shifted;
                        break;
                        
                    case 1: // Small random erasing (simulates noise/occlusion)
                        int numErasures = ThreadLocalRandom.current().nextInt(1, 4); // 1-3 erasures
                        
                        for (int e = 0; e < numErasures; e++) {
                            int eraseX = ThreadLocalRandom.current().nextInt(pixelSize);
                            int eraseY = ThreadLocalRandom.current().nextInt(pixelSize);
                            int eraseSize = ThreadLocalRandom.current().nextInt(1, 4); // 1-3 pixel radius
                            boolean eraseToWhite = ThreadLocalRandom.current().nextBoolean();
                            
                            for (int dy = -eraseSize; dy <= eraseSize; dy++) {
                                for (int dx = -eraseSize; dx <= eraseSize; dx++) {
                                    int y = eraseY + dy;
                                    int x = eraseX + dx;
                                    
                                    if (y >= 0 && y < pixelSize && x >= 0 && x < pixelSize) {
                                        augmentedInput[y * pixelSize + x] = eraseToWhite ? 1.0 : 0.0;
                                    }
                                }
                            }
                        }
                        break;
                        
                    case 2: // Improved rotation (more accurate approximation)
                        double[] rotated = new double[augmentedInput.length];
                        Arrays.fill(rotated, 0.0); // Initialize with zeros
                        
                        double angle = ThreadLocalRandom.current().nextDouble(-0.2, 0.2); // rotation in radians
                        
                        for (int y = 0; y < pixelSize; y++) {
                            for (int x = 0; x < pixelSize; x++) {
                                // Rotate around center
                                double xOrigin = x - centerX;
                                double yOrigin = y - centerY;
                                
                                int srcX = (int)(xOrigin * Math.cos(angle) - yOrigin * Math.sin(angle) + centerX);
                                int srcY = (int)(xOrigin * Math.sin(angle) + yOrigin * Math.cos(angle) + centerY);
                                
                                if (srcX >= 0 && srcX < pixelSize && srcY >= 0 && srcY < pixelSize) {
                                    rotated[y * pixelSize + x] = augmentedInput[srcY * pixelSize + srcX];
                                }
                            }
                        }
                        
                        augmentedInput = rotated;
                        break;
                        
                    case 3: // Scaling (zoom in or out)
                        double[] scaled = new double[augmentedInput.length];
                        Arrays.fill(scaled, 0.0);
                        
                        double scaleFactor = ThreadLocalRandom.current().nextDouble(0.8, 1.2); // 80% to 120%
                        
                        for (int y = 0; y < pixelSize; y++) {
                            for (int x = 0; x < pixelSize; x++) {
                                // Scale around center
                                double srcX = (x - centerX) / scaleFactor + centerX;
                                double srcY = (y - centerY) / scaleFactor + centerY;
                                
                                // Bilinear interpolation
                                int x1 = (int)Math.floor(srcX);
                                int y1 = (int)Math.floor(srcY);
                                int x2 = Math.min(x1 + 1, pixelSize - 1);
                                int y2 = Math.min(y1 + 1, pixelSize - 1);
                                
                                if (x1 >= 0 && x1 < pixelSize && y1 >= 0 && y1 < pixelSize) {
                                    double xWeight = srcX - x1;
                                    double yWeight = srcY - y1;
                                    
                                    double val = augmentedInput[y1 * pixelSize + x1] * (1 - xWeight) * (1 - yWeight) +
                                                (x2 < pixelSize ? augmentedInput[y1 * pixelSize + x2] : 0) * xWeight * (1 - yWeight) +
                                                (y2 < pixelSize ? augmentedInput[y2 * pixelSize + x1] : 0) * (1 - xWeight) * yWeight +
                                                (x2 < pixelSize && y2 < pixelSize ? augmentedInput[y2 * pixelSize + x2] : 0) * xWeight * yWeight;
                                    
                                    scaled[y * pixelSize + x] = val;
                                }
                            }
                        }
                        
                        augmentedInput = scaled;
                        break;
                        
                    case 4: // Elastic distortion (simulates natural variations in handwriting)
                        double[] distorted = new double[augmentedInput.length];
                        double[][] displacementX = new double[pixelSize][pixelSize];
                        double[][] displacementY = new double[pixelSize][pixelSize];
                        
                        // Generate random displacement fields
                        double elasticScale = ThreadLocalRandom.current().nextDouble(3.0, 6.0);
                        
                        // Generate smoother fields by starting with a smaller grid
                        int fieldSize = 7;
                        double[][] smallFieldX = new double[fieldSize][fieldSize];
                        double[][] smallFieldY = new double[fieldSize][fieldSize];
                        
                        for (int i = 0; i < fieldSize; i++) {
                            for (int j = 0; j < fieldSize; j++) {
                                smallFieldX[i][j] = ThreadLocalRandom.current().nextDouble(-1, 1);
                                smallFieldY[i][j] = ThreadLocalRandom.current().nextDouble(-1, 1);
                            }
                        }
                        
                        // Interpolate to full size
                        for (int y = 0; y < pixelSize; y++) {
                            for (int x = 0; x < pixelSize; x++) {
                                double fy = y * (fieldSize - 1.0) / pixelSize;
                                double fx = x * (fieldSize - 1.0) / pixelSize;
                                int y1 = (int)Math.floor(fy);
                                int x1 = (int)Math.floor(fx);
                                int y2 = Math.min(y1 + 1, fieldSize - 1);
                                int x2 = Math.min(x1 + 1, fieldSize - 1);
                                double yw = fy - y1;
                                double xw = fx - x1;
                                
                                displacementX[y][x] = smallFieldX[y1][x1] * (1 - xw) * (1 - yw) +
                                                    smallFieldX[y1][x2] * xw * (1 - yw) +
                                                    smallFieldX[y2][x1] * (1 - xw) * yw +
                                                    smallFieldX[y2][x2] * xw * yw;
                                                    
                                displacementY[y][x] = smallFieldY[y1][x1] * (1 - xw) * (1 - yw) +
                                                    smallFieldY[y1][x2] * xw * (1 - yw) +
                                                    smallFieldY[y2][x1] * (1 - xw) * yw +
                                                    smallFieldY[y2][x2] * xw * yw;
                                                    
                                displacementX[y][x] *= elasticScale;
                                displacementY[y][x] *= elasticScale;
                            }
                        }
                        
                        // Apply displacement field
                        for (int y = 0; y < pixelSize; y++) {
                            for (int x = 0; x < pixelSize; x++) {
                                double srcX = x + displacementX[y][x];
                                double srcY = y + displacementY[y][x];
                                
                                // Clamp to image boundaries
                                srcX = Math.max(0, Math.min(pixelSize - 1, srcX));
                                srcY = Math.max(0, Math.min(pixelSize - 1, srcY));
                                
                                // Simple nearest neighbor sampling
                                int sx = (int)Math.round(srcX);
                                int sy = (int)Math.round(srcY);
                                
                                distorted[y * pixelSize + x] = augmentedInput[sy * pixelSize + sx];
                            }
                        }
                        
                        augmentedInput = distorted;
                        break;
                }
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
        System.out.println("Архітектура: " + inputSize + " → " + hidden0Size + " → " + 
                           hidden1Size + " → " + hidden2Size + " → " + outputSize);
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
                
                // Add 3-5 augmented copies of each sample with varying transformations
                int numAugmentations = ThreadLocalRandom.current().nextInt(3, 6);
                for (int i = 0; i < numAugmentations; i++) {
                    augmentedData.add(augmentSample(s));
                }
            }
            
            // Print augmentation info only on first epoch
            if (epoch == 0) {
                System.out.println("Аугментованих прикладів: " + augmentedData.size());
                System.out.println("Співвідношення аугментації: " + String.format("%.1f", 
                    (double)augmentedData.size() / trainingData.size()) + "x");
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
                double[] hidden0Outputs = outputs[0];
                double[] hidden1Outputs = outputs[1];
                double[] hidden2Outputs = outputs[2];
                double[] finalOutputs = outputs[3];
                
                // Calculate output errors
                double[] outputErrors = new double[outputSize];
                for (int m = 0; m < outputSize; m++) {
                    outputErrors[m] = target[m] - finalOutputs[m];
                    totalTrainingError += Math.pow(outputErrors[m], 2);
                }
                
                // Calculate output layer gradient
                double[] outputDeltas = new double[outputSize];
                for (int m = 0; m < outputSize; m++) {
                    outputDeltas[m] = outputErrors[m]; // Simplified gradient for linear output
                }
                
                // Calculate hidden2 layer errors
                double[] hidden2Errors = new double[hidden2Size];
                for (int l = 0; l < hidden2Size; l++) {
                    // Skip if this neuron was dropped out
                    if (hidden2Outputs[l] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    double error = 0.0;
                    for (int m = 0; m < outputSize; m++) {
                        error += outputDeltas[m] * weightsHidden2Output[l][m];
                    }
                    hidden2Errors[l] = error;
                }
                
                // Calculate hidden2 layer delta
                double[] hidden2Deltas = new double[hidden2Size];
                for (int l = 0; l < hidden2Size; l++) {
                    // Skip if this neuron was dropped out
                    if (hidden2Outputs[l] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    hidden2Deltas[l] = hidden2Errors[l] * hidden2Outputs[l] * (1 - hidden2Outputs[l]);
                    if (isTraining && dropoutRate > 0) {
                        hidden2Deltas[l] *= (1.0 - dropoutRate); // Scale back the delta
                    }
                }
                
                // Calculate hidden1 layer errors
                double[] hidden1Errors = new double[hidden1Size];
                for (int k = 0; k < hidden1Size; k++) {
                    // Skip if this neuron was dropped out
                    if (hidden1Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    double error = 0.0;
                    for (int l = 0; l < hidden2Size; l++) {
                        error += hidden2Deltas[l] * weightsHidden1Hidden2[k][l];
                    }
                    hidden1Errors[k] = error;
                }
                
                // Calculate hidden1 layer delta
                double[] hidden1Deltas = new double[hidden1Size];
                for (int k = 0; k < hidden1Size; k++) {
                    // Skip if this neuron was dropped out
                    if (hidden1Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    hidden1Deltas[k] = hidden1Errors[k] * hidden1Outputs[k] * (1 - hidden1Outputs[k]);
                    if (isTraining && dropoutRate > 0) {
                        hidden1Deltas[k] *= (1.0 - dropoutRate); // Scale back the delta
                    }
                }
                
                // Calculate hidden0 layer errors
                double[] hidden0Errors = new double[hidden0Size];
                for (int j = 0; j < hidden0Size; j++) {
                    // Skip if this neuron was dropped out
                    if (hidden0Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    double error = 0.0;
                    for (int k = 0; k < hidden1Size; k++) {
                        error += hidden1Deltas[k] * weightsHidden0Hidden1[j][k];
                    }
                    hidden0Errors[j] = error;
                }
                
                // Calculate hidden0 layer delta
                double[] hidden0Deltas = new double[hidden0Size];
                for (int j = 0; j < hidden0Size; j++) {
                    // Skip if this neuron was dropped out
                    if (hidden0Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    hidden0Deltas[j] = hidden0Errors[j] * hidden0Outputs[j] * (1 - hidden0Outputs[j]);
                    if (isTraining && dropoutRate > 0) {
                        hidden0Deltas[j] *= (1.0 - dropoutRate); // Scale back the delta
                    }
                }
                
                // Update weights and biases for output layer
                for (int l = 0; l < hidden2Size; l++) {
                    // Skip if this neuron was dropped out
                    if (hidden2Outputs[l] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    for (int m = 0; m < outputSize; m++) {
                        weightsHidden2Output[l][m] += learningRate * outputDeltas[m] * hidden2Outputs[l];
                    }
                }
                
                for (int m = 0; m < outputSize; m++) {
                    biasesOutput[m] += learningRate * outputDeltas[m];
                }
                
                // Update weights and biases for hidden2 layer
                for (int k = 0; k < hidden1Size; k++) {
                    // Skip if this neuron was dropped out
                    if (hidden1Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    for (int l = 0; l < hidden2Size; l++) {
                        // Skip if this neuron was dropped out
                        if (hidden2Outputs[l] == 0 && isTraining && dropoutRate > 0) continue;
                        
                        weightsHidden1Hidden2[k][l] += learningRate * hidden2Deltas[l] * hidden1Outputs[k];
                    }
                }
                
                for (int l = 0; l < hidden2Size; l++) {
                    // Skip if this neuron was dropped out
                    if (hidden2Outputs[l] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    biasesHidden2[l] += learningRate * hidden2Deltas[l];
                }
                
                // Update weights and biases for hidden1 layer
                for (int j = 0; j < hidden0Size; j++) {
                    // Skip if this neuron was dropped out
                    if (hidden0Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    for (int k = 0; k < hidden1Size; k++) {
                        // Skip if this neuron was dropped out
                        if (hidden1Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                        
                        weightsHidden0Hidden1[j][k] += learningRate * hidden1Deltas[k] * hidden0Outputs[j];
                    }
                }
                
                for (int k = 0; k < hidden1Size; k++) {
                    // Skip if this neuron was dropped out
                    if (hidden1Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    biasesHidden1[k] += learningRate * hidden1Deltas[k];
                }
                
                // Update weights and biases for hidden0 layer
                for (int i = 0; i < inputSize; i++) {
                    for (int j = 0; j < hidden0Size; j++) {
                        // Skip if this neuron was dropped out
                        if (hidden0Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                        
                        weightsInputHidden0[i][j] += learningRate * hidden0Deltas[j] * input[i];
                    }
                }
                
                for (int j = 0; j < hidden0Size; j++) {
                    // Skip if this neuron was dropped out
                    if (hidden0Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    biasesHidden0[j] += learningRate * hidden0Deltas[j];
                }
            }
            
            // Switch to evaluation mode (no dropout)
            isTraining = false;
            
            // Evaluate on validation set
            double trainingError = totalTrainingError / (augmentedData.size() * outputSize);
            double validationError = evaluateError(validationData);
            
            // Print progress for every epoch (removed the condition)
            System.out.printf("Епоха %d/%d, помилка (тренування): %.6f, помилка (валідація): %.6f%n", 
                             epoch + 1, epochs, trainingError, validationError);
            
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
        return outputs[3]; // Return the final outputs (raw logits)
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
            oos.writeInt(hidden0Size);
            oos.writeInt(hidden1Size);
            oos.writeInt(hidden2Size);
            oos.writeInt(outputSize);
            oos.writeDouble(learningRate);
            oos.writeDouble(dropoutRate);
            
            // Save weights and biases
            oos.writeObject(weightsInputHidden0);
            oos.writeObject(biasesHidden0);
            oos.writeObject(weightsHidden0Hidden1);
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
            
            // Check if this is a compatible format
            try {
                this.hidden0Size = ois.readInt();
                this.hidden1Size = ois.readInt();
                this.hidden2Size = ois.readInt();
                this.outputSize = ois.readInt();
                this.learningRate = ois.readDouble();
                this.dropoutRate = ois.readDouble();
                
                // Load weights and biases for deeper architecture
                this.weightsInputHidden0 = (double[][]) ois.readObject();
                this.biasesHidden0 = (double[]) ois.readObject();
                this.weightsHidden0Hidden1 = (double[][]) ois.readObject();
                this.biasesHidden1 = (double[]) ois.readObject();
                this.weightsHidden1Hidden2 = (double[][]) ois.readObject();
                this.biasesHidden2 = (double[]) ois.readObject();
                this.weightsHidden2Output = (double[][]) ois.readObject();
                this.biasesOutput = (double[]) ois.readObject();
                
                System.out.println("Модель успішно завантажено з файлу: " + path);
                System.out.println("Архітектура: " + inputSize + " → " + hidden0Size + " → " + 
                                  hidden1Size + " → " + hidden2Size + " → " + outputSize);
            } catch (Exception e) {
                System.err.println("Формат файлу моделі не відповідає очікуваному: " + e.getMessage());
                throw e;
            }
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Помилка при завантаженні моделі: " + e.getMessage());
            throw e;
        }
    }
}
