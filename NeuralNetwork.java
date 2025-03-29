import java.io.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class NeuralNetwork {
    private int inputSize;
    private int hidden0Size;
    private int hidden1Size;
    private int hidden2Size;
    private int hidden3Size;
    private int outputSize;
    
    private double[][] weightsInputHidden0;  
    private double[] biasesHidden0;
    private double[][] weightsHidden0Hidden1;
    private double[] biasesHidden1;
    private double[][] weightsHidden1Hidden2;
    private double[] biasesHidden2;
    private double[][] weightsHidden2Hidden3;
    private double[] biasesHidden3;
    private double[][] weightsHidden3Output;
    private double[] biasesOutput;

    private double learningRate;

    private double dropoutRate = 0.0;
    private boolean isTraining = false;

    private int patience = 10;
    private double bestValidationError = Double.MAX_VALUE;
    private int epochsSinceImprovement = 0;
    private double validationSplit = 0.2;

    private double[][] bestWeightsInputHidden0;
    private double[] bestBiasesHidden0;
    private double[][] bestWeightsHidden0Hidden1;
    private double[] bestBiasesHidden1;
    private double[][] bestWeightsHidden1Hidden2;
    private double[] bestBiasesHidden2;
    private double[][] bestWeightsHidden2Hidden3;
    private double[] bestBiasesHidden3;
    private double[][] bestWeightsHidden3Output;
    private double[] bestBiasesOutput;

    public NeuralNetwork(int inputSize, int hidden0Size, int hidden1Size, int hidden2Size, int hidden3Size, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hidden0Size = hidden0Size;
        this.hidden1Size = hidden1Size;
        this.hidden2Size = hidden2Size;
        this.hidden3Size = hidden3Size;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        initializeWeightsAndBiases();
    }
    
    public NeuralNetwork() {
        this(784, 512, 256, 128, 64, 3, 0.01);
    }

    public void setDropoutRate(double rate) {
        if (rate < 0.0 || rate >= 1.0) {
            throw new IllegalArgumentException("Dropout rate must be between 0 and 1");
        }
        this.dropoutRate = rate;
    }

    public void setPatience(int patience) {
        this.patience = patience;
    }

    public void setValidationSplit(double ratio) {
        if (ratio <= 0.0 || ratio >= 1.0) {
            throw new IllegalArgumentException("Validation split must be between 0 and 1");
        }
        this.validationSplit = ratio;
    }

    private void initializeWeightsAndBiases() {
        weightsInputHidden0 = new double[inputSize][hidden0Size];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hidden0Size; j++) {
                double limit = Math.sqrt(6.0 / (inputSize + hidden0Size));
                weightsInputHidden0[i][j] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }

        biasesHidden0 = new double[hidden0Size];
        for (int j = 0; j < hidden0Size; j++) {
            biasesHidden0[j] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }

        weightsHidden0Hidden1 = new double[hidden0Size][hidden1Size];
        for (int j = 0; j < hidden0Size; j++) {
            for (int k = 0; k < hidden1Size; k++) {
                double limit = Math.sqrt(6.0 / (hidden0Size + hidden1Size));
                weightsHidden0Hidden1[j][k] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }

        biasesHidden1 = new double[hidden1Size];
        for (int k = 0; k < hidden1Size; k++) {
            biasesHidden1[k] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }

        weightsHidden1Hidden2 = new double[hidden1Size][hidden2Size];
        for (int k = 0; k < hidden1Size; k++) {
            for (int l = 0; l < hidden2Size; l++) {
                double limit = Math.sqrt(6.0 / (hidden1Size + hidden2Size));
                weightsHidden1Hidden2[k][l] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }

        biasesHidden2 = new double[hidden2Size];
        for (int l = 0; l < hidden2Size; l++) {
            biasesHidden2[l] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }

        weightsHidden2Hidden3 = new double[hidden2Size][hidden3Size];
        for (int l = 0; l < hidden2Size; l++) {
            for (int m = 0; m < hidden3Size; m++) {
                double limit = Math.sqrt(6.0 / (hidden2Size + hidden3Size));
                weightsHidden2Hidden3[l][m] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }

        biasesHidden3 = new double[hidden3Size];
        for (int m = 0; m < hidden3Size; m++) {
            biasesHidden3[m] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }

        weightsHidden3Output = new double[hidden3Size][outputSize];
        for (int m = 0; m < hidden3Size; m++) {
            for (int n = 0; n < outputSize; n++) {
                double limit = Math.sqrt(6.0 / (hidden3Size + outputSize));
                weightsHidden3Output[m][n] = ThreadLocalRandom.current().nextDouble(-limit, limit);
            }
        }

        biasesOutput = new double[outputSize];
        for (int n = 0; n < outputSize; n++) {
            biasesOutput[n] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
        }
    }
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double[][] forwardPass(double[] input) {
        double[] hidden0Inputs = new double[hidden0Size];
        double[] hidden0Outputs = new double[hidden0Size];
        
        for (int j = 0; j < hidden0Size; j++) {
            hidden0Inputs[j] = biasesHidden0[j];
            for (int i = 0; i < inputSize; i++) {
                hidden0Inputs[j] += input[i] * weightsInputHidden0[i][j];
            }
            hidden0Outputs[j] = sigmoid(hidden0Inputs[j]);

            if (isTraining && dropoutRate > 0) {
                if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                    hidden0Outputs[j] = 0;
                } else {
                    hidden0Outputs[j] /= (1.0 - dropoutRate);
                }
            }
        }

        double[] hidden1Inputs = new double[hidden1Size];
        double[] hidden1Outputs = new double[hidden1Size];
        
        for (int k = 0; k < hidden1Size; k++) {
            hidden1Inputs[k] = biasesHidden1[k];
            for (int j = 0; j < hidden0Size; j++) {
                hidden1Inputs[k] += hidden0Outputs[j] * weightsHidden0Hidden1[j][k];
            }
            hidden1Outputs[k] = sigmoid(hidden1Inputs[k]);

            if (isTraining && dropoutRate > 0) {
                if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                    hidden1Outputs[k] = 0;
                } else {
                    hidden1Outputs[k] /= (1.0 - dropoutRate);
                }
            }
        }

        double[] hidden2Inputs = new double[hidden2Size];
        double[] hidden2Outputs = new double[hidden2Size];
        
        for (int l = 0; l < hidden2Size; l++) {
            hidden2Inputs[l] = biasesHidden2[l];
            for (int k = 0; k < hidden1Size; k++) {
                hidden2Inputs[l] += hidden1Outputs[k] * weightsHidden1Hidden2[k][l];
            }
            hidden2Outputs[l] = sigmoid(hidden2Inputs[l]);

            if (isTraining && dropoutRate > 0) {
                if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                    hidden2Outputs[l] = 0;
                } else {
                    hidden2Outputs[l] /= (1.0 - dropoutRate);
                }
            }
        }

        double[] hidden3Inputs = new double[hidden3Size];
        double[] hidden3Outputs = new double[hidden3Size];
        
        for (int m = 0; m < hidden3Size; m++) {
            hidden3Inputs[m] = biasesHidden3[m];
            for (int l = 0; l < hidden2Size; l++) {
                hidden3Inputs[m] += hidden2Outputs[l] * weightsHidden2Hidden3[l][m];
            }
            hidden3Outputs[m] = sigmoid(hidden3Inputs[m]);

            if (isTraining && dropoutRate > 0) {
                if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                    hidden3Outputs[m] = 0;
                } else {
                    hidden3Outputs[m] /= (1.0 - dropoutRate);
                }
            }
        }

        double[] outputInputs = new double[outputSize];
        for (int n = 0; n < outputSize; n++) {
            outputInputs[n] = biasesOutput[n];
            for (int m = 0; m < hidden3Size; m++) {
                outputInputs[n] += hidden3Outputs[m] * weightsHidden3Output[m][n];
            }
        }
        
        return new double[][] { hidden0Outputs, hidden1Outputs, hidden2Outputs, hidden3Outputs, outputInputs };
    }
    
    private void saveModelState() {
        bestWeightsInputHidden0 = new double[inputSize][hidden0Size];
        bestBiasesHidden0 = new double[hidden0Size];
        bestWeightsHidden0Hidden1 = new double[hidden0Size][hidden1Size];
        bestBiasesHidden1 = new double[hidden1Size];
        bestWeightsHidden1Hidden2 = new double[hidden1Size][hidden2Size];
        bestBiasesHidden2 = new double[hidden2Size];
        bestWeightsHidden2Hidden3 = new double[hidden2Size][hidden3Size];
        bestBiasesHidden3 = new double[hidden3Size];
        bestWeightsHidden3Output = new double[hidden3Size][outputSize];
        bestBiasesOutput = new double[outputSize];

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hidden0Size; j++) {
                bestWeightsInputHidden0[i][j] = weightsInputHidden0[i][j];
            }
        }

        for (int j = 0; j < hidden0Size; j++) {
            bestBiasesHidden0[j] = biasesHidden0[j];
            for (int k = 0; k < hidden1Size; k++) {
                bestWeightsHidden0Hidden1[j][k] = weightsHidden0Hidden1[j][k];
            }
        }

        for (int k = 0; k < hidden1Size; k++) {
            bestBiasesHidden1[k] = biasesHidden1[k];
            for (int l = 0; l < hidden2Size; l++) {
                bestWeightsHidden1Hidden2[k][l] = weightsHidden1Hidden2[k][l];
            }
        }

        for (int l = 0; l < hidden2Size; l++) {
            bestBiasesHidden2[l] = biasesHidden2[l];
            for (int m = 0; m < hidden3Size; m++) {
                bestWeightsHidden2Hidden3[l][m] = weightsHidden2Hidden3[l][m];
            }
        }

        for (int m = 0; m < hidden3Size; m++) {
            bestBiasesHidden3[m] = biasesHidden3[m];
            for (int n = 0; n < outputSize; n++) {
                bestWeightsHidden3Output[m][n] = weightsHidden3Output[m][n];
            }
        }

        for (int n = 0; n < outputSize; n++) {
            bestBiasesOutput[n] = biasesOutput[n];
        }
    }
    
    private void restoreBestModel() {
        if (bestWeightsInputHidden0 == null) return;

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hidden0Size; j++) {
                weightsInputHidden0[i][j] = bestWeightsInputHidden0[i][j];
            }
        }

        for (int j = 0; j < hidden0Size; j++) {
            biasesHidden0[j] = bestBiasesHidden0[j];
            for (int k = 0; k < hidden1Size; k++) {
                weightsHidden0Hidden1[j][k] = bestWeightsHidden0Hidden1[j][k];
            }
        }

        for (int k = 0; k < hidden1Size; k++) {
            biasesHidden1[k] = bestBiasesHidden1[k];
            for (int l = 0; l < hidden2Size; l++) {
                weightsHidden1Hidden2[k][l] = bestWeightsHidden1Hidden2[k][l];
            }
        }

        for (int l = 0; l < hidden2Size; l++) {
            biasesHidden2[l] = bestBiasesHidden2[l];
            for (int m = 0; m < hidden3Size; m++) {
                weightsHidden2Hidden3[l][m] = bestWeightsHidden2Hidden3[l][m];
            }
        }

        for (int m = 0; m < hidden3Size; m++) {
            biasesHidden3[m] = bestBiasesHidden3[m];
            for (int n = 0; n < outputSize; n++) {
                weightsHidden3Output[m][n] = bestWeightsHidden3Output[m][n];
            }
        }

        for (int n = 0; n < outputSize; n++) {
            biasesOutput[n] = bestBiasesOutput[n];
        }
    }
    
    private double evaluateError(List<Sample> samples) {
        double totalError = 0.0;
        
        for (Sample sample : samples) {
            double[] input = sample.getInput();
            double[] target = sample.getTarget();
            
            double[][] outputs = forwardPass(input);
            double[] finalOutputs = outputs[4];

            for (int k = 0; k < outputSize; k++) {
                totalError += Math.pow(target[k] - finalOutputs[k], 2);
            }
        }
        
        return totalError / (samples.size() * outputSize);
    }
    
    private Sample augmentSample(Sample sample) {
        double[] originalInput = sample.getInput();
        double[] augmentedInput = new double[originalInput.length];

        for (int i = 0; i < originalInput.length; i++) {
            augmentedInput[i] = originalInput[i] + ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
            augmentedInput[i] = Math.min(1.0, Math.max(0.0, augmentedInput[i]));
        }

        if (ThreadLocalRandom.current().nextDouble() < 0.9) {
            int pixelSize = 28;
            double centerX = pixelSize / 2.0;
            double centerY = pixelSize / 2.0;

            int numTransformations = ThreadLocalRandom.current().nextInt(1, 4);
            
            for (int t = 0; t < numTransformations; t++) {
                int transformType = ThreadLocalRandom.current().nextInt(5);
                
                switch (transformType) {
                    case 0:
                        int shiftX = ThreadLocalRandom.current().nextInt(-2, 3);
                        int shiftY = ThreadLocalRandom.current().nextInt(-2, 3);
                        double[] shifted = new double[augmentedInput.length];
                        
                        for (int y = 0; y < pixelSize; y++) {
                            for (int x = 0; x < pixelSize; x++) {
                                int sourceX = x - shiftX;
                                int sourceY = y - shiftY;
                                
                                if (sourceX >= 0 && sourceX < pixelSize && 
                                    sourceY >= 0 && sourceY < pixelSize) {
                                    shifted[y * pixelSize + x] = augmentedInput[sourceY * pixelSize + sourceX];
                                } else {
                                    shifted[y * pixelSize + x] = 0.0;
                                }
                            }
                        }
                        
                        augmentedInput = shifted;
                        break;
                        
                    case 1:
                        int numErasures = ThreadLocalRandom.current().nextInt(1, 4);
                        
                        for (int e = 0; e < numErasures; e++) {
                            int eraseX = ThreadLocalRandom.current().nextInt(pixelSize);
                            int eraseY = ThreadLocalRandom.current().nextInt(pixelSize);
                            int eraseSize = ThreadLocalRandom.current().nextInt(1, 4);
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
                        
                    case 2:
                        double[] rotated = new double[augmentedInput.length];
                        Arrays.fill(rotated, 0.0);
                        
                        double angle = ThreadLocalRandom.current().nextDouble(-0.2, 0.2);
                        
                        for (int y = 0; y < pixelSize; y++) {
                            for (int x = 0; x < pixelSize; x++) {
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
                        
                    case 3:
                        double[] scaled = new double[augmentedInput.length];
                        Arrays.fill(scaled, 0.0);
                        
                        double scaleFactor = ThreadLocalRandom.current().nextDouble(0.8, 1.2); // 80% to 120%
                        
                        for (int y = 0; y < pixelSize; y++) {
                            for (int x = 0; x < pixelSize; x++) {
                                double srcX = (x - centerX) / scaleFactor + centerX;
                                double srcY = (y - centerY) / scaleFactor + centerY;

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
                        
                    case 4:
                        double[] distorted = new double[augmentedInput.length];
                        double[][] displacementX = new double[pixelSize][pixelSize];
                        double[][] displacementY = new double[pixelSize][pixelSize];

                        double elasticScale = ThreadLocalRandom.current().nextDouble(3.0, 6.0);

                        int fieldSize = 7;
                        double[][] smallFieldX = new double[fieldSize][fieldSize];
                        double[][] smallFieldY = new double[fieldSize][fieldSize];
                        
                        for (int i = 0; i < fieldSize; i++) {
                            for (int j = 0; j < fieldSize; j++) {
                                smallFieldX[i][j] = ThreadLocalRandom.current().nextDouble(-1, 1);
                                smallFieldY[i][j] = ThreadLocalRandom.current().nextDouble(-1, 1);
                            }
                        }

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

                        for (int y = 0; y < pixelSize; y++) {
                            for (int x = 0; x < pixelSize; x++) {
                                double srcX = x + displacementX[y][x];
                                double srcY = y + displacementY[y][x];

                                srcX = Math.max(0, Math.min(pixelSize - 1, srcX));
                                srcY = Math.max(0, Math.min(pixelSize - 1, srcY));

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
    
    public void train(List<Sample> samples, int epochs) {
        int totalSamples = samples.size();
        if (totalSamples == 0) {
            System.err.println("Brak danych do uczenia!");
            return;
        }

        double startLR = 0.0001;
        double peakLR = 0.01;
        int warmupEpochs = 10;

        learningRate = startLR;
        
        System.out.println("Rozpoczęcie uczenia sieci neuronowej...");
        System.out.println("Architektura: " + inputSize + " → " + hidden0Size + " → " + 
                           hidden1Size + " → " + hidden2Size + " → " + hidden3Size + " → " + outputSize);
        System.out.println("Liczba epok: " + epochs);
        System.out.println("Rozmiar zbioru uczącego: " + totalSamples);
        System.out.println("Learning rate: początkowy=" + startLR + ", maksymalny=" + peakLR);
        System.out.println("Rozgrzewanie: " + warmupEpochs + " epok");
        System.out.println("Dropout rate: " + dropoutRate);

        Collections.shuffle(samples);
        int validationSize = (int)(totalSamples * validationSplit);
        int trainingSize = totalSamples - validationSize;
        
        List<Sample> trainingData = new ArrayList<>(samples.subList(0, trainingSize));
        List<Sample> validationData = new ArrayList<>(samples.subList(trainingSize, totalSamples));
        
        System.out.println("Rozmiar zbioru treningowego: " + trainingData.size());
        System.out.println("Rozmiar zbioru walidacyjnego: " + validationData.size());

        bestValidationError = Double.MAX_VALUE;
        epochsSinceImprovement = 0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            if (epoch < warmupEpochs) {
                learningRate = startLR + (peakLR - startLR) * (epoch / (double)warmupEpochs);
                System.out.println("Rozgrzewanie: Learning rate zwiększony do: " + learningRate);
            } else if ((epoch - warmupEpochs) % 25 == 0 && epoch > warmupEpochs) {
                learningRate *= 0.9;
                System.out.println("Learning rate zmniejszony do: " + learningRate);
            }

            List<Sample> augmentedData = new ArrayList<>();
            for (Sample s : trainingData) {
                augmentedData.add(s);

                int numAugmentations = ThreadLocalRandom.current().nextInt(3, 6);
                for (int i = 0; i < numAugmentations; i++) {
                    augmentedData.add(augmentSample(s));
                }
            }

            if (epoch == 0) {
                System.out.println("Liczba próbek augmentowanych: " + augmentedData.size());
                System.out.println("Stosunek augmentacji: " + String.format("%.1f", 
                    (double)augmentedData.size() / trainingData.size()) + "x");
            }
            
            double totalTrainingError = 0.0;

            isTraining = true;

            Collections.shuffle(augmentedData);
            
            for (Sample sample : augmentedData) {
                double[] input = sample.getInput();
                double[] target = sample.getTarget();

                double[][] outputs = forwardPass(input);
                double[] hidden0Outputs = outputs[0];
                double[] hidden1Outputs = outputs[1];
                double[] hidden2Outputs = outputs[2];
                double[] hidden3Outputs = outputs[3];
                double[] finalOutputs = outputs[4];

                double[] outputErrors = new double[outputSize];
                for (int n = 0; n < outputSize; n++) {
                    outputErrors[n] = target[n] - finalOutputs[n];
                    totalTrainingError += Math.pow(outputErrors[n], 2);
                }

                double[] outputDeltas = new double[outputSize];
                for (int n = 0; n < outputSize; n++) {
                    outputDeltas[n] = outputErrors[n];
                }

                double[] hidden3Errors = new double[hidden3Size];
                for (int m = 0; m < hidden3Size; m++) {
                    if (hidden3Outputs[m] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    double error = 0.0;
                    for (int n = 0; n < outputSize; n++) {
                        error += outputDeltas[n] * weightsHidden3Output[m][n];
                    }
                    hidden3Errors[m] = error;
                }

                double[] hidden3Deltas = new double[hidden3Size];
                for (int m = 0; m < hidden3Size; m++) {
                    if (hidden3Outputs[m] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    hidden3Deltas[m] = hidden3Errors[m] * hidden3Outputs[m] * (1 - hidden3Outputs[m]);
                    if (isTraining && dropoutRate > 0) {
                        hidden3Deltas[m] *= (1.0 - dropoutRate);
                    }
                }

                double[] hidden2Errors = new double[hidden2Size];
                for (int l = 0; l < hidden2Size; l++) {
                    if (hidden2Outputs[l] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    double error = 0.0;
                    for (int m = 0; m < hidden3Size; m++) {
                        error += hidden3Deltas[m] * weightsHidden2Hidden3[l][m];
                    }
                    hidden2Errors[l] = error;
                }

                double[] hidden2Deltas = new double[hidden2Size];
                for (int l = 0; l < hidden2Size; l++) {
                    if (hidden2Outputs[l] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    hidden2Deltas[l] = hidden2Errors[l] * hidden2Outputs[l] * (1 - hidden2Outputs[l]);
                    if (isTraining && dropoutRate > 0) {
                        hidden2Deltas[l] *= (1.0 - dropoutRate);
                    }
                }

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

                double[] hidden1Deltas = new double[hidden1Size];
                for (int k = 0; k < hidden1Size; k++) {
                    if (hidden1Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    hidden1Deltas[k] = hidden1Errors[k] * hidden1Outputs[k] * (1 - hidden1Outputs[k]);
                    if (isTraining && dropoutRate > 0) {
                        hidden1Deltas[k] *= (1.0 - dropoutRate);
                    }
                }

                double[] hidden0Errors = new double[hidden0Size];
                for (int j = 0; j < hidden0Size; j++) {
                    if (hidden0Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    double error = 0.0;
                    for (int k = 0; k < hidden1Size; k++) {
                        error += hidden1Deltas[k] * weightsHidden0Hidden1[j][k];
                    }
                    hidden0Errors[j] = error;
                }

                double[] hidden0Deltas = new double[hidden0Size];
                for (int j = 0; j < hidden0Size; j++) {
                    if (hidden0Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    hidden0Deltas[j] = hidden0Errors[j] * hidden0Outputs[j] * (1 - hidden0Outputs[j]);
                    if (isTraining && dropoutRate > 0) {
                        hidden0Deltas[j] *= (1.0 - dropoutRate);
                    }
                }

                for (int m = 0; m < hidden3Size; m++) {
                    if (hidden3Outputs[m] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    for (int n = 0; n < outputSize; n++) {
                        weightsHidden3Output[m][n] += learningRate * outputDeltas[n] * hidden3Outputs[m];
                    }
                }
                
                for (int n = 0; n < outputSize; n++) {
                    biasesOutput[n] += learningRate * outputDeltas[n];
                }

                for (int l = 0; l < hidden2Size; l++) {
                    if (hidden2Outputs[l] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    for (int m = 0; m < hidden3Size; m++) {
                        if (hidden3Outputs[m] == 0 && isTraining && dropoutRate > 0) continue;
                        
                        weightsHidden2Hidden3[l][m] += learningRate * hidden3Deltas[m] * hidden2Outputs[l];
                    }
                }
                
                for (int m = 0; m < hidden3Size; m++) {
                    if (hidden3Outputs[m] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    biasesHidden3[m] += learningRate * hidden3Deltas[m];
                }

                for (int k = 0; k < hidden1Size; k++) {
                    if (hidden1Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    for (int l = 0; l < hidden2Size; l++) {
                        if (hidden2Outputs[l] == 0 && isTraining && dropoutRate > 0) continue;
                        
                        weightsHidden1Hidden2[k][l] += learningRate * hidden2Deltas[l] * hidden1Outputs[k];
                    }
                }
                
                for (int l = 0; l < hidden2Size; l++) {
                    if (hidden2Outputs[l] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    biasesHidden2[l] += learningRate * hidden2Deltas[l];
                }

                for (int j = 0; j < hidden0Size; j++) {
                    if (hidden0Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    for (int k = 0; k < hidden1Size; k++) {
                        if (hidden1Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                        
                        weightsHidden0Hidden1[j][k] += learningRate * hidden1Deltas[k] * hidden0Outputs[j];
                    }
                }
                
                for (int k = 0; k < hidden1Size; k++) {
                    if (hidden1Outputs[k] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    biasesHidden1[k] += learningRate * hidden1Deltas[k];
                }

                for (int i = 0; i < inputSize; i++) {
                    for (int j = 0; j < hidden0Size; j++) {
                        if (hidden0Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                        
                        weightsInputHidden0[i][j] += learningRate * hidden0Deltas[j] * input[i];
                    }
                }
                
                for (int j = 0; j < hidden0Size; j++) {
                    if (hidden0Outputs[j] == 0 && isTraining && dropoutRate > 0) continue;
                    
                    biasesHidden0[j] += learningRate * hidden0Deltas[j];
                }
            }

            isTraining = false;

            double trainingError = totalTrainingError / (augmentedData.size() * outputSize);
            double validationError = evaluateError(validationData);

            System.out.printf("Epoka %d/%d, błąd (trening): %.6f, błąd (walidacja): %.6f%n", 
                             epoch + 1, epochs, trainingError, validationError);

            if (validationError < bestValidationError) {
                bestValidationError = validationError;
                epochsSinceImprovement = 0;
                saveModelState();
            } else {
                epochsSinceImprovement++;
            }
            
            if (epochsSinceImprovement >= patience) {
                System.out.println("Wczesne zatrzymanie na epoce " + (epoch + 1) + 
                                  " (błąd walidacji nie poprawiał się przez " + patience + " epok)");
                break;
            }
        }

        restoreBestModel();
        System.out.println("Uczenie zakończone! Najlepszy błąd walidacji: " + bestValidationError);
    }
    
    public double[] predict(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Nieprawidłowy rozmiar danych wejściowych: " + input.length + 
                                              " (oczekiwano " + inputSize + ")");
        }

        isTraining = false;
        
        double[][] outputs = forwardPass(input);
        return outputs[4];
    }
    
    public void saveModel(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeInt(inputSize);
            oos.writeInt(hidden0Size);
            oos.writeInt(hidden1Size);
            oos.writeInt(hidden2Size);
            oos.writeInt(hidden3Size);
            oos.writeInt(outputSize);
            oos.writeDouble(learningRate);
            oos.writeDouble(dropoutRate);

            oos.writeObject(weightsInputHidden0);
            oos.writeObject(biasesHidden0);
            oos.writeObject(weightsHidden0Hidden1);
            oos.writeObject(biasesHidden1);
            oos.writeObject(weightsHidden1Hidden2);
            oos.writeObject(biasesHidden2);
            oos.writeObject(weightsHidden2Hidden3);
            oos.writeObject(biasesHidden3);
            oos.writeObject(weightsHidden3Output);
            oos.writeObject(biasesOutput);
            
            System.out.println("Model został pomyślnie zapisany do pliku: " + path);
        } catch (IOException e) {
            System.err.println("Błąd podczas zapisywania modelu: " + e.getMessage());
            throw e;
        }
    }
    
    public void loadModel(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            this.inputSize = ois.readInt();

            try {
                this.hidden0Size = ois.readInt();
                this.hidden1Size = ois.readInt();
                this.hidden2Size = ois.readInt();
                this.hidden3Size = ois.readInt();
                this.outputSize = ois.readInt();
                this.learningRate = ois.readDouble();
                this.dropoutRate = ois.readDouble();

                this.weightsInputHidden0 = (double[][]) ois.readObject();
                this.biasesHidden0 = (double[]) ois.readObject();
                this.weightsHidden0Hidden1 = (double[][]) ois.readObject();
                this.biasesHidden1 = (double[]) ois.readObject();
                this.weightsHidden1Hidden2 = (double[][]) ois.readObject();
                this.biasesHidden2 = (double[]) ois.readObject();
                this.weightsHidden2Hidden3 = (double[][]) ois.readObject();
                this.biasesHidden3 = (double[]) ois.readObject();
                this.weightsHidden3Output = (double[][]) ois.readObject();
                this.biasesOutput = (double[]) ois.readObject();
                
                System.out.println("Model został pomyślnie załadowany z pliku: " + path);
                System.out.println("Architektura: " + inputSize + " → " + hidden0Size + " → " + 
                                  hidden1Size + " → " + hidden2Size + " → " + hidden3Size + " → " + outputSize);
            } catch (Exception e) {
                System.err.println("Format pliku modelu nie odpowiada oczekiwanemu: " + e.getMessage());
                throw e;
            }
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Błąd podczas ładowania modelu: " + e.getMessage());
            throw e;
        }
    }
}
