import java.io.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Implementacja sieci neuronowej do rozpoznawania liter.
 * Architektura: 784 → 512 → 256 → 128 → 64 → 3 (wejście → warstwy ukryte → wyjście)
 */
public class NeuralNetwork {
    // Wymiary warstw sieci
    private int inputSize;          // Rozmiar warstwy wejściowej (784 = 28x28 pikseli)
    private int hidden0Size;        // Rozmiar pierwszej warstwy ukrytej
    private int hidden1Size;        // Rozmiar drugiej warstwy ukrytej
    private int hidden2Size;        // Rozmiar trzeciej warstwy ukrytej
    private int hidden3Size;        // Rozmiar czwartej warstwy ukrytej
    private int outputSize;         // Rozmiar warstwy wyjściowej (3 neurony dla M, O, N)

    // Tablica przechowująca rozmiary wszystkich warstw dla łatwiejszej iteracji
    private int[] layerSizes;

    // Wagi i biasy
    private double[][][] weights;    // [warstwa][neuronWejściowy][neuronWyjściowy]
    private double[][] biases;       // [warstwa][neuron]
    
    // Najlepszy stan modelu do wczesnego zatrzymania
    private double[][][] bestWeights;
    private double[][] bestBiases;

    private double learningRate;     // Współczynnik uczenia
    private double dropoutRate = 0.0; // Współczynnik dropout (zapobiega przeuczeniu)
    private boolean isTraining = false; // Flaga trybu treningowego

    // Parametry wczesnego zatrzymania
    private int patience = 10;        // Liczba epok bez poprawy przed zatrzymaniem
    private double bestValidationError = Double.MAX_VALUE; // Najniższy błąd walidacji
    private int epochsSinceImprovement = 0;  // Liczba epok bez poprawy
    private double validationSplit = 0.2;    // Procent danych używany do walidacji

    /**
     * Konstruktor z pełną specyfikacją architektury sieci
     */
    public NeuralNetwork(int inputSize, int hidden0Size, int hidden1Size, int hidden2Size, int hidden3Size, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hidden0Size = hidden0Size;
        this.hidden1Size = hidden1Size;
        this.hidden2Size = hidden2Size;
        this.hidden3Size = hidden3Size;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        
        // Inicjalizacja tablicy rozmiarów warstw dla wygody
        this.layerSizes = new int[]{inputSize, hidden0Size, hidden1Size, hidden2Size, hidden3Size, outputSize};
        
        // Inicjalizacja wag i biasów
        int numLayers = layerSizes.length - 1;
        this.weights = new double[numLayers][][];
        this.biases = new double[numLayers][];
        
        initializeWeightsAndBiases();
    }
    
    /**
     * Konstruktor domyślny z predefiniowaną architekturą 784→512→256→128→64→3
     */
    public NeuralNetwork() {
        this(784, 512, 256, 128, 64, 3, 0.01);
    }

    /**
     * Ustawia współczynnik dropout (procent neuronów tymczasowo wyłączanych podczas treningu)
     */
    public void setDropoutRate(double rate) {
        if (rate < 0.0 || rate >= 1.0) {
            throw new IllegalArgumentException("Współczynnik dropout musi być pomiędzy 0 a 1");
        }
        this.dropoutRate = rate;
    }

    /**
     * Ustawia cierpliwość dla wczesnego zatrzymania (ile epok bez poprawy przed zatrzymaniem)
     */
    public void setPatience(int patience) {
        this.patience = patience;
    }

    /**
     * Ustawia procent danych używany do walidacji
     */
    public void setValidationSplit(double ratio) {
        if (ratio <= 0.0 || ratio >= 1.0) {
            throw new IllegalArgumentException("Podział walidacyjny musi być pomiędzy 0 a 1");
        }
        this.validationSplit = ratio;
    }

    /**
     * Inicjalizacja wag i biasów z użyciem inicjalizacji Xaviera/Glorota
     */
    private void initializeWeightsAndBiases() {
        int numLayers = layerSizes.length - 1;
        
        // Inicjalizacja wag i biasów dla każdej warstwy
        for (int layer = 0; layer < numLayers; layer++) {
            int inputNeurons = layerSizes[layer];
            int outputNeurons = layerSizes[layer + 1];
            
            weights[layer] = new double[inputNeurons][outputNeurons];
            biases[layer] = new double[outputNeurons];
            
            // Inicjalizacja Xaviera/Glorota dla wag
            double limit = Math.sqrt(6.0 / (inputNeurons + outputNeurons));
            
            for (int i = 0; i < inputNeurons; i++) {
                for (int j = 0; j < outputNeurons; j++) {
                    weights[layer][i][j] = ThreadLocalRandom.current().nextDouble(-limit, limit);
                }
            }
            
            // Inicjalizacja biasów małymi losowymi wartościami
            for (int j = 0; j < outputNeurons; j++) {
                biases[layer][j] = ThreadLocalRandom.current().nextDouble(-0.1, 0.1);
            }
        }
    }
    
    /**
     * Funkcja aktywacji sigmoid: f(x) = 1 / (1 + e^(-x))
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Przeprowadza przejście w przód (forward pass) przez sieć neuronową
     * Zwraca tablicę wyjść każdej warstwy
     */
    private double[][] forwardPass(double[] input) {
        int numLayers = layerSizes.length;
        double[][] layerOutputs = new double[numLayers][];
        
        // Ustaw wejście jako wyjście pierwszej warstwy
        layerOutputs[0] = input;
        
        // Przetwarzanie każdej warstwy
        for (int layer = 0; layer < numLayers - 1; layer++) {
            int currentLayerSize = layerSizes[layer];
            int nextLayerSize = layerSizes[layer + 1];
            layerOutputs[layer + 1] = new double[nextLayerSize];
            
            // Oblicz wyjście dla każdego neuronu w bieżącej warstwie
            for (int j = 0; j < nextLayerSize; j++) {
                double sum = biases[layer][j];
                
                for (int i = 0; i < currentLayerSize; i++) {
                    sum += layerOutputs[layer][i] * weights[layer][i][j];
                }
                
                // Zastosuj funkcję aktywacji (sigmoid dla warstw ukrytych, brak aktywacji dla warstwy wyjściowej)
                boolean isOutputLayer = (layer == numLayers - 2);
                
                if (!isOutputLayer) {
                    layerOutputs[layer + 1][j] = sigmoid(sum);
                    
                    // Zastosuj dropout podczas treningu
                    if (isTraining && dropoutRate > 0) {
                        if (ThreadLocalRandom.current().nextDouble() < dropoutRate) {
                            layerOutputs[layer + 1][j] = 0;
                        } else {
                            layerOutputs[layer + 1][j] /= (1.0 - dropoutRate);
                        }
                    }
                } else {
                    layerOutputs[layer + 1][j] = sum; // Liniowe wyjście dla ostatniej warstwy
                }
            }
        }
        
        return layerOutputs;
    }
    
    /**
     * Zapisuje aktualny stan modelu jako najlepszy
     */
    private void saveModelState() {
        int numLayers = layerSizes.length - 1;
        bestWeights = new double[numLayers][][];
        bestBiases = new double[numLayers][];
        
        // Głęboka kopia wszystkich wag i biasów
        for (int layer = 0; layer < numLayers; layer++) {
            int inputNeurons = layerSizes[layer];
            int outputNeurons = layerSizes[layer + 1];
            
            bestWeights[layer] = new double[inputNeurons][outputNeurons];
            bestBiases[layer] = new double[outputNeurons];
            
            // Kopiowanie wag
            for (int i = 0; i < inputNeurons; i++) {
                for (int j = 0; j < outputNeurons; j++) {
                    bestWeights[layer][i][j] = weights[layer][i][j];
                }
            }
            
            // Kopiowanie biasów
            for (int j = 0; j < outputNeurons; j++) {
                bestBiases[layer][j] = biases[layer][j];
            }
        }
    }
    
    /**
     * Przywraca najlepszy zapisany stan modelu
     */
    private void restoreBestModel() {
        if (bestWeights == null) return;
        
        int numLayers = layerSizes.length - 1;
        
        // Przywróć wszystkie wagi i biasy
        for (int layer = 0; layer < numLayers; layer++) {
            int inputNeurons = layerSizes[layer];
            int outputNeurons = layerSizes[layer + 1];
            
            // Przywróć wagi
            for (int i = 0; i < inputNeurons; i++) {
                for (int j = 0; j < outputNeurons; j++) {
                    weights[layer][i][j] = bestWeights[layer][i][j];
                }
            }
            
            // Przywróć biasy
            for (int j = 0; j < outputNeurons; j++) {
                biases[layer][j] = bestBiases[layer][j];
            }
        }
    }
    
    /**
     * Oblicza błąd średniokwadratowy na zbiorze próbek
     */
    private double evaluateError(List<Sample> samples) {
        double totalError = 0.0;
        
        for (Sample sample : samples) {
            double[] input = sample.getInput();
            double[] target = sample.getTarget();
            
            double[][] outputs = forwardPass(input);
            double[] finalOutputs = outputs[outputs.length - 1];

            for (int k = 0; k < outputSize; k++) {
                totalError += Math.pow(target[k] - finalOutputs[k], 2);
            }
        }
        
        return totalError / (samples.size() * outputSize);
    }
    
    /**
     * Tworzy zmodyfikowaną wersję próbki przez zastosowanie różnych transformacji:
     * - dodanie szumu
     * - przesunięcie obrazu
     * - wymazanie części obrazu
     * - obrót obrazu
     * - skalowanie obrazu
     * - zniekształcenie elastyczne
     */
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
    
    /**
     * Główna metoda treningu sieci neuronowej
     */
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
        
        // Wyświetlenie informacji o konfiguracji treningu
        System.out.println("Rozpoczęcie uczenia sieci neuronowej...");
        System.out.println("Architektura: " + getArchitectureString());
        System.out.println("Liczba epok: " + epochs);
        System.out.println("Rozmiar zbioru uczącego: " + totalSamples);
        System.out.println("Learning rate: początkowy=" + startLR + ", maksymalny=" + peakLR);
        System.out.println("Rozgrzewanie: " + warmupEpochs + " epok");
        System.out.println("Dropout rate: " + dropoutRate);

        // Podział danych na zbiór treningowy i walidacyjny
        Collections.shuffle(samples);
        int validationSize = (int)(totalSamples * validationSplit);
        int trainingSize = totalSamples - validationSize;
        
        List<Sample> trainingData = new ArrayList<>(samples.subList(0, trainingSize));
        List<Sample> validationData = new ArrayList<>(samples.subList(trainingSize, totalSamples));
        
        System.out.println("Rozmiar zbioru treningowego: " + trainingData.size());
        System.out.println("Rozmiar zbioru walidacyjnego: " + validationData.size());

        // Resetowanie zmiennych śledzących dla wczesnego zatrzymania
        bestValidationError = Double.MAX_VALUE;
        epochsSinceImprovement = 0;
        
        // Pętla treningowa
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Harmonogram współczynnika uczenia
            updateLearningRate(epoch, startLR, peakLR, warmupEpochs);
            
            // Tworzenie rozszerzonego zbioru danych
            List<Sample> augmentedData = createAugmentedData(trainingData, epoch);
            
            // Trening na rozszerzonym zbiorze danych
            double totalTrainingError = trainEpoch(augmentedData);
            
            // Ocena na zbiorze walidacyjnym
            double trainingError = totalTrainingError / (augmentedData.size() * outputSize);
            double validationError = evaluateError(validationData);

            // Raportowanie postępu
            System.out.printf("Epoka %d/%d, błąd (trening): %.6f, błąd (walidacja): %.6f%n", 
                             epoch + 1, epochs, trainingError, validationError);

            // Sprawdzenie warunku wczesnego zatrzymania
            if (checkEarlyStopping(validationError, epoch)) {
                break;
            }
        }

        // Przywrócenie najlepszego modelu i raportowanie zakończenia
        restoreBestModel();
        System.out.println("Uczenie zakończone! Najlepszy błąd walidacji: " + bestValidationError);
    }
    
    /**
     * Aktualizuje współczynnik uczenia według harmonogramu
     * - rozgrzewanie: liniowy wzrost od startLR do peakLR
     * - po rozgrzaniu: zmniejszenie o 10% co 25 epok
     */
    private void updateLearningRate(int epoch, double startLR, double peakLR, int warmupEpochs) {
        if (epoch < warmupEpochs) {
            // Liniowe rozgrzewanie
            learningRate = startLR + (peakLR - startLR) * (epoch / (double)warmupEpochs);
            System.out.println("Rozgrzewanie: Learning rate zwiększony do: " + learningRate);
        } else if ((epoch - warmupEpochs) % 25 == 0 && epoch > warmupEpochs) {
            // Zmniejszenie co 25 epok po rozgrzaniu
            learningRate *= 0.9;
            System.out.println("Learning rate zmniejszony do: " + learningRate);
        }
    }
    
    /**
     * Tworzy zbiór danych rozszerzony o augmentowane próbki
     */
    private List<Sample> createAugmentedData(List<Sample> trainingData, int epoch) {
        List<Sample> augmentedData = new ArrayList<>();
        
        for (Sample s : trainingData) {
            // Dodaj oryginalną próbkę
            augmentedData.add(s);

            // Dodaj zmodyfikowane wersje
            int numAugmentations = ThreadLocalRandom.current().nextInt(3, 6);
            for (int i = 0; i < numAugmentations; i++) {
                augmentedData.add(augmentSample(s));
            }
        }

        // Wyświetlenie statystyk augmentacji w pierwszej epoce
        if (epoch == 0) {
            System.out.println("Liczba próbek augmentowanych: " + augmentedData.size());
            System.out.println("Stosunek augmentacji: " + String.format("%.1f", 
                (double)augmentedData.size() / trainingData.size()) + "x");
        }
        
        Collections.shuffle(augmentedData);
        return augmentedData;
    }
    
    /**
     * Przeprowadza jedną epokę treningu na wszystkich próbkach
     */
    private double trainEpoch(List<Sample> augmentedData) {
        isTraining = true;
        double totalError = 0.0;
        
        for (Sample sample : augmentedData) {
            totalError += trainOnSample(sample);
        }
        
        isTraining = false;
        return totalError;
    }
    
    /**
     * Trenuje sieć na pojedynczej próbce (jeden krok propagacji wstecznej)
     */
    private double trainOnSample(Sample sample) {
        double[] input = sample.getInput();
        double[] target = sample.getTarget();
        
        // Przejście w przód
        double[][] layerOutputs = forwardPass(input);
        
        int numLayers = layerSizes.length;
        
        // Błąd warstwy wyjściowej i delty
        double[][] deltas = new double[numLayers - 1][];
        double totalError = 0.0;
        
        // Oblicz błąd dla warstwy wyjściowej
        deltas[numLayers - 2] = new double[outputSize];
        for (int n = 0; n < outputSize; n++) {
            double error = target[n] - layerOutputs[numLayers - 1][n];
            totalError += Math.pow(error, 2);
            deltas[numLayers - 2][n] = error; // Dla warstwy wyjściowej delta = błąd
        }
        
        // Propagacja wsteczna przez warstwy ukryte
        for (int layer = numLayers - 3; layer >= 0; layer--) {
            int currentLayerSize = layerSizes[layer + 1];
            int nextLayerSize = layerSizes[layer + 2];
            
            deltas[layer] = new double[currentLayerSize];
            
            // Oblicz delty dla bieżącej warstwy
            for (int j = 0; j < currentLayerSize; j++) {
                // Pomijaj neurony wyłączone przez dropout
                if (layerOutputs[layer + 1][j] == 0 && isTraining && dropoutRate > 0) {
                    continue;
                }
                
                // Oblicz błąd z następnej warstwy
                double error = 0.0;
                for (int k = 0; k < nextLayerSize; k++) {
                    error += deltas[layer + 1][k] * weights[layer + 1][j][k];
                }
                
                // Oblicz deltę z pochodną sigmoidu
                double output = layerOutputs[layer + 1][j];
                deltas[layer][j] = error * output * (1 - output);
                
                // Skaluj deltę dla dropout
                if (isTraining && dropoutRate > 0) {
                    deltas[layer][j] *= (1.0 - dropoutRate);
                }
            }
        }
        
        // Aktualizacja wag i biasów
        for (int layer = 0; layer < numLayers - 1; layer++) {
            int fromSize = layerSizes[layer];
            int toSize = layerSizes[layer + 1];
            
            // Aktualizacja wag i biasów dla każdego neuronu w tej warstwie
            for (int to = 0; to < toSize; to++) {
                // Pomijaj aktualizacje dla neuronów wyłączonych przez dropout
                if (layer < numLayers - 2 && // Nie warstwa wyjściowa
                    layerOutputs[layer + 1][to] == 0 && 
                    isTraining && dropoutRate > 0) {
                    continue;
                }
                
                // Aktualizacja biasu
                biases[layer][to] += learningRate * deltas[layer][to];
                
                // Aktualizacja wag
                for (int from = 0; from < fromSize; from++) {
                    weights[layer][from][to] += learningRate * deltas[layer][to] * layerOutputs[layer][from];
                }
            }
        }
        
        return totalError;
    }
    
    /**
     * Sprawdza warunek wczesnego zatrzymania i zapisuje model, jeśli jest lepszy
     * Zwraca true, jeśli trening powinien zostać zatrzymany
     */
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
    
    /**
     * Tworzy tekstową reprezentację architektury sieci
     */
    private String getArchitectureString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < layerSizes.length; i++) {
            sb.append(layerSizes[i]);
            if (i < layerSizes.length - 1) {
                sb.append(" → ");
            }
        }
        return sb.toString();
    }
    
    /**
     * Wykonuje predykcję dla podanego wektora wejściowego
     */
    public double[] predict(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Nieprawidłowy rozmiar danych wejściowych: " + input.length + 
                                              " (oczekiwano " + inputSize + ")");
        }

        isTraining = false;
        
        double[][] outputs = forwardPass(input);
        return outputs[outputs.length - 1];
    }
    
    /**
     * Zapisuje model do pliku
     */
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

            // Zapis wag i biasów z użyciem oryginalnej struktury dla kompatybilności wstecznej
            oos.writeObject(weights[0]);
            oos.writeObject(biases[0]);
            oos.writeObject(weights[1]);
            oos.writeObject(biases[1]);
            oos.writeObject(weights[2]); 
            oos.writeObject(biases[2]);
            oos.writeObject(weights[3]);
            oos.writeObject(biases[3]);
            oos.writeObject(weights[4]);
            oos.writeObject(biases[4]);
            
            System.out.println("Model został pomyślnie zapisany do pliku: " + path);
        } catch (IOException e) {
            System.err.println("Błąd podczas zapisywania modelu: " + e.getMessage());
            throw e;
        }
    }
    
    /**
     * Ładuje model z pliku
     */
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
                
                // Aktualizacja rozmiarów warstw
                this.layerSizes = new int[]{inputSize, hidden0Size, hidden1Size, hidden2Size, hidden3Size, outputSize};
                
                // Inicjalizacja tablic
                int numLayers = layerSizes.length - 1;
                this.weights = new double[numLayers][][];
                this.biases = new double[numLayers][];

                // Ładowanie wag i biasów
                weights[0] = (double[][]) ois.readObject();
                biases[0] = (double[]) ois.readObject();
                weights[1] = (double[][]) ois.readObject();
                biases[1] = (double[]) ois.readObject();
                weights[2] = (double[][]) ois.readObject();
                biases[2] = (double[]) ois.readObject();
                weights[3] = (double[][]) ois.readObject();
                biases[3] = (double[]) ois.readObject();
                weights[4] = (double[][]) ois.readObject();
                biases[4] = (double[]) ois.readObject();
                
                System.out.println("Model został pomyślnie załadowany z pliku: " + path);
                System.out.println("Architektura: " + getArchitectureString());
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
