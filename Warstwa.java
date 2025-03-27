import java.io.Serializable;
import java.util.Random;

public class Warstwa implements Serializable {
    private static final long serialVersionUID = 1L;
    
    Neuron [] neurony;
    int liczba_neuronow;
    private Neuron.ActivationFunction activationFunction;
    private double[] outputs; // Зберігаємо виходи шару
    
    // Масив для зберігання dropout маски
    private boolean[] dropoutMask;
    private Random random = new Random(42);
    
    public Warstwa(){
        neurony = null;
        liczba_neuronow = 0;
        activationFunction = Neuron.ActivationFunction.SIGMOID;
    }
    
    public Warstwa(int liczba_wejsc, int liczba_neuronow){
        this(liczba_wejsc, liczba_neuronow, Neuron.ActivationFunction.SIGMOID);
    }
    
    public Warstwa(int liczba_wejsc, int liczba_neuronow, Neuron.ActivationFunction activationFunction){
        this.liczba_neuronow = liczba_neuronow;
        this.activationFunction = activationFunction;
        neurony = new Neuron[liczba_neuronow];
        for(int i = 0; i < liczba_neuronow; i++)
            neurony[i] = new Neuron(liczba_wejsc, activationFunction);
    }
    
    double [] oblicz_wyjscie(double [] wejscia){
        outputs = new double[liczba_neuronow];
        
        // Розрахунок виходів нейронів
        for(int i = 0; i < liczba_neuronow; i++)
            outputs[i] = neurony[i].oblicz_wyjscie(wejscia);
        
        // Якщо це шар з Softmax активацією, застосовуємо Softmax до всього вектора
        if(activationFunction == Neuron.ActivationFunction.SOFTMAX) {
            applySoftmax(outputs);
        }
        
        return outputs;
    }
    
    // Метод для обчислення виходу шару з dropout
    public double[] obliczWyjscieZDropout(double[] wejscia, double dropoutRate) {
        outputs = new double[liczba_neuronow];
        dropoutMask = new boolean[liczba_neuronow];
        
        // Створюємо маску dropout
        for (int i = 0; i < liczba_neuronow; i++) {
            dropoutMask[i] = random.nextDouble() > dropoutRate;
        }
        
        // Обчислюємо вихід та застосовуємо dropout
        for (int i = 0; i < liczba_neuronow; i++) {
            if (dropoutMask[i]) {
                // Множимо вихід на 1/(1-dropoutRate) для збереження очікування
                outputs[i] = neurony[i].oblicz_wyjscie(wejscia) / (1.0 - dropoutRate);
            } else {
                outputs[i] = 0.0;
            }
        }
        
        // Для softmax потрібно застосувати після dropout
        if (activationFunction == Neuron.ActivationFunction.SOFTMAX) {
            applySoftmax(outputs);
        }
        
        return outputs;
    }
    
    // Метод для застосування функції Softmax до вектора з числовою стабільністю
    private void applySoftmax(double[] vector) {
        // Знаходимо максимальне значення для чисельної стабільності
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < vector.length; i++) {
            if (vector[i] > max) max = vector[i];
        }
        
        // Обчислюємо експоненти з вирахуванням максимального значення
        double sum = 0.0;
        for (int i = 0; i < vector.length; i++) {
            // Обмеження для запобігання overflow при exp()
            double expValue = Math.exp(Math.min(vector[i] - max, 700.0));
            vector[i] = expValue;
            sum += expValue;
        }
        
        // Додаткова перевірка suм для уникнення поділу на нуль
        if (sum < 1e-15) {
            sum = 1e-15;
        }
        
        // Нормалізуємо для отримання ймовірностей
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= sum;
            
            // Забезпечуємо, що значення знаходяться в допустимому діапазоні [0, 1]
            vector[i] = Math.min(Math.max(vector[i], 0.0), 1.0);
        }
    }
    
    // Метод для обчислення похідних для вихідного шару (з цільовими значеннями)
    public void calculateOutputDeltas(double[] targets) {
        for(int i = 0; i < liczba_neuronow; i++) {
            double error;
            
            if(activationFunction == Neuron.ActivationFunction.SOFTMAX) {
                // Для softmax + cross-entropy похідна спрощується до outputs - targets
                error = outputs[i] - targets[i];
                neurony[i].updateDelta(error);
            } else {
                // Для інших функцій активації
                error = targets[i] - outputs[i];
                neurony[i].updateDelta(error);
            }
        }
    }
    
    // Метод для обчислення похідних для прихованого шару
    public void calculateHiddenDeltas(Warstwa nextLayer) {
        for(int i = 0; i < liczba_neuronow; i++) {
            double error = 0.0;
            
            // Підсумовуємо зважені дельти з наступного шару
            for(int j = 0; j < nextLayer.liczba_neuronow; j++) {
                error += nextLayer.neurony[j].getWeights()[i + 1] * nextLayer.neurony[j].getDelta();
            }
            
            neurony[i].updateDelta(error);
        }
    }
    
    // Метод для оновлення ваг у шарі
    public void updateWeights(double learningRate) {
        for(int i = 0; i < liczba_neuronow; i++) {
            neurony[i].updateWeights(learningRate);
        }
    }
    
    // Метод для оновлення ваг з L2 регуляризацією
    public void updateWeightsWithRegularization(double learningRate, double lambda) {
        for (int i = 0; i < liczba_neuronow; i++) {
            // Якщо був застосований dropout і нейрон був вимкнений, не оновлюємо ваги
            if (dropoutMask != null && !dropoutMask[i]) {
                continue;
            }
            
            double[] weights = neurony[i].getWeights();
            double delta = neurony[i].getDelta();
            
            // Оновлюємо bias без регуляризації
            neurony[i].updateWeightWithIndex(0, learningRate * delta, 0);
            
            // Оновлюємо інші ваги з регуляризацією
            for (int j = 1; j < weights.length; j++) {
                // Правильне застосування L2-регуляризації
                double regularizationTerm = learningRate * lambda;
                neurony[i].updateWeightWithIndex(j, learningRate * delta * neurony[i].getLastInput(j-1), regularizationTerm);
            }
        }
    }
    
    // Створення глибокої копії шару для збереження найкращих ваг
    public Warstwa deepCopy() {
        Warstwa copy = new Warstwa(neurony[0].liczba_wejsc, liczba_neuronow, activationFunction);
        
        for (int i = 0; i < liczba_neuronow; i++) {
            double[] originalWeights = neurony[i].getWeights();
            double[] copyWeights = copy.neurony[i].getWeights();
            
            System.arraycopy(originalWeights, 0, copyWeights, 0, originalWeights.length);
        }
        
        return copy;
    }
    
    // Getters
    public double[] getOutputs() {
        return outputs;
    }
    
    public Neuron.ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}
