import java.io.*;
import java.util.*;

public class NeuralNetwork implements Serializable {
    private Layer hidden;
    private Layer output;
    private double learningRate = 0.1;
    private static final long serialVersionUID = 1L;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        hidden = new Layer(inputSize, hiddenSize);
        output = new Layer(hiddenSize, outputSize);
    }

    public double[] predict(double[] input) {
        double[] hiddenOut = hidden.forward(input);
        double[] z = output.forward(hiddenOut);
        return ActivationFunction.softmax(z); // краще softmax на виході
    }

    public void train(List<Sample> samples, int epochs) {
        System.out.println("Початок тренування на " + samples.size() + " зразках, " + epochs + " епох");
        long startTime = System.currentTimeMillis();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Перемішуємо зразки на початку кожної епохи
            Collections.shuffle(samples);
            
            double loss = 0;
            for (Sample s : samples) {
                forwardBackward(s.input, s.target);
                double[] pred = predict(s.input);
                for (int i = 0; i < pred.length; i++) {
                    loss += Math.pow(pred[i] - s.target[i], 2);
                }
            }
            loss /= samples.size();
            
            // Відображення часу та прогресу
            if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == epochs - 1) {
                long currentTime = System.currentTimeMillis();
                double timeElapsed = (currentTime - startTime) / 1000.0;
                double timePerEpoch = timeElapsed / (epoch + 1);
                double timeRemaining = timePerEpoch * (epochs - epoch - 1);
                
                System.out.printf("Епоха %d/%d - Помилка: %.6f - Час: %.2f сек (залишилось: %.2f сек)%n", 
                    epoch + 1, epochs, loss, timeElapsed, timeRemaining);
            }
        }
        
        long endTime = System.currentTimeMillis();
        double totalTime = (endTime - startTime) / 1000.0;
        System.out.printf("Тренування завершено за %.2f секунд%n", totalTime);
    }

    private void forwardBackward(double[] input, double[] target) {
        double[] hiddenOut = hidden.forward(input);
        double[] z = output.forward(hiddenOut);
        double[] predicted = ActivationFunction.softmax(z);

        // Output layer error
        double[] deltaOutput = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            deltaOutput[i] = (predicted[i] - target[i]) * predicted[i] * (1 - predicted[i]);
        }

        // Hidden layer error
        double[] deltaHidden = new double[hidden.outputSize];
        for (int i = 0; i < hidden.outputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < output.outputSize; j++) {
                sum += deltaOutput[j] * output.weights[j][i];
            }
            deltaHidden[i] = sum * hidden.outputs[i] * (1 - hidden.outputs[i]);
        }

        // Update output layer weights
        for (int i = 0; i < output.outputSize; i++) {
            for (int j = 0; j < output.inputSize; j++) {
                output.weights[i][j] -= learningRate * deltaOutput[i] * hidden.outputs[j];
            }
            output.biases[i] -= learningRate * deltaOutput[i];
        }

        // Update hidden layer weights
        for (int i = 0; i < hidden.outputSize; i++) {
            for (int j = 0; j < hidden.inputSize; j++) {
                hidden.weights[i][j] -= learningRate * deltaHidden[i] * hidden.inputs[j];
            }
            hidden.biases[i] -= learningRate * deltaHidden[i];
        }
    }

    // Save & Load
    public void saveModel(String path) throws IOException {
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path));
        out.writeObject(hidden.weights);
        out.writeObject(hidden.biases);
        out.writeObject(output.weights);
        out.writeObject(output.biases);
        out.close();
    }

    @SuppressWarnings("unchecked")
    public void loadModel(String path) throws IOException, ClassNotFoundException {
        ObjectInputStream in = new ObjectInputStream(new FileInputStream(path));
        hidden.weights = (double[][]) in.readObject();
        hidden.biases = (double[]) in.readObject();
        output.weights = (double[][]) in.readObject();
        output.biases = (double[]) in.readObject();
        in.close();
    }
}
