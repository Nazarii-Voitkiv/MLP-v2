import java.util.Random;
import java.io.Serializable;

public class Layer implements Serializable {
    public int inputSize;
    public int outputSize;
    public double[][] weights;
    public double[] biases;
    public double[] outputs;
    public double[] inputs;
    public double[] zValues;

    public Layer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        weights = new double[outputSize][inputSize];
        biases = new double[outputSize];
        outputs = new double[outputSize];
        zValues = new double[outputSize];

        Random random = new Random();
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = random.nextGaussian() * 0.01;
            }
            biases[i] = 0.0;
        }
    }

    public double[] forward(double[] input) {
        this.inputs = input;
        for (int i = 0; i < outputSize; i++) {
            zValues[i] = biases[i];
            for (int j = 0; j < inputSize; j++) {
                zValues[i] += weights[i][j] * input[j];
            }
            outputs[i] = ActivationFunction.sigmoid(zValues[i]);
        }
        return outputs;
    }
}
