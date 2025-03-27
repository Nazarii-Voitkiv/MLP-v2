import java.util.Arrays;

public class ActivationFunction {

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    public static double[] softmax(double[] z) {
        double max = Arrays.stream(z).max().orElse(0);
        double sum = 0.0;
        double[] result = new double[z.length];

        for (int i = 0; i < z.length; i++) {
            result[i] = Math.exp(z[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < z.length; i++) {
            result[i] /= sum;
        }

        return result;
    }
}
