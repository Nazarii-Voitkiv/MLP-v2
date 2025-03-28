public class Sample {
    private double[] input;  // 784 elements (28x28 pixels)
    private double[] target; // 3 elements for one-hot encoding [M, O, N]

    public Sample(double[] input, double[] target) {
        this.input = input;
        this.target = target;
    }

    public double[] getInput() {
        return input;
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public double[] getTarget() {
        return target;
    }

    public void setTarget(double[] target) {
        this.target = target;
    }
}
