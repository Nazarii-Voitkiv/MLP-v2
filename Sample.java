public class Sample {
    private double[] input;
    private double[] target;

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
