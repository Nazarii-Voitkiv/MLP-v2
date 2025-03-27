import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataProcessor {
    // Arrays to store the processed data
    private double[][] images;
    private double[][] labels;
    
    /**
     * Read and process data from a CSV file
     * @param filePath path to the CSV file
     * @throws IOException if an I/O error occurs
     */
    public void loadDataFromCSV(String filePath) throws IOException {
        loadDataFromCSV(filePath, -1); // Load all samples
    }
    
    /**
     * Read and process data from a CSV file with sample limit
     * @param filePath path to the CSV file
     * @param maxSamples maximum number of samples to load (-1 for all)
     * @throws IOException if an I/O error occurs
     */
    public void loadDataFromCSV(String filePath, int maxSamples) throws IOException {
        List<double[]> imagesList = new ArrayList<>();
        List<double[]> labelsList = new ArrayList<>();
        
        int lineCount = 0;
        int pixelCount = 16384; // Constant for 128x128 pixels
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath), 8192)) {
            String line;
            while ((line = br.readLine()) != null && (maxSamples == -1 || lineCount < maxSamples)) {
                String[] values = line.split(",");
                
                if (values.length < pixelCount + 1) {
                    System.out.println("Попередження: Рядок " + lineCount + " містить менше значень, ніж очікувалося");
                    continue;
                }
                
                // Get the label and convert to one-hot encoding
                String label = values[0];
                double[] oneHotLabel = convertToOneHot(label);
                labelsList.add(oneHotLabel);
                
                // Get the image data and normalize
                double[] imageData = new double[pixelCount];
                for (int i = 0; i < pixelCount && i + 1 < values.length; i++) {
                    try {
                        // Convert string to double and normalize (divide by 255)
                        imageData[i] = Double.parseDouble(values[i + 1]) / 255.0;
                    } catch (NumberFormatException e) {
                        System.out.println("Помилка конвертації у рядку " + lineCount + 
                                          ", піксель " + i + ": " + values[i + 1]);
                        imageData[i] = 0.0; // Default value in case of parsing error
                    }
                }
                imagesList.add(imageData);
                lineCount++;
                
                // Periodic garbage collection to help with memory
                if (lineCount % 100 == 0) {
                    System.out.println("Завантажено " + lineCount + " зразків...");
                    System.gc(); // Request garbage collection
                }
            }
        }
        
        // Convert lists to arrays
        images = new double[imagesList.size()][];
        labels = new double[labelsList.size()][];
        
        for (int i = 0; i < imagesList.size(); i++) {
            images[i] = imagesList.get(i);
            labels[i] = labelsList.get(i);
        }
        
        System.out.println("Завантажено всього " + images.length + " зразків з мітками");
    }
    
    /**
     * Convert label to one-hot encoding
     * M → [1, 0, 0]
     * O → [0, 1, 0]
     * N → [0, 0, 1]
     * @param label the label to convert
     * @return one-hot encoded array
     */
    private double[] convertToOneHot(String label) {
        double[] oneHot = new double[3];
        
        switch (label) {
            case "M":
                oneHot[0] = 1.0;
                break;
            case "O":
                oneHot[1] = 1.0;
                break;
            case "N":
                oneHot[2] = 1.0;
                break;
            default:
                throw new IllegalArgumentException("Invalid label: " + label);
        }
        
        return oneHot;
    }
    
    /**
     * Get the processed image data
     * @return 2D array of normalized images
     */
    public double[][] getImages() {
        return images;
    }
    
    /**
     * Get the one-hot encoded labels
     * @return 2D array of one-hot encoded labels
     */
    public double[][] getLabels() {
        return labels;
    }
    
    /**
     * Get information about the dataset
     * @return String with dataset information
     */
    public String getDataInfo() {
        if (images == null || labels == null) {
            return "No data loaded";
        }
        return "Dataset contains " + images.length + " samples, each with " + 
               images[0].length + " features and " + labels[0].length + " classes";
    }
}
