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
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath), 16384)) { // Збільшуємо розмір буфера
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
                        // Оптимізація: перевіряємо чи не порожній рядок перед парсингом
                        String val = values[i + 1].trim();
                        if (!val.isEmpty()) {
                            imageData[i] = Double.parseDouble(val) / 255.0;
                        } else {
                            imageData[i] = 0.0;
                        }
                    } catch (NumberFormatException e) {
                        imageData[i] = 0.0; // Default value in case of parsing error
                    }
                }
                imagesList.add(imageData);
                lineCount++;
                
                // Periodic garbage collection to help with memory - збільшуємо частоту
                if (lineCount % 50 == 0) {
                    System.out.println("Завантажено " + lineCount + " зразків...");
                    // Знищуємо непотрібні об'єкти для економії пам'яті
                    values = null;
                    line = null;
                    System.gc(); // Request garbage collection
                }
            }
        }
        
        // Convert lists to arrays
        System.out.println("Конвертація даних у масиви...");
        images = new double[imagesList.size()][];
        labels = new double[labelsList.size()][];
        
        for (int i = 0; i < imagesList.size(); i++) {
            images[i] = imagesList.get(i);
            labels[i] = labelsList.get(i);
        }
        
        // Звільнення пам'яті після конвертації
        imagesList.clear();
        labelsList.clear();
        System.gc();
        
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

    // Додайте новий метод для балансування даних
    public void loadBalancedDataFromCSV(String filePath, int samplesPerClass) throws IOException {
        List<double[]> imagesM = new ArrayList<>();
        List<double[]> labelsM = new ArrayList<>();
        List<double[]> imagesO = new ArrayList<>();
        List<double[]> labelsO = new ArrayList<>();
        List<double[]> imagesN = new ArrayList<>();
        List<double[]> labelsN = new ArrayList<>();
        
        int countM = 0, countO = 0, countN = 0;
        int pixelCount = 16384; // Constant for 128x128 pixels
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null && 
                   (countM < samplesPerClass || countO < samplesPerClass || countN < samplesPerClass)) {
                
                String[] values = line.split(",");
                if (values.length < pixelCount + 1) continue;
                
                String label = values[0];
                
                // Пропускаємо, якщо вже досягнуто ліміт для цього класу
                if ((label.equals("M") && countM >= samplesPerClass) ||
                    (label.equals("O") && countO >= samplesPerClass) ||
                    (label.equals("N") && countN >= samplesPerClass)) {
                    continue;
                }
                
                // Створюємо масив пікселів
                double[] imageData = new double[pixelCount];
                for (int i = 0; i < pixelCount && i + 1 < values.length; i++) {
                    try {
                        String val = values[i + 1].trim();
                        if (!val.isEmpty()) {
                            imageData[i] = Double.parseDouble(val) / 255.0;
                        } else {
                            imageData[i] = 0.0;
                        }
                    } catch (NumberFormatException e) {
                        imageData[i] = 0.0; // Default value in case of parsing error
                    }
                }
                
                // Додаємо до відповідного списку
                double[] oneHotLabel = convertToOneHot(label);
                
                if (label.equals("M")) {
                    imagesM.add(imageData);
                    labelsM.add(oneHotLabel);
                    countM++;
                    if (countM % 100 == 0) System.out.println("Завантажено M: " + countM);
                } else if (label.equals("O")) {
                    imagesO.add(imageData);
                    labelsO.add(oneHotLabel);
                    countO++;
                    if (countO % 100 == 0) System.out.println("Завантажено O: " + countO);
                } else if (label.equals("N")) {
                    imagesN.add(imageData);
                    labelsN.add(oneHotLabel);
                    countN++;
                    if (countN % 100 == 0) System.out.println("Завантажено N: " + countN);
                }
            }
        }
        
        // Об'єднуємо списки
        List<double[]> allImages = new ArrayList<>();
        List<double[]> allLabels = new ArrayList<>();
        
        allImages.addAll(imagesM);
        allImages.addAll(imagesO);
        allImages.addAll(imagesN);
        
        allLabels.addAll(labelsM);
        allLabels.addAll(labelsO);
        allLabels.addAll(labelsN);
        
        // Конвертуємо у масиви
        images = allImages.toArray(new double[0][]);
        labels = allLabels.toArray(new double[0][]);
        
        System.out.println("Завантажено балансований набір даних:");
        System.out.println("M: " + countM + ", O: " + countO + ", N: " + countN);
    }
}
