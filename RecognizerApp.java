import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.List;
import java.util.regex.*;

public class RecognizerApp extends JFrame {
    private static final int CANVAS_SIZE = 420; // Increased from 280
    private static final int PIXEL_SIZE = 28;
    private static final int INTERNAL_PIXEL_SIZE = 56;
    private static final String MODEL_PATH = "model.dat";
    private static final String DATA_DIR = "data";
    private static final String TEST_DATA_DIR = "test_data";
    private static final char[] LETTERS = {'M', 'O', 'N'};

    private DrawingPanel drawingPanel;
    private JLabel resultLabel;
    private JTextArea trainingAccuracyTextArea; // New field for training accuracy
    private JTextArea testAccuracyTextArea;     // Renamed from accuracyTextArea
    private JButton recognizeButton, clearButton, addToTrainingButton, addToTestingButton;
    private NeuralNetwork neuralNetwork;

    // Radio buttons for letter selection
    private JRadioButton radioM, radioO, radioN;
    private ButtonGroup letterGroup;

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (!new File(MODEL_PATH).exists()) {
            JOptionPane.showMessageDialog(null,
                "Błąd: Nie znaleziono pliku modelu '" + MODEL_PATH + "'.\n\n" +
                "Uruchom najpierw program Trainer, aby wytrenować i zapisać model.",
                "Brak pliku modelu", JOptionPane.ERROR_MESSAGE);
            System.exit(1);
            return;
        }
        
        SwingUtilities.invokeLater(RecognizerApp::new);
    }

    public RecognizerApp() {
        setTitle("Rozpoznawanie liter M O N");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(null);
        setSize(CANVAS_SIZE + 420, CANVAS_SIZE + 250); // Increased height further to accommodate panel at bottom
        setLocationRelativeTo(null);
        setResizable(false);
        
        if (!loadNeuralNetwork()) {
            dispose();
            return;
        }

        initializeUI();
        
        // Run model evaluation on both datasets
        evaluateModelOnTrainingData();
        evaluateModelOnTestData();
        
        setVisible(true);
    }
    
    private void initializeUI() {
        createDrawingPanel();
        createResultLabel();
        
        // First create the top row buttons
        int buttonWidth = 250;
        int buttonHeight = 50;
        int startX = CANVAS_SIZE + 40;
        int startY = CANVAS_SIZE - 250;
        int gap = 20;
        
        // First row - two buttons side by side
        clearButton = createButton("Wyczyść", e -> drawingPanel.clear(), 
            startX, startY, (buttonWidth - gap) / 2, buttonHeight);
            
        recognizeButton = createButton("Rozpoznaj", e -> recognizeDrawing(), 
            startX + (buttonWidth - gap) / 2 + gap, startY, (buttonWidth - gap) / 2, buttonHeight);
        
        add(clearButton);
        add(recognizeButton);
        
        // Now create radio buttons BETWEEN the button rows
        createRadioButtons(startX, startY + buttonHeight + 10);
        
        // Then create the bottom buttons
        createBottomButtons(startX, startY + buttonHeight + 80); // Added extra vertical space
        
        // Create accuracy panel at the bottom of the window
        createAccuracyPanel();
    }
    
    private void createDrawingPanel() {
        drawingPanel = new DrawingPanel();
        drawingPanel.setBounds(20, 20, CANVAS_SIZE, CANVAS_SIZE);
        add(drawingPanel);
    }
    
    private void createResultLabel() {
        resultLabel = new JLabel("<html>Narysuj literę (M, O lub N)</html>");
        resultLabel.setHorizontalAlignment(SwingConstants.CENTER);
        resultLabel.setFont(new Font(resultLabel.getFont().getName(), Font.BOLD, 18));
        resultLabel.setBounds(CANVAS_SIZE + 40, 20, 250, 80);
        add(resultLabel);
    }
    
    private void createAccuracyPanel() {
        // Create panel for training data accuracy (left side)
        JPanel trainingAccuracyPanel = new JPanel(new BorderLayout());
        trainingAccuracyPanel.setBorder(BorderFactory.createTitledBorder("Dokładność na danych treningowych"));
        trainingAccuracyPanel.setBounds(20, CANVAS_SIZE + 40, (CANVAS_SIZE + 380) / 2 - 10, 160);
        
        trainingAccuracyTextArea = new JTextArea();
        trainingAccuracyTextArea.setEditable(false);
        trainingAccuracyTextArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        trainingAccuracyTextArea.setBackground(new Color(240, 240, 240));
        
        JScrollPane trainingScrollPane = new JScrollPane(trainingAccuracyTextArea);
        trainingAccuracyPanel.add(trainingScrollPane, BorderLayout.CENTER);
        
        // Create panel for test data accuracy (right side)
        JPanel testAccuracyPanel = new JPanel(new BorderLayout());
        testAccuracyPanel.setBorder(BorderFactory.createTitledBorder("Dokładność na danych testowych"));
        testAccuracyPanel.setBounds(20 + (CANVAS_SIZE + 380) / 2 + 10, CANVAS_SIZE + 40, 
                                   (CANVAS_SIZE + 380) / 2 - 10, 160);
        
        testAccuracyTextArea = new JTextArea();
        testAccuracyTextArea.setEditable(false);
        testAccuracyTextArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        testAccuracyTextArea.setBackground(new Color(240, 240, 240));
        
        JScrollPane testScrollPane = new JScrollPane(testAccuracyTextArea);
        testAccuracyPanel.add(testScrollPane, BorderLayout.CENTER);
        
        add(trainingAccuracyPanel);
        add(testAccuracyPanel);
    }
    
    private void createRadioButtons(int startX, int yPos) {
        // Create a panel for radio buttons
        JPanel radioPanel = new JPanel();
        radioPanel.setLayout(new FlowLayout(FlowLayout.CENTER, 20, 0));
        radioPanel.setBorder(BorderFactory.createTitledBorder("Wybierz literę"));
        radioPanel.setBounds(startX, yPos, 250, 60);
        
        // Create radio buttons
        radioM = new JRadioButton("M");
        radioO = new JRadioButton("O");
        radioN = new JRadioButton("N");
        
        // Set font and size
        Font radioFont = new Font(radioM.getFont().getName(), Font.BOLD, 16);
        radioM.setFont(radioFont);
        radioO.setFont(radioFont);
        radioN.setFont(radioFont);
        
        // Group radio buttons to ensure only one is selected
        letterGroup = new ButtonGroup();
        letterGroup.add(radioM);
        letterGroup.add(radioO);
        letterGroup.add(radioN);
        
        // Add action listener to enable/disable buttons
        ActionListener radioListener = e -> updateButtonStates();
        radioM.addActionListener(radioListener);
        radioO.addActionListener(radioListener);
        radioN.addActionListener(radioListener);
        
        // Add radio buttons to panel
        radioPanel.add(radioM);
        radioPanel.add(radioO);
        radioPanel.add(radioN);
        
        add(radioPanel);
    }
    
    private void createBottomButtons(int startX, int startY) {
        int buttonWidth = 250;
        int buttonHeight = 50;
        int gap = 10;
        
        // Create the bottom buttons (add to training/testing)
        addToTrainingButton = createButton("Dodaj do ciągu uczącego", 
            e -> addToDataset(DATA_DIR), startX, startY, buttonWidth, buttonHeight);
        
        addToTestingButton = createButton("Dodaj do ciągu testowego", 
            e -> addToDataset(TEST_DATA_DIR), startX, startY + buttonHeight + gap, buttonWidth, buttonHeight);
        
        // Initially disable the add buttons until a radio button is selected
        addToTrainingButton.setEnabled(false);
        addToTestingButton.setEnabled(false);
        
        add(addToTrainingButton);
        add(addToTestingButton);
    }
    
    private JButton createButton(String text, ActionListener action, int x, int y, int width, int height) {
        JButton button = new JButton(text);
        button.setFont(new Font(button.getFont().getName(), Font.BOLD, 14));
        button.setMargin(new Insets(10, 10, 10, 10));
        button.addActionListener(action);
        button.setBounds(x, y, width, height);
        return button;
    }
    
    private JButton createButton(String text, ActionListener action, int x, int y) {
        return createButton(text, action, x, y, 250, 50);
    }
    
    private void updateButtonStates() {
        boolean isLetterSelected = radioM.isSelected() || radioO.isSelected() || radioN.isSelected();
        addToTrainingButton.setEnabled(isLetterSelected);
        addToTestingButton.setEnabled(isLetterSelected);
    }
    
    private char getSelectedLetter() {
        if (radioM.isSelected()) return 'M';
        if (radioO.isSelected()) return 'O';
        if (radioN.isSelected()) return 'N';
        return ' '; // This shouldn't happen if buttons are properly disabled
    }
    
    private void addToDataset(String dirName) {
        double[] imageData = drawingPanel.getBinarizedImage();
        if (!hasDrawing(imageData)) {
            JOptionPane.showMessageDialog(this, 
                "Nie ma nic do zapisania. Najpierw narysuj literę.", 
                "Błąd", JOptionPane.WARNING_MESSAGE);
            return;
        }

        char letter = getSelectedLetter();
        saveDrawingAsSample(letter, imageData, dirName);
    }
    
    private boolean loadNeuralNetwork() {
        try {
            neuralNetwork = new NeuralNetwork();
            neuralNetwork.loadModel(MODEL_PATH);
            System.out.println("Model został pomyślnie załadowany");
            return true;
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, 
                "Błąd ładowania modelu: " + e.getMessage() + 
                "\n\nAplikacja zostanie zamknięta.", 
                "Błąd krytyczny", JOptionPane.ERROR_MESSAGE);
            System.err.println("Błąd ładowania modelu: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    private void recognizeDrawing() {
        if (neuralNetwork == null) {
            resultLabel.setText("Błąd: model nie jest załadowany");
            return;
        }

        try {
            double[] imageData = ImageProcessor.centerImage(drawingPanel.getBinarizedImage());
            double[] rawOutputs = neuralNetwork.predict(imageData);
            int maxIndex = findMaxIndex(rawOutputs);
            char recognizedLetter = LETTERS[maxIndex];
            
            resultLabel.setText(String.format("<html>Rozpoznano literę:<br><b>%c</b> (%s)</html>", 
                               recognizedLetter, getConfidenceLevel(rawOutputs[maxIndex])));
            
        } catch (Exception e) {
            resultLabel.setText("Błąd rozpoznawania: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private int findMaxIndex(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];
        
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
    
    private String getConfidenceLevel(double value) {
        if (value >= 0.95) return "Bardzo wysoka pewność";
        if (value >= 0.85) return "Wysoka pewność";
        if (value >= 0.75) return "Dobra pewność";
        if (value >= 0.6) return "Średnia pewność";
        if (value >= 0.45) return "Niska pewność";
        return "Bardzo niska pewność";
    }
    
    private boolean hasDrawing(double[] imageData) {
        for (double pixel : imageData) {
            if (pixel > 0.1) {
                return true;
            }
        }
        return false;
    }
    
    private void saveDrawingAsSample(char letter, double[] imageData, String dirName) {
        try {
            imageData = ImageProcessor.centerImage(imageData);
            
            File dir = new File(dirName);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            
            String fileName = generateUniqueFileName(letter, dirName);
            saveToCSV(imageData, fileName, dirName);

            JOptionPane.showMessageDialog(this,
                "✅ Zapisano przykład jako " + fileName,
                "Sukces", JOptionPane.INFORMATION_MESSAGE);
                
            resultLabel.setText("Zapisano jako " + fileName);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this,
                "Błąd podczas zapisywania: " + e.getMessage(),
                "Błąd", JOptionPane.ERROR_MESSAGE);
            e.printStackTrace();
        }
    }
    
    private String generateUniqueFileName(char letter, String dirName) {
        int maxNumber = 0;
        File dataDir = new File(dirName);

        if (!dataDir.exists()) {
            dataDir.mkdirs();
        }
        
        File[] files = dataDir.listFiles();
        
        if (files != null) {
            Pattern pattern = Pattern.compile(letter + "_(\\d+)\\.csv");
            
            for (File file : files) {
                Matcher matcher = pattern.matcher(file.getName());
                if (matcher.matches()) {
                    maxNumber = Math.max(maxNumber, Integer.parseInt(matcher.group(1)));
                }
            }
        }
        
        return String.format("%c_%03d.csv", letter, maxNumber + 1);
    }

    private void saveToCSV(double[] data, String fileName, String dirName) throws IOException {
        File file = new File(dirName, fileName);
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(file))) {
            StringBuilder sb = new StringBuilder();
            
            for (int i = 0; i < data.length; i++) {
                sb.append(data[i]);
                if (i < data.length - 1) {
                    sb.append(",");
                }
            }
            
            writer.write(sb.toString());
        }
    }

    private void evaluateModelOnTrainingData() {
        if (neuralNetwork == null) {
            trainingAccuracyTextArea.setText("Model nie został załadowany");
            return;
        }
        
        // Load training samples
        List<Sample> trainingSamples = loadTrainingSamples();
        if (trainingSamples.isEmpty()) {
            trainingAccuracyTextArea.setText("Brak danych treningowych w folderze " + DATA_DIR);
            return;
        }
        
        // Calculate accuracy using the same structure as the test data method
        int[] correctPredictions = new int[LETTERS.length];
        int[] totalSamples = new int[LETTERS.length];
        
        for (Sample sample : trainingSamples) {
            double[] prediction = neuralNetwork.predict(sample.getInput());
            int predictedIndex = findMaxIndex(prediction);
            int targetIndex = findMaxIndex(sample.getTarget());
            
            totalSamples[targetIndex]++;
            if (predictedIndex == targetIndex) {
                correctPredictions[targetIndex]++;
            }
        }
        
        // Calculate overall accuracy
        int totalCorrect = Arrays.stream(correctPredictions).sum();
        int total = Arrays.stream(totalSamples).sum();
        double overallAccuracy = total > 0 ? (double) totalCorrect / total * 100 : 0;
        
        // Display results
        StringBuilder sb = new StringBuilder();
        sb.append("Dokładność rozpoznawania:\n");
        sb.append(String.format("Ogólna: %.2f%% (%d/%d)\n", 
                overallAccuracy, totalCorrect, total));
        
        for (int i = 0; i < LETTERS.length; i++) {
            double accuracy = totalSamples[i] > 0 ? 
                (double) correctPredictions[i] / totalSamples[i] * 100 : 0;
            sb.append(String.format("Litera %c: %.2f%% (%d/%d)\n", 
                    LETTERS[i], accuracy, correctPredictions[i], totalSamples[i]));
        }
        
        trainingAccuracyTextArea.setText(sb.toString());
    }

    private void evaluateModelOnTestData() {
        if (neuralNetwork == null) {
            testAccuracyTextArea.setText("Model nie został załadowany");
            return;
        }
        
        // Load test samples
        List<Sample> testSamples = loadTestSamples();
        if (testSamples.isEmpty()) {
            testAccuracyTextArea.setText("Brak danych testowych w folderze " + TEST_DATA_DIR);
            return;
        }
        
        // Calculate accuracy
        int[] correctPredictions = new int[LETTERS.length];
        int[] totalSamples = new int[LETTERS.length];
        
        for (Sample sample : testSamples) {
            double[] prediction = neuralNetwork.predict(sample.getInput());
            int predictedIndex = findMaxIndex(prediction);
            int targetIndex = findMaxIndex(sample.getTarget());
            
            totalSamples[targetIndex]++;
            if (predictedIndex == targetIndex) {
                correctPredictions[targetIndex]++;
            }
        }
        
        // Calculate overall accuracy
        int totalCorrect = Arrays.stream(correctPredictions).sum();
        int total = Arrays.stream(totalSamples).sum();
        double overallAccuracy = total > 0 ? (double) totalCorrect / total * 100 : 0;
        
        // Display results
        StringBuilder sb = new StringBuilder();
        sb.append("Dokładność rozpoznawania:\n");
        sb.append(String.format("Ogólna: %.2f%% (%d/%d poprawnych)\n", 
                overallAccuracy, totalCorrect, total));
        
        for (int i = 0; i < LETTERS.length; i++) {
            double accuracy = totalSamples[i] > 0 ? 
                (double) correctPredictions[i] / totalSamples[i] * 100 : 0;
            sb.append(String.format("Litera %c: %.2f%% (%d/%d poprawnych)\n", 
                    LETTERS[i], accuracy, correctPredictions[i], totalSamples[i]));
        }
        
        testAccuracyTextArea.setText(sb.toString());
    }
    
    private List<Sample> loadTrainingSamples() {
        List<Sample> samples = new ArrayList<>();
        File dataDir = new File(DATA_DIR);
        
        if (!dataDir.exists() || !dataDir.isDirectory()) {
            return samples;
        }
        
        File[] files = dataDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".csv"));
        
        if (files == null || files.length == 0) {
            return samples;
        }
        
        Pattern pattern = Pattern.compile("([MON])_(\\d+)\\.csv");
        
        for (File file : files) {
            try {
                Matcher matcher = pattern.matcher(file.getName());
                if (!matcher.matches()) {
                    continue;
                }
                
                char letter = matcher.group(1).charAt(0);
                double[] target = new double[3];
                switch (letter) {
                    case 'M': target[0] = 1.0; break;
                    case 'O': target[1] = 1.0; break;
                    case 'N': target[2] = 1.0; break;
                    default: continue;
                }
                
                String content = new String(Files.readAllBytes(file.toPath()));
                String[] values = content.trim().split(",");
                
                if (values.length != 784) {
                    continue;
                }
                
                double[] input = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    input[i] = Double.parseDouble(values[i]);
                }
                
                samples.add(new Sample(input, target));
                
            } catch (Exception e) {
                System.err.println("Error loading training sample: " + e.getMessage());
            }
        }
        
        System.out.println("Loaded " + samples.size() + " training samples");
        return samples;
    }

    private List<Sample> loadTestSamples() {
        List<Sample> samples = new ArrayList<>();
        File testDir = new File(TEST_DATA_DIR);
        
        if (!testDir.exists() || !testDir.isDirectory()) {
            return samples;
        }
        
        File[] files = testDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".csv"));
        
        if (files == null || files.length == 0) {
            return samples;
        }
        
        Pattern pattern = Pattern.compile("([MON])_(\\d+)\\.csv");
        
        for (File file : files) {
            try {
                Matcher matcher = pattern.matcher(file.getName());
                if (!matcher.matches()) {
                    continue;
                }
                
                char letter = matcher.group(1).charAt(0);
                double[] target = new double[3];
                switch (letter) {
                    case 'M': target[0] = 1.0; break;
                    case 'O': target[1] = 1.0; break;
                    case 'N': target[2] = 1.0; break;
                    default: continue;
                }
                
                String content = new String(Files.readAllBytes(file.toPath()));
                String[] values = content.trim().split(",");
                
                if (values.length != 784) {
                    continue;
                }
                
                double[] input = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    input[i] = Double.parseDouble(values[i]);
                }
                
                samples.add(new Sample(input, target));
                
            } catch (Exception e) {
                System.err.println("Error loading test sample: " + e.getMessage());
            }
        }
        
        System.out.println("Loaded " + samples.size() + " test samples");
        return samples;
    }

    private class DrawingPanel extends JPanel {
        private BufferedImage image;
        private Graphics2D g2d;
        private int lastX = -1, lastY = -1;

        public DrawingPanel() {
            initPanel();
            initDrawingSurface();
            setupMouseHandlers();
        }
        
        private void initPanel() {
            setPreferredSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setMinimumSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setMaximumSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setBackground(Color.WHITE);
            setBorder(BorderFactory.createLineBorder(Color.GRAY, 2));
        }
        
        private void initDrawingSurface() {
            image = new BufferedImage(INTERNAL_PIXEL_SIZE, INTERNAL_PIXEL_SIZE, BufferedImage.TYPE_BYTE_GRAY);
            g2d = image.createGraphics();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, INTERNAL_PIXEL_SIZE, INTERNAL_PIXEL_SIZE);
            g2d.setColor(Color.BLACK);
        }
        
        private void setupMouseHandlers() {
            MouseAdapter mouseHandler = new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent e) {
                    lastX = e.getX();
                    lastY = e.getY();
                    drawPoint(lastX, lastY);
                }
                
                @Override
                public void mouseDragged(MouseEvent e) {
                    int x = e.getX();
                    int y = e.getY();
                    
                    if (lastX != -1 && lastY != -1) {
                        drawLine(lastX, lastY, x, y);
                    }
                    
                    lastX = x;
                    lastY = y;
                }
                
                @Override
                public void mouseReleased(MouseEvent e) {
                    lastX = -1;
                    lastY = -1;
                }
            };

            addMouseListener(mouseHandler);
            addMouseMotionListener(mouseHandler);
        }

        private void drawPoint(int x, int y) {
            int pixelX = x * INTERNAL_PIXEL_SIZE / CANVAS_SIZE;
            int pixelY = y * INTERNAL_PIXEL_SIZE / CANVAS_SIZE;

            g2d.fillOval(pixelX - 2, pixelY - 2, 5, 5);
            repaint();
        }
        
        private void drawLine(int x1, int y1, int x2, int y2) {
            int pixelX1 = x1 * INTERNAL_PIXEL_SIZE / CANVAS_SIZE;
            int pixelY1 = y1 * INTERNAL_PIXEL_SIZE / CANVAS_SIZE;
            int pixelX2 = x2 * INTERNAL_PIXEL_SIZE / CANVAS_SIZE;
            int pixelY2 = y2 * INTERNAL_PIXEL_SIZE / CANVAS_SIZE;

            g2d.setStroke(new BasicStroke(4, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
            g2d.drawLine(pixelX1, pixelY1, pixelX2, pixelY2);
            repaint();
        }

        public void clear() {
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, INTERNAL_PIXEL_SIZE, INTERNAL_PIXEL_SIZE);
            g2d.setColor(Color.BLACK);
            repaint();
            resultLabel.setText("<html>Panel wyczyszczony.<br>Narysuj nową literę.</html>");
        }

        public double[] getBinarizedImage() {
            double[] data = new double[PIXEL_SIZE * PIXEL_SIZE];
            for (int y = 0; y < PIXEL_SIZE; y++) {
                for (int x = 0; x < PIXEL_SIZE; x++) {
                    data[y * PIXEL_SIZE + x] = (getPixelSum(x, y) / 4 < 128) ? 1.0 : 0.0;
                }
            }
            return data;
        }
        
        private int getPixelSum(int x, int y) {
            int startX = x * 2;
            int startY = y * 2;
            int sum = 0;
            
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int highResX = startX + dx;
                    int highResY = startY + dy;
                    if (highResX < INTERNAL_PIXEL_SIZE && highResY < INTERNAL_PIXEL_SIZE) {
                        sum += image.getRGB(highResX, highResY) & 0xFF;
                    }
                }
            }
            
            return sum;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawImage(image.getScaledInstance(CANVAS_SIZE, CANVAS_SIZE, Image.SCALE_SMOOTH), 0, 0, this);
        }
    }
}
