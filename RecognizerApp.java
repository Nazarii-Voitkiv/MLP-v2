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
    private static final int CANVAS_SIZE = 420;
    private static final int PIXEL_SIZE = 28;
    private static final int INTERNAL_PIXEL_SIZE = 56;
    private static final String MODEL_PATH = "model.dat";
    private static final String DATA_DIR = "data";
    private static final String TEST_DATA_DIR = "test_data";
    private static final char[] LETTERS = {'M', 'O', 'N'};

    private DrawingPanel drawingPanel;
    private JLabel resultLabel;
    private JTextArea trainingAccuracyTextArea;
    private JTextArea testAccuracyTextArea;
    private JButton recognizeButton, clearButton, addToTrainingButton, addToTestingButton;
    private NeuralNetwork neuralNetwork;
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
        // Збільшуємо розмір вікна, щоб додати місце для відступу між частинами
        setSize(CANVAS_SIZE * 2 + 120, CANVAS_SIZE + 300);
        setLocationRelativeTo(null);
        setResizable(false);
        
        if (!loadNeuralNetwork()) {
            dispose();
            return;
        }

        initializeUI();
        
        evaluateModel(DATA_DIR, trainingAccuracyTextArea);
        evaluateModel(TEST_DATA_DIR, testAccuracyTextArea);
        
        setVisible(true);
    }
    
    private void initializeUI() {
        // Звичайний бічний відступ
        int margin = 40;
        // Додатковий відступ між лівою та правою частиною
        int centerGap = 40;
        
        createDrawingPanel(margin);
        
        // Розраховуємо позицію для правої частини, додаючи додатковий відступ centerGap
        int rightPanelX = CANVAS_SIZE + margin + centerGap;
        
        // Змінюємо ширину правої панелі, щоб дорівнювала CANVAS_SIZE
        int rightPanelWidth = CANVAS_SIZE;
        
        createResultLabel(rightPanelX, rightPanelWidth, margin);

        int buttonHeight = 60;
        int gap = 10; // Зменшуємо gap для кращої пропорції кнопок
        int startY = 120;

        // Змінюємо розмір кнопок, щоб вони разом з gap дорівнювали CANVAS_SIZE
        int buttonWidth = (rightPanelWidth - gap) / 2;

        clearButton = createButton("Wyczyść", e -> drawingPanel.clear(), 
            rightPanelX, startY, buttonWidth, buttonHeight);
            
        recognizeButton = createButton("Rozpoznaj", e -> recognizeDrawing(), 
            rightPanelX + buttonWidth + gap, startY, buttonWidth, buttonHeight);
        
        add(clearButton);
        add(recognizeButton);

        createRadioButtons(rightPanelX, startY + buttonHeight + gap);
        createBottomButtons(rightPanelX, startY + buttonHeight + gap + 80);
        
        // Передаємо centerGap для врахування у нижній панелі
        createAccuracyPanel(margin, centerGap);
    }
    
    private void createDrawingPanel(int margin) {
        drawingPanel = new DrawingPanel();
        drawingPanel.setBounds(margin, margin, CANVAS_SIZE, CANVAS_SIZE); // Use consistent margin
        add(drawingPanel);
    }
    
    private void createResultLabel(int rightPanelX, int rightPanelWidth, int margin) {
        resultLabel = new JLabel("<html>Narysuj literę (M, O lub N)</html>");
        resultLabel.setHorizontalAlignment(SwingConstants.CENTER);
        resultLabel.setFont(new Font(resultLabel.getFont().getName(), Font.BOLD, 20));
        
        // Змінюємо ширину мітки, щоб дорівнювала CANVAS_SIZE
        resultLabel.setBounds(rightPanelX, margin, rightPanelWidth, 80);
        add(resultLabel);
    }
    
    private void createAccuracyPanel(int margin, int centerGap) {
        // Встановлюємо ширину панелей рівною ширині квадрата для малювання
        int panelWidth = CANVAS_SIZE; // Змінено з CANVAS_SIZE - centerGap/2 на повний розмір CANVAS_SIZE
        int panelHeight = 160;
        int startY = CANVAS_SIZE + 60;
        
        // Ліва панель
        JPanel trainingAccuracyPanel = new JPanel(new BorderLayout());
        trainingAccuracyPanel.setBorder(BorderFactory.createTitledBorder("Dokładność na danych treningowych"));
        trainingAccuracyPanel.setBounds(margin, startY, panelWidth, panelHeight);
        
        trainingAccuracyTextArea = createAccuracyTextArea();
        JScrollPane trainingScrollPane = new JScrollPane(trainingAccuracyTextArea);
        trainingAccuracyPanel.add(trainingScrollPane, BorderLayout.CENTER);
        
        // Права панель з додатковим відступом
        JPanel testAccuracyPanel = new JPanel(new BorderLayout());
        testAccuracyPanel.setBorder(BorderFactory.createTitledBorder("Dokładność na danych testowych"));
        // Додаємо centerGap, щоб відсунути праву панель
        testAccuracyPanel.setBounds(CANVAS_SIZE + margin + centerGap, startY, panelWidth, panelHeight);
        
        testAccuracyTextArea = createAccuracyTextArea();
        JScrollPane testScrollPane = new JScrollPane(testAccuracyTextArea);
        testAccuracyPanel.add(testScrollPane, BorderLayout.CENTER);
        
        add(trainingAccuracyPanel);
        add(testAccuracyPanel);
    }
    
    private JTextArea createAccuracyTextArea() {
        JTextArea textArea = new JTextArea();
        textArea.setEditable(false);
        textArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        textArea.setBackground(new Color(240, 240, 240));
        return textArea;
    }
    
    private void createRadioButtons(int startX, int yPos) {
        JPanel radioPanel = new JPanel();
        radioPanel.setLayout(new FlowLayout(FlowLayout.CENTER, 30, 10));
        radioPanel.setBorder(BorderFactory.createTitledBorder("Wybierz literę"));

        int panelWidth = CANVAS_SIZE; // Make panel width match canvas width
        radioPanel.setBounds(startX, yPos, panelWidth, 70);
        
        radioM = new JRadioButton("M");
        radioO = new JRadioButton("O");
        radioN = new JRadioButton("N");
        
        Font radioFont = new Font(radioM.getFont().getName(), Font.BOLD, 20); // Larger font
        radioM.setFont(radioFont);
        radioO.setFont(radioFont);
        radioN.setFont(radioFont);
        
        letterGroup = new ButtonGroup();
        letterGroup.add(radioM);
        letterGroup.add(radioO);
        letterGroup.add(radioN);
        
        ActionListener radioListener = e -> updateButtonStates();
        radioM.addActionListener(radioListener);
        radioO.addActionListener(radioListener);
        radioN.addActionListener(radioListener);
        
        radioPanel.add(radioM);
        radioPanel.add(radioO);
        radioPanel.add(radioN);
        
        add(radioPanel);
    }
    
    private void createBottomButtons(int startX, int startY) {
        int buttonWidth = CANVAS_SIZE; // Make buttons the same width as canvas
        int buttonHeight = 60;
        int gap = 15;

        addToTrainingButton = createButton("Dodaj do ciągu uczącego", 
            e -> addToDataset(DATA_DIR), startX, startY, buttonWidth, buttonHeight);
        
        addToTestingButton = createButton("Dodaj do ciągu testowego", 
            e -> addToDataset(TEST_DATA_DIR), startX, startY + buttonHeight + gap, buttonWidth, buttonHeight);
        
        addToTrainingButton.setEnabled(false);
        addToTestingButton.setEnabled(false);
        
        add(addToTrainingButton);
        add(addToTestingButton);
    }
    
    private JButton createButton(String text, ActionListener action, int x, int y, int width, int height) {
        JButton button = new JButton(text);
        button.setFont(new Font(button.getFont().getName(), Font.BOLD, 16)); // Larger font for all buttons
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
        return ' ';
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
            return true;
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, 
                "Błąd ładowania modelu: " + e.getMessage() + 
                "\n\nAplikacja zostanie zamknięta.", 
                "Błąd krytyczny", JOptionPane.ERROR_MESSAGE);
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
            
            if(dirName.equals(DATA_DIR)) {
                evaluateModel(DATA_DIR, trainingAccuracyTextArea);
            } else if(dirName.equals(TEST_DATA_DIR)) {
                evaluateModel(TEST_DATA_DIR, testAccuracyTextArea);
            }
            
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

    private void evaluateModel(String dirPath, JTextArea outputArea) {
        if (neuralNetwork == null) {
            outputArea.setText("Model nie został załadowany");
            return;
        }
        
        List<Sample> samples = loadSamples(dirPath);
        if (samples.isEmpty()) {
            outputArea.setText("Brak danych w folderze " + dirPath);
            return;
        }
        
        int[] correctPredictions = new int[LETTERS.length];
        int[] totalSamples = new int[LETTERS.length];
        
        for (Sample sample : samples) {
            double[] prediction = neuralNetwork.predict(sample.getInput());
            int predictedIndex = findMaxIndex(prediction);
            int targetIndex = findMaxIndex(sample.getTarget());
            
            totalSamples[targetIndex]++;
            if (predictedIndex == targetIndex) {
                correctPredictions[targetIndex]++;
            }
        }
        
        int totalCorrect = Arrays.stream(correctPredictions).sum();
        int total = Arrays.stream(totalSamples).sum();
        double overallAccuracy = total > 0 ? (double) totalCorrect / total * 100 : 0;
        
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
        
        outputArea.setText(sb.toString());
    }

    private List<Sample> loadSamples(String dirPath) {
        List<Sample> samples = new ArrayList<>();
        File dataDir = new File(dirPath);
        
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
                
                double[] input = parseCsvFile(file);
                if (input.length != 784) continue;
                
                samples.add(new Sample(input, target));
                
            } catch (Exception e) {

            }
        }
        
        return samples;
    }
    
    private double[] parseCsvFile(File file) throws IOException {
        String content = new String(Files.readAllBytes(file.toPath()));
        String[] values = content.trim().split(",");
        double[] input = new double[values.length];
        
        for (int i = 0; i < values.length; i++) {
            input[i] = Double.parseDouble(values[i]);
        }
        
        return input;
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

