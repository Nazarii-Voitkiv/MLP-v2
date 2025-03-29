import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.regex.*;

public class RecognizerApp extends JFrame {
    private static final int CANVAS_SIZE = 280;
    private static final int PIXEL_SIZE = 28;
    private static final String MODEL_PATH = "model.dat";
    private static final String DATA_DIR = "data";

    private DrawingPanel drawingPanel;
    private JLabel resultLabel;
    private JButton recognizeButton, clearButton, wrongLetterButton;
    private NeuralNetwork neuralNetwork;

    public RecognizerApp() {
        setTitle("Rozpoznawanie liter");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout(10, 10));

        ((JComponent) getContentPane()).setBorder(new EmptyBorder(10, 10, 10, 10));

        loadNeuralNetwork();

        drawingPanel = new DrawingPanel();

        JPanel centeringPanel = new JPanel(new GridBagLayout());
        centeringPanel.add(drawingPanel);
        centeringPanel.setBackground(Color.DARK_GRAY);

        resultLabel = new JLabel("Narysuj literę (M, O lub N)");
        resultLabel.setHorizontalAlignment(SwingConstants.CENTER);
        resultLabel.setFont(new Font(resultLabel.getFont().getName(), Font.BOLD, 18));
        resultLabel.setBorder(new EmptyBorder(10, 10, 10, 10));

        JPanel buttonPanel = new JPanel(new BorderLayout(10, 10));
        buttonPanel.setBorder(new EmptyBorder(10, 10, 10, 10));

        JPanel topButtonPanel = new JPanel(new GridLayout(1, 2, 10, 0));
        
        recognizeButton = new JButton("Rozpoznaj");
        clearButton = new JButton("Wyczyść");
        wrongLetterButton = new JButton("❌ To nie ta litera");

        configureButton(recognizeButton);
        configureButton(clearButton);
        configureButton(wrongLetterButton);

        recognizeButton.addActionListener(e -> recognizeDrawing());
        clearButton.addActionListener(e -> drawingPanel.clear());
        wrongLetterButton.addActionListener(e -> handleWrongLetter());

        topButtonPanel.add(recognizeButton);
        topButtonPanel.add(clearButton);

        JPanel bottomButtonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        bottomButtonPanel.add(wrongLetterButton);

        buttonPanel.add(topButtonPanel, BorderLayout.NORTH);
        buttonPanel.add(bottomButtonPanel, BorderLayout.SOUTH);

        add(resultLabel, BorderLayout.NORTH);
        add(centeringPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);

        // Increased window height by 50 pixels
        setSize(CANVAS_SIZE + 120, CANVAS_SIZE + 270);
        setLocationRelativeTo(null);
        setVisible(true);
    }
    
    private void configureButton(JButton button) {
        button.setFont(new Font(button.getFont().getName(), Font.BOLD, 14));
        button.setMargin(new Insets(10, 10, 10, 10));
    }
    
    private void loadNeuralNetwork() {
        try {
            neuralNetwork = new NeuralNetwork();
            neuralNetwork.loadModel(MODEL_PATH);
            System.out.println("Model został pomyślnie załadowany");
        } catch (IOException | ClassNotFoundException e) {
            JOptionPane.showMessageDialog(this, 
                "Błąd ładowania modelu: " + e.getMessage(), 
                "Błąd", JOptionPane.ERROR_MESSAGE);
            System.err.println("Błąd ładowania modelu: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void recognizeDrawing() {
        if (neuralNetwork == null) {
            resultLabel.setText("Błąd: model nie jest załadowany");
            return;
        }

        double[] imageData = drawingPanel.getBinarizedImage();
        imageData = ImageProcessor.centerImage(imageData);

        try {
            double[] rawOutputs = neuralNetwork.predict(imageData);

            int maxIndex = 0;
            double maxValue = rawOutputs[0];
            for (int i = 1; i < rawOutputs.length; i++) {
                if (rawOutputs[i] > maxValue) {
                    maxValue = rawOutputs[i];
                    maxIndex = i;
                }
            }

            char recognizedLetter = ' ';
            switch (maxIndex) {
                case 0: recognizedLetter = 'M'; break;
                case 1: recognizedLetter = 'O'; break;
                case 2: recognizedLetter = 'N'; break;
            }

            String confidenceLevel;
            if (maxValue >= 0.95) {
                confidenceLevel = "Bardzo wysoka pewność";
            } else if (maxValue >= 0.85) {
                confidenceLevel = "Wysoka pewność";
            } else if (maxValue >= 0.75) {
                confidenceLevel = "Dobra pewność";
            } else if (maxValue >= 0.6) {
                confidenceLevel = "Średnia pewność";
            } else if (maxValue >= 0.45) {
                confidenceLevel = "Niska pewność";
            } else {
                confidenceLevel = "Bardzo niska pewność";
            }
            
            resultLabel.setText(String.format("Rozpoznano literę: %c (%s)", recognizedLetter, confidenceLevel));
            
        } catch (Exception e) {
            resultLabel.setText("Błąd rozpoznawania: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void handleWrongLetter() {
        double[] imageData = drawingPanel.getBinarizedImage();
        boolean hasDrawing = false;
        for (double pixel : imageData) {
            if (pixel > 0.1) {
                hasDrawing = true;
                break;
            }
        }
        
        if (!hasDrawing) {
            JOptionPane.showMessageDialog(this, 
                "Nie ma nic do zapisania. Najpierw narysuj literę.", 
                "Błąd", JOptionPane.WARNING_MESSAGE);
            return;
        }

        Object[] options = { "M", "O", "N" };

        int choice = JOptionPane.showOptionDialog(this,
            "Jaka to była litera?",
            "Wybierz właściwą opcję",
            JOptionPane.DEFAULT_OPTION,
            JOptionPane.QUESTION_MESSAGE,
            null,
            options,
            options[0]);
            
        if (choice == JOptionPane.CLOSED_OPTION) {
            return;
        }

        char letter = ' ';
        switch (choice) {
            case 0: letter = 'M'; break;
            case 1: letter = 'O'; break;
            case 2: letter = 'N'; break;
        }
        
        try {
            imageData = ImageProcessor.centerImage(imageData);

            String fileName = generateUniqueFileName(letter);
            saveToCSV(imageData, fileName);

            JOptionPane.showMessageDialog(this,
                "✅ Zapisano przykład jako " + fileName,
                "Sukces",
                JOptionPane.INFORMATION_MESSAGE);
                
            resultLabel.setText("Zapisano jako " + fileName);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this,
                "Błąd podczas zapisywania: " + e.getMessage(),
                "Błąd",
                JOptionPane.ERROR_MESSAGE);
            e.printStackTrace();
        }
    }
    
    private String generateUniqueFileName(char letter) {
        int maxNumber = 0;
        File dataDir = new File(DATA_DIR);

        if (!dataDir.exists()) {
            dataDir.mkdirs();
        }
        
        File[] files = dataDir.listFiles();
        
        if (files != null) {
            Pattern pattern = Pattern.compile(letter + "_(\\d+)\\.csv");
            
            for (File file : files) {
                Matcher matcher = pattern.matcher(file.getName());
                if (matcher.matches()) {
                    int number = Integer.parseInt(matcher.group(1));
                    maxNumber = Math.max(maxNumber, number);
                }
            }
        }
        
        return String.format("%c_%03d.csv", letter, maxNumber + 1);
    }

    private void saveToCSV(double[] data, String fileName) throws IOException {
        File file = new File(DATA_DIR, fileName);
        
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

    private class DrawingPanel extends JPanel {
        private BufferedImage image;
        private Graphics2D g2d;

        public DrawingPanel() {
            setPreferredSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setMinimumSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setMaximumSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setBackground(Color.BLACK);
            setBorder(BorderFactory.createLineBorder(Color.GRAY, 2));

            image = new BufferedImage(PIXEL_SIZE, PIXEL_SIZE, BufferedImage.TYPE_BYTE_GRAY);
            g2d = image.createGraphics();
            g2d.setColor(Color.BLACK);
            g2d.fillRect(0, 0, PIXEL_SIZE, PIXEL_SIZE);
            g2d.setColor(Color.WHITE);

            MouseAdapter mouseHandler = new MouseAdapter() {
                @Override public void mousePressed(MouseEvent e) { draw(e.getX(), e.getY()); }
                @Override public void mouseDragged(MouseEvent e) { draw(e.getX(), e.getY()); }
            };

            addMouseListener(mouseHandler);
            addMouseMotionListener(mouseHandler);
        }

        private void draw(int x, int y) {
            int pixelX = x * PIXEL_SIZE / CANVAS_SIZE;
            int pixelY = y * PIXEL_SIZE / CANVAS_SIZE;

            g2d.fillRect(Math.max(0, pixelX - 1), Math.max(0, pixelY - 1),
                    Math.min(3, PIXEL_SIZE - pixelX + 1),
                    Math.min(3, PIXEL_SIZE - pixelY + 1));

            repaint();
        }

        public void clear() {
            g2d.setColor(Color.BLACK);
            g2d.fillRect(0, 0, PIXEL_SIZE, PIXEL_SIZE);
            g2d.setColor(Color.WHITE);
            repaint();
            resultLabel.setText("Panel wyczyszczony. Narysuj nową literę.");
        }

        public double[] getBinarizedImage() {
            double[] data = new double[PIXEL_SIZE * PIXEL_SIZE];

            for (int y = 0; y < PIXEL_SIZE; y++) {
                for (int x = 0; x < PIXEL_SIZE; x++) {
                    int rgb = image.getRGB(x, y) & 0xFF;
                    data[y * PIXEL_SIZE + x] = (rgb > 128) ? 1.0 : 0.0;
                }
            }

            return data;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            g.drawImage(image.getScaledInstance(CANVAS_SIZE, CANVAS_SIZE, Image.SCALE_SMOOTH), 0, 0, this);

            g.setColor(new Color(80, 80, 80));
            for (int i = 0; i <= PIXEL_SIZE; i++) {
                int pos = i * CANVAS_SIZE / PIXEL_SIZE;
                g.drawLine(pos, 0, pos, CANVAS_SIZE);
                g.drawLine(0, pos, CANVAS_SIZE, pos);
            }
        }
    }

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        SwingUtilities.invokeLater(RecognizerApp::new);
    }
}
