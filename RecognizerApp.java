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

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }

        File modelFile = new File(MODEL_PATH);
        if (!modelFile.exists()) {
            JOptionPane.showMessageDialog(null,
                "Błąd: Nie znaleziono pliku modelu '" + MODEL_PATH + "'.\n\n" +
                "Uruchom najpierw program Trainer, aby wytrenować i zapisać model.",
                "Brak pliku modelu",
                JOptionPane.ERROR_MESSAGE);
            System.exit(1);
            return;
        }
        
        SwingUtilities.invokeLater(RecognizerApp::new);
    }

    public RecognizerApp() {
        setTitle("Rozpoznawanie liter");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        setLayout(null);

        if (!loadNeuralNetwork()) {
            dispose();
            return;
        }

        drawingPanel = new DrawingPanel();
        drawingPanel.setBounds(20, 20, CANVAS_SIZE, CANVAS_SIZE);
        add(drawingPanel);

        resultLabel = new JLabel("<html>Narysuj literę (M, O lub N)</html>");
        resultLabel.setHorizontalAlignment(SwingConstants.CENTER);
        resultLabel.setFont(new Font(resultLabel.getFont().getName(), Font.BOLD, 18));
        resultLabel.setBounds(CANVAS_SIZE + 40, 20, 250, 80);
        add(resultLabel);

        recognizeButton = new JButton("Rozpoznaj");
        clearButton = new JButton("Wyczyść");
        wrongLetterButton = new JButton("❌ To nie ta litera");

        configureButton(recognizeButton);
        configureButton(clearButton);
        configureButton(wrongLetterButton);

        recognizeButton.setBounds(CANVAS_SIZE + 40, CANVAS_SIZE - 180, 250, 50);
        clearButton.setBounds(CANVAS_SIZE + 40, CANVAS_SIZE - 120, 250, 50);
        wrongLetterButton.setBounds(CANVAS_SIZE + 40, CANVAS_SIZE - 60, 250, 50);

        recognizeButton.addActionListener(e -> recognizeDrawing());
        clearButton.addActionListener(e -> drawingPanel.clear());
        wrongLetterButton.addActionListener(e -> handleWrongLetter());

        add(recognizeButton);
        add(clearButton);
        add(wrongLetterButton);

        setSize(CANVAS_SIZE + 320, CANVAS_SIZE + 80);
        setLocationRelativeTo(null);
        setResizable(false);
        setVisible(true);
    }
    
    private void configureButton(JButton button) {
        button.setFont(new Font(button.getFont().getName(), Font.BOLD, 14));
        button.setMargin(new Insets(10, 10, 10, 10));
    }
    
    private boolean loadNeuralNetwork() {
        try {
            neuralNetwork = new NeuralNetwork();
            neuralNetwork.loadModel(MODEL_PATH);
            System.out.println("Model został pomyślnie załadowany");
            return true;
        } catch (IOException | ClassNotFoundException e) {
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
            
            resultLabel.setText(String.format("<html>Rozpoznano literę:<br><b>%c</b> (%s)</html>", 
                               recognizedLetter, confidenceLevel));
            
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
        private static final int INTERNAL_PIXEL_SIZE = 56;
        private BufferedImage image;
        private Graphics2D g2d;

        public DrawingPanel() {
            setPreferredSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setMinimumSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setMaximumSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setBackground(Color.WHITE);
            setBorder(BorderFactory.createLineBorder(Color.GRAY, 2));

            image = new BufferedImage(INTERNAL_PIXEL_SIZE, INTERNAL_PIXEL_SIZE, BufferedImage.TYPE_BYTE_GRAY);
            g2d = image.createGraphics();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, INTERNAL_PIXEL_SIZE, INTERNAL_PIXEL_SIZE);
            g2d.setColor(Color.BLACK);

            MouseAdapter mouseHandler = new MouseAdapter() {
                private int lastX = -1;
                private int lastY = -1;
                
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
                    
                    int average = sum / 4;

                    data[y * PIXEL_SIZE + x] = (average < 128) ? 1.0 : 0.0;
                }
            }

            return data;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            g.drawImage(image.getScaledInstance(CANVAS_SIZE, CANVAS_SIZE, Image.SCALE_SMOOTH), 0, 0, this);
        }
    }
}
