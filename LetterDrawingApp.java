import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Aplikacja do tworzenia próbek liter dla treningu sieci neuronowej
 */
public class LetterDrawingApp extends JFrame {
    private static final int CANVAS_SIZE = 280;
    private static final int PIXEL_SIZE = 28;
    private static final String DATA_DIR = "data";

    private DrawingPanel drawingPanel;
    private JButton saveAsM, saveAsO, saveAsN, clearButton;
    private JLabel statusLabel;

    public LetterDrawingApp() {
        setTitle("Tworzenie próbek liter");  // Translated from Ukrainian
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout(10, 10));

        drawingPanel = new DrawingPanel();

        saveAsM = new JButton("Zapisz jako M");  // Translated
        saveAsO = new JButton("Zapisz jako O");  // Translated
        saveAsN = new JButton("Zapisz jako N");  // Translated
        clearButton = new JButton("Wyczyść");    // Translated

        statusLabel = new JLabel("Gotowy do rysowania"); // Translated
        statusLabel.setHorizontalAlignment(SwingConstants.CENTER);

        saveAsM.addActionListener(e -> saveLetterImage('M'));
        saveAsO.addActionListener(e -> saveLetterImage('O'));
        saveAsN.addActionListener(e -> saveLetterImage('N'));
        clearButton.addActionListener(e -> drawingPanel.clear());

        // Używamy GridLayout dla lepszego rozmieszczenia przycisków
        JPanel buttonPanel = new JPanel(new GridLayout(2, 2, 5, 5));
        buttonPanel.setBorder(BorderFactory.createEmptyBorder(5, 10, 5, 10));
        buttonPanel.add(saveAsM);
        buttonPanel.add(saveAsO);
        buttonPanel.add(saveAsN);
        buttonPanel.add(clearButton);

        add(drawingPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);
        add(statusLabel, BorderLayout.NORTH);

        // Zwiększamy rozmiar okna dla lepszego wyświetlania przycisków
        setSize(CANVAS_SIZE + 60, CANVAS_SIZE + 130);
        setLocationRelativeTo(null);
        setVisible(true);

        createDataDirectory();
    }

    private void createDataDirectory() {
        Path dataPath = Paths.get(DATA_DIR);
        if (!Files.exists(dataPath)) {
            try {
                Files.createDirectory(dataPath);
                statusLabel.setText("Utworzono katalog: " + DATA_DIR); // Translated
            } catch (IOException e) {
                statusLabel.setText("Błąd tworzenia katalogu: " + e.getMessage()); // Translated
                e.printStackTrace();
            }
        }
    }

    private void saveLetterImage(char letter) {
        double[] imageData = drawingPanel.getBinarizedImage();
        String fileName = generateUniqueFileName(letter);

        try {
            saveToCSV(imageData, fileName);
            statusLabel.setText("Zapisano jako " + fileName); // Translated
        } catch (IOException e) {
            statusLabel.setText("Błąd zapisywania: " + e.getMessage()); // Translated
            e.printStackTrace();
        }
    }

    private String generateUniqueFileName(char letter) {
        int maxNumber = 0;
        File dataDir = new File(DATA_DIR);
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

        return String.format("%s_%02d.csv", letter, maxNumber + 1);
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
            setBackground(Color.BLACK);
            setBorder(BorderFactory.createLineBorder(Color.GRAY, 1));

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
            statusLabel.setText("Panel wyczyszczony"); // Translated
        }

        public double[] getBinarizedImage() {
            double[] data = new double[PIXEL_SIZE * PIXEL_SIZE];

            for (int y = 0; y < PIXEL_SIZE; y++) {
                for (int x = 0; x < PIXEL_SIZE; x++) {
                    int rgb = image.getRGB(x, y) & 0xFF;
                    data[y * PIXEL_SIZE + x] = (rgb > 128) ? 1.0 : 0.0;
                }
            }

            return ImageProcessor.centerImage(data);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            g.drawImage(image.getScaledInstance(CANVAS_SIZE, CANVAS_SIZE, Image.SCALE_SMOOTH), 0, 0, this);

            g.setColor(new Color(50, 50, 50));
            for (int i = 0; i <= PIXEL_SIZE; i++) {
                int pos = i * CANVAS_SIZE / PIXEL_SIZE;
                g.drawLine(pos, 0, pos, CANVAS_SIZE);
                g.drawLine(0, pos, CANVAS_SIZE, pos);
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(LetterDrawingApp::new);
    }
}
