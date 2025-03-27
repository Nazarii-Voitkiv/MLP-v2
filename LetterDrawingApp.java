import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;

public class LetterDrawingApp extends JFrame {
    
    private Siec neuralNetwork;
    private JPanel resultPanel;
    private DrawingPanel drawingPanel;
    private JLabel resultLabel;
    private JProgressBar[] confidenceBars;
    private JLabel[] confidenceLabels;
    private final String[] classLabels = {"M", "O", "N"};
    private final Color[] classColors = {Color.RED, Color.GREEN, Color.BLUE};
    
    public LetterDrawingApp(String title) {
        super(title);
        
        // Загрузка або створення нейронної мережі
        try {
            File modelFile = new File("letter_classification_model.model");
            if (modelFile.exists()) {
                neuralNetwork = Siec.loadModel("letter_classification_model.model");
                System.out.println("Модель завантажена успішно");
            } else {
                System.out.println("Файл моделі не знайдено. Будь ласка, спочатку навчіть модель.");
                System.exit(1);
            }
        } catch (Exception e) {
            System.err.println("Помилка при завантаженні моделі: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
        
        // Налаштування вікна
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        setSize(800, 600);
        
        // Створюємо панель для малювання
        drawingPanel = new DrawingPanel();
        add(drawingPanel, BorderLayout.CENTER);
        
        // Створюємо панель для відображення результатів
        createResultPanel();
        add(resultPanel, BorderLayout.EAST);
        
        // Створюємо панель інструментів
        createToolBar();
        
        setLocationRelativeTo(null);
        setVisible(true);
    }
    
    private void createResultPanel() {
        resultPanel = new JPanel();
        resultPanel.setPreferredSize(new Dimension(250, getHeight()));
        resultPanel.setLayout(new BoxLayout(resultPanel, BoxLayout.Y_AXIS));
        resultPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        
        // Заголовок результатів
        JLabel headerLabel = new JLabel("Намалюйте літеру (M, O або N)");
        headerLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
        resultPanel.add(headerLabel);
        resultPanel.add(Box.createVerticalStrut(20));
        
        // Індикатори та підписи для впевненості в кожній літері
        confidenceBars = new JProgressBar[classLabels.length];
        confidenceLabels = new JLabel[classLabels.length];
        
        for (int i = 0; i < classLabels.length; i++) {
            JLabel classLabel = new JLabel("Літера " + classLabels[i] + ":");
            classLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
            resultPanel.add(classLabel);
            
            confidenceBars[i] = new JProgressBar(0, 100);
            confidenceBars[i].setValue(0);
            confidenceBars[i].setStringPainted(true);
            confidenceBars[i].setForeground(classColors[i]);
            resultPanel.add(confidenceBars[i]);
            
            confidenceLabels[i] = new JLabel("0.00%");
            confidenceLabels[i].setAlignmentX(Component.CENTER_ALIGNMENT);
            resultPanel.add(confidenceLabels[i]);
            
            resultPanel.add(Box.createVerticalStrut(15));
        }
        
        // Результат розпізнавання
        JLabel predictionLabel = new JLabel("Розпізнано як:");
        predictionLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
        resultPanel.add(predictionLabel);
        resultPanel.add(Box.createVerticalStrut(10));
        
        JLabel prediction = new JLabel("?");
        prediction.setFont(new Font("Arial", Font.BOLD, 72));
        prediction.setAlignmentX(Component.CENTER_ALIGNMENT);
        resultPanel.add(prediction);
        
        // Встановлюємо поле resultLabel коректно
        resultLabel = prediction;
    }
    
    private void createToolBar() {
        JToolBar toolbar = new JToolBar();
        toolbar.setFloatable(false);
        
        JButton clearButton = new JButton("Очистити");
        clearButton.addActionListener(e -> {
            drawingPanel.clear();
            resetResults();
        });
        toolbar.add(clearButton);
        
        JButton recognizeButton = new JButton("Розпізнати");
        recognizeButton.addActionListener(e -> recognizeLetter());
        toolbar.add(recognizeButton);
        
        add(toolbar, BorderLayout.NORTH);
    }
    
    private void resetResults() {
        for (int i = 0; i < classLabels.length; i++) {
            confidenceBars[i].setValue(0);
            confidenceLabels[i].setText("0.00%");
        }
        resultLabel.setText("?");
    }
    
    private void recognizeLetter() {
        try {
            // Отримуємо зображення з панелі малювання і конвертуємо його для нейронної мережі
            BufferedImage image = drawingPanel.getImage();
            double[] pixelData = preprocessImage(image);
            
            if (pixelData == null) {
                JOptionPane.showMessageDialog(this, 
                    "Нічого не намальовано. Будь ласка, намалюйте літеру.", 
                    "Зображення порожнє", 
                    JOptionPane.WARNING_MESSAGE);
                return;
            }
            
            // Отримуємо прогноз від нейронної мережі
            double[] prediction = neuralNetwork.oblicz_wyjscie(pixelData);
            
            // Знаходимо найбільш вірогідний клас
            int maxIndex = 0;
            for (int i = 1; i < prediction.length; i++) {
                if (prediction[i] > prediction[maxIndex]) {
                    maxIndex = i;
                }
            }
            
            // Оновлюємо відображення результатів
            DecimalFormat df = new DecimalFormat("0.00");
            for (int i = 0; i < prediction.length; i++) {
                int percentage = (int)(prediction[i] * 100);
                confidenceBars[i].setValue(percentage);
                confidenceLabels[i].setText(df.format(prediction[i] * 100) + "%");
            }
            
            // Оновлюємо загальний результат
            resultLabel.setText(classLabels[maxIndex]);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, 
                "Помилка розпізнавання: " + e.getMessage(), 
                "Помилка", 
                JOptionPane.ERROR_MESSAGE);
            e.printStackTrace();
        }
    }
    
    private double[] preprocessImage(BufferedImage image) {
        // Перевіряємо, чи не порожнє зображення
        if (isImageEmpty(image)) {
            return null;
        }
        
        // Масштабування зображення до 128x128
        BufferedImage scaledImage = new BufferedImage(128, 128, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = scaledImage.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.drawImage(image, 0, 0, 128, 128, null);
        g2d.dispose();
        
        // Конвертація зображення в одновимірний масив даних і нормалізація
        double[] pixelData = new double[128 * 128];
        for (int y = 0; y < 128; y++) {
            for (int x = 0; x < 128; x++) {
                int rgb = scaledImage.getRGB(x, y);
                // Вилучення яскравості (grayscale)
                int gray = (rgb >> 16) & 0xFF;  // Беремо червоний канал як сірий
                // Інвертуємо кольори: чорний фон (0) -> білий текст (255)
                gray = 255 - gray;
                // Нормалізація значень до діапазону 0-1
                pixelData[y * 128 + x] = gray / 255.0;
            }
        }
        
        return pixelData;
    }
    
    private boolean isImageEmpty(BufferedImage image) {
        // Перевіряємо, чи є хоч один не-білий піксель
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;
                // Якщо піксель не білий, зображення не порожнє
                if (red < 250 || green < 250 || blue < 250) {
                    return false;
                }
            }
        }
        return true;
    }
    
    // Клас для панелі малювання
    class DrawingPanel extends JPanel {
        private BufferedImage drawingImage;
        private Graphics2D g2d;
        private Point lastPoint;
        
        public DrawingPanel() {
            setPreferredSize(new Dimension(500, 500));
            setBackground(Color.WHITE);
            
            drawingImage = new BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB);
            g2d = drawingImage.createGraphics();
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, 500, 500);
            g2d.setColor(Color.BLACK);
            g2d.setStroke(new BasicStroke(20, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
            
            // Додаємо обробники подій миші для малювання
            addMouseListener(new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent e) {
                    lastPoint = e.getPoint();
                }
                
                @Override
                public void mouseReleased(MouseEvent e) {
                    lastPoint = null;
                }
            });
            
            addMouseMotionListener(new MouseMotionAdapter() {
                @Override
                public void mouseDragged(MouseEvent e) {
                    if (lastPoint != null) {
                        g2d.drawLine(lastPoint.x, lastPoint.y, e.getX(), e.getY());
                        lastPoint = e.getPoint();
                        repaint();
                    }
                }
            });
        }
        
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            g.drawImage(drawingImage, 0, 0, this);
        }
        
        public void clear() {
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, getWidth(), getHeight());
            g2d.setColor(Color.BLACK);
            repaint();
        }
        
        public BufferedImage getImage() {
            return drawingImage;
        }
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new LetterDrawingApp("Розпізнавання літер"));
    }
}
