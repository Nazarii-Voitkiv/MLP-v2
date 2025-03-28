import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class RecognizerApp extends JFrame {
    private static final int CANVAS_SIZE = 280;
    private static final int PIXEL_SIZE = 28;
    private static final String MODEL_PATH = "model.dat";

    private DrawingPanel drawingPanel;
    private JLabel resultLabel;
    private JButton recognizeButton, clearButton, saveButton, trainButton;
    private NeuralNetwork neuralNetwork;

    public RecognizerApp() {
        setTitle("Розпізнавання літер");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout(10, 10));
        
        // Add padding to the main content pane
        ((JComponent) getContentPane()).setBorder(new EmptyBorder(10, 10, 10, 10));

        // Завантаження моделі при старті (тільки 1 раз)
        loadNeuralNetwork();

        // Створення панелі для малювання
        drawingPanel = new DrawingPanel();
        
        // Wrap the drawing panel in a centered panel
        JPanel centeringPanel = new JPanel(new GridBagLayout());
        centeringPanel.add(drawingPanel);
        centeringPanel.setBackground(Color.DARK_GRAY);

        // Створення верхньої панелі з результатом
        resultLabel = new JLabel("Намалюйте літеру (M, O або N)");
        resultLabel.setHorizontalAlignment(SwingConstants.CENTER);
        resultLabel.setFont(new Font(resultLabel.getFont().getName(), Font.BOLD, 18));
        resultLabel.setBorder(new EmptyBorder(10, 10, 10, 10));

        // Створення нижньої панелі з кнопками (GridLayout 2x2)
        JPanel buttonPanel = new JPanel(new GridLayout(2, 2, 10, 10));
        buttonPanel.setBorder(new EmptyBorder(10, 10, 10, 10));
        
        recognizeButton = new JButton("Розпізнати");
        clearButton = new JButton("Очистити");
        saveButton = new JButton("Зберегти як CSV");
        trainButton = new JButton("Донавчити на цьому прикладі");
        
        // Поки що деякі кнопки неактивні
        saveButton.setEnabled(false);
        trainButton.setEnabled(false);
        
        // Налаштування розміру та вигляду кнопок
        configureButton(recognizeButton);
        configureButton(clearButton);
        configureButton(saveButton);
        configureButton(trainButton);
        
        // Додавання обробників подій
        recognizeButton.addActionListener(e -> recognizeDrawing());
        clearButton.addActionListener(e -> drawingPanel.clear());
        saveButton.addActionListener(e -> saveAsCSV());
        trainButton.addActionListener(e -> trainOnExample());
        
        // Додавання кнопок на панель
        buttonPanel.add(recognizeButton);
        buttonPanel.add(clearButton);
        buttonPanel.add(saveButton);
        buttonPanel.add(trainButton);

        // Додавання компонентів на форму
        add(resultLabel, BorderLayout.NORTH);
        add(centeringPanel, BorderLayout.CENTER); // Use the centering panel instead
        add(buttonPanel, BorderLayout.SOUTH);

        // Налаштування розміру вікна (adjust to be slightly wider)
        setSize(CANVAS_SIZE + 120, CANVAS_SIZE + 200);
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
            System.out.println("Модель успішно завантажено");
        } catch (IOException | ClassNotFoundException e) {
            JOptionPane.showMessageDialog(this, 
                "Помилка завантаження моделі: " + e.getMessage(), 
                "Помилка", JOptionPane.ERROR_MESSAGE);
            System.err.println("Помилка завантаження моделі: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void recognizeDrawing() {
        // Перевірка, чи завантажена модель
        if (neuralNetwork == null) {
            resultLabel.setText("Помилка: модель не завантажена");
            return;
        }

        // Отримання зображення і центрування
        double[] imageData = drawingPanel.getBinarizedImage();
        imageData = ImageProcessor.centerImage(imageData);

        try {
            // Отримуємо "сирі" значення (без softmax)
            double[] rawOutputs = neuralNetwork.predict(imageData);
            
            // Знаходження індексу максимального значення
            int maxIndex = 0;
            double maxValue = rawOutputs[0];
            for (int i = 1; i < rawOutputs.length; i++) {
                if (rawOutputs[i] > maxValue) {
                    maxValue = rawOutputs[i];
                    maxIndex = i;
                }
            }
            
            // Визначення літери
            char recognizedLetter = ' ';
            switch (maxIndex) {
                case 0: recognizedLetter = 'M'; break;
                case 1: recognizedLetter = 'O'; break;
                case 2: recognizedLetter = 'N'; break;
            }
            
            // Форматування результату
            String resultText;
            if (maxValue >= 0.5) {
                resultText = String.format(
                    "Результат: %c (M:%.2f, O:%.2f, N:%.2f)",
                    recognizedLetter,
                    rawOutputs[0],
                    rawOutputs[1],
                    rawOutputs[2]
                );
            } else {
                resultText = String.format(
                    "Не впевнений. Найближче: %c (M:%.2f, O:%.2f, N:%.2f)",
                    recognizedLetter,
                    rawOutputs[0],
                    rawOutputs[1],
                    rawOutputs[2]
                );
            }
            
            resultLabel.setText(resultText);
            
        } catch (Exception e) {
            resultLabel.setText("Помилка розпізнавання: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void saveAsCSV() {
        // Заглушка для майбутньої реалізації
        System.out.println("Функція збереження в CSV поки не реалізована");
    }
    
    private void trainOnExample() {
        // Заглушка для майбутньої реалізації
        System.out.println("Функція донавчання поки не реалізована");
    }

    // Клас для панелі малювання
    private class DrawingPanel extends JPanel {
        private BufferedImage image;
        private Graphics2D g2d;

        public DrawingPanel() {
            setPreferredSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setMinimumSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setBackground(Color.BLACK);
            setBorder(BorderFactory.createLineBorder(Color.GRAY, 2));

            // Створення зображення 28x28 пікселів
            image = new BufferedImage(PIXEL_SIZE, PIXEL_SIZE, BufferedImage.TYPE_BYTE_GRAY);
            g2d = image.createGraphics();
            g2d.setColor(Color.BLACK);
            g2d.fillRect(0, 0, PIXEL_SIZE, PIXEL_SIZE);
            g2d.setColor(Color.WHITE);

            // Додавання обробників подій миші
            MouseAdapter mouseHandler = new MouseAdapter() {
                @Override public void mousePressed(MouseEvent e) { draw(e.getX(), e.getY()); }
                @Override public void mouseDragged(MouseEvent e) { draw(e.getX(), e.getY()); }
            };

            addMouseListener(mouseHandler);
            addMouseMotionListener(mouseHandler);
        }

        private void draw(int x, int y) {
            // Перетворення координат з екрану в координати зображення
            int pixelX = x * PIXEL_SIZE / CANVAS_SIZE;
            int pixelY = y * PIXEL_SIZE / CANVAS_SIZE;
            
            // Малювання 3x3 блоку для кращої видимості
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
            resultLabel.setText("Панель очищено. Намалюйте нову літеру.");
        }

        public double[] getBinarizedImage() {
            double[] data = new double[PIXEL_SIZE * PIXEL_SIZE];

            // Конвертація зображення у бінарний масив
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

            // Малювання збільшеного зображення
            g.drawImage(image.getScaledInstance(CANVAS_SIZE, CANVAS_SIZE, Image.SCALE_SMOOTH), 0, 0, this);

            // Малювання сітки (світло-сіра)
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
            // Встановлення системного Look and Feel для кращого вигляду
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        SwingUtilities.invokeLater(RecognizerApp::new);
    }
}
