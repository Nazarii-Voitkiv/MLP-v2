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

        // Створення нижньої панелі з кнопками (2 рядки)
        JPanel buttonPanel = new JPanel(new BorderLayout(10, 10));
        buttonPanel.setBorder(new EmptyBorder(10, 10, 10, 10));
        
        // Панель для верхніх кнопок (Розпізнати та Очистити)
        JPanel topButtonPanel = new JPanel(new GridLayout(1, 2, 10, 0));
        
        recognizeButton = new JButton("Розпізнати");
        clearButton = new JButton("Очистити");
        wrongLetterButton = new JButton("❌ Це не та літера");
        
        // Налаштування розміру та вигляду кнопок
        configureButton(recognizeButton);
        configureButton(clearButton);
        configureButton(wrongLetterButton);
        
        // Додавання обробників подій
        recognizeButton.addActionListener(e -> recognizeDrawing());
        clearButton.addActionListener(e -> drawingPanel.clear());
        wrongLetterButton.addActionListener(e -> handleWrongLetter());
        
        // Додавання верхніх кнопок на панель
        topButtonPanel.add(recognizeButton);
        topButtonPanel.add(clearButton);
        
        // Додавання кнопки "Це не та літера" в окрему панель для центрування
        JPanel bottomButtonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        bottomButtonPanel.add(wrongLetterButton);
        
        // Додавання панелей з кнопками на головну панель кнопок
        buttonPanel.add(topButtonPanel, BorderLayout.NORTH);
        buttonPanel.add(bottomButtonPanel, BorderLayout.SOUTH);

        // Додавання компонентів на форму
        add(resultLabel, BorderLayout.NORTH);
        add(centeringPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);

        // Налаштування розміру вікна
        setSize(CANVAS_SIZE + 120, CANVAS_SIZE + 220); // Increased height from +150 to +220
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

    /**
     * Обробник для кнопки "Це не та літера"
     */
    private void handleWrongLetter() {
        // Перевірка, чи є що зберігати
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
                "Немає що зберігати. Спочатку намалюйте літеру.", 
                "Помилка", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        // Масив варіантів для вибору
        Object[] options = { "M", "O", "N" };
        
        // Показуємо діалог з вибором правильної літери
        int choice = JOptionPane.showOptionDialog(this,
            "Яка це була літера?",
            "Виберіть правильний варіант",
            JOptionPane.DEFAULT_OPTION,
            JOptionPane.QUESTION_MESSAGE,
            null,
            options,
            options[0]);
            
        if (choice == JOptionPane.CLOSED_OPTION) {
            // Користувач закрив діалог
            return;
        }
        
        // Визначаємо обрану літеру
        char letter = ' ';
        switch (choice) {
            case 0: letter = 'M'; break;
            case 1: letter = 'O'; break;
            case 2: letter = 'N'; break;
        }
        
        try {
            // Центруємо зображення перед збереженням
            imageData = ImageProcessor.centerImage(imageData);
            
            // Генеруємо унікальне ім'я файлу і зберігаємо
            String fileName = generateUniqueFileName(letter);
            saveToCSV(imageData, fileName);
            
            // Показуємо повідомлення про успіх
            JOptionPane.showMessageDialog(this,
                "✅ Збережено приклад як " + fileName,
                "Успіх",
                JOptionPane.INFORMATION_MESSAGE);
                
            resultLabel.setText("Збережено як " + fileName);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this,
                "Помилка при збереженні: " + e.getMessage(),
                "Помилка",
                JOptionPane.ERROR_MESSAGE);
            e.printStackTrace();
        }
    }
    
    /**
     * Генерує унікальне ім'я файлу для зазначеної літери
     */
    private String generateUniqueFileName(char letter) {
        int maxNumber = 0;
        File dataDir = new File(DATA_DIR);
        
        // Створюємо директорію, якщо вона не існує
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
    
    /**
     * Зберігає масив даних у CSV-файл
     */
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

    // Клас для панелі малювання
    private class DrawingPanel extends JPanel {
        private BufferedImage image;
        private Graphics2D g2d;

        public DrawingPanel() {
            setPreferredSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setMinimumSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            // Add maximum size to ensure the panel doesn't shrink
            setMaximumSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
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
