import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.List;
import java.util.regex.*;
import java.util.stream.Collectors;

public class SampleViewer extends JFrame {
    private static final int CANVAS_SIZE = 400;
    private static final int PIXEL_SIZE = 28;
    private static final String DATA_DIR = "data";
    private static final String TEST_DATA_DIR = "test_data";
    private static final Pattern LETTER_PATTERN = Pattern.compile("([MON])_(\\d+)\\.csv");
    
    private JPanel cardPanel;
    private CardLayout cardLayout;
    private JList<String> fileList;
    private JRadioButton dataRadio, testDataRadio;
    private JPanel imagePanel;
    private JLabel infoLabel;
    private JButton deleteButton;
    private JButton prevButton, nextButton;
    
    private Map<String, double[]> cachedSamples = new HashMap<>();
    private String currentDirectory = DATA_DIR;
    
    public SampleViewer() {
        setTitle("Przeglądarka próbek");
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setSize(650, 545);
        setLocationRelativeTo(null);
        
        createUI();
        loadFiles();
        
        setVisible(true);
    }
    
    private void createUI() {
        setLayout(new BorderLayout(10, 10));

        JPanel dirPanel = new JPanel(new GridLayout(2, 1, 5, 5));
        dirPanel.setBorder(BorderFactory.createTitledBorder("Wybierz katalog"));
        dataRadio = new JRadioButton("Dane treningowe (data)");
        testDataRadio = new JRadioButton("Dane testowe (test_data)");

        dataRadio.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        testDataRadio.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        dataRadio.setSelected(true);
        
        ButtonGroup dirGroup = new ButtonGroup();
        dirGroup.add(dataRadio);
        dirGroup.add(testDataRadio);
        
        dataRadio.addActionListener(e -> changeDirectory(DATA_DIR));
        testDataRadio.addActionListener(e -> changeDirectory(TEST_DATA_DIR));
        
        dirPanel.add(dataRadio);
        dirPanel.add(testDataRadio);

        JPanel listPanel = new JPanel(new BorderLayout());
        listPanel.setBorder(BorderFactory.createTitledBorder("Pliki"));
        fileList = new JList<>();
        fileList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        fileList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                displaySelectedFile();
            }
        });

        fileList.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 14));
        
        JScrollPane scrollPane = new JScrollPane(fileList);
        listPanel.add(scrollPane, BorderLayout.CENTER);
        
        JPanel leftPanel = new JPanel(new BorderLayout());
        leftPanel.add(dirPanel, BorderLayout.NORTH);
        leftPanel.add(listPanel, BorderLayout.CENTER);
        leftPanel.setPreferredSize(new Dimension(220, getHeight()));

        cardPanel = new JPanel();
        cardLayout = new CardLayout();
        cardPanel.setLayout(cardLayout);
        
        imagePanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                drawSelectedSample(g);
            }
        };
        imagePanel.setBackground(Color.WHITE);
        imagePanel.setPreferredSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));

        JPanel imagePanelWrapper = new JPanel(new BorderLayout());
        imagePanelWrapper.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        imagePanelWrapper.add(imagePanel, BorderLayout.CENTER);

        JPanel emptyPanel = new JPanel();
        emptyPanel.setBackground(Color.WHITE);
        JLabel emptyLabel = new JLabel("Brak wybranego pliku");
        emptyLabel.setHorizontalAlignment(SwingConstants.CENTER);
        emptyLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
        emptyPanel.add(emptyLabel);
        
        cardPanel.add(emptyPanel, "empty");
        cardPanel.add(imagePanelWrapper, "image");
        cardLayout.show(cardPanel, "empty");

        JPanel infoPanel = new JPanel(new BorderLayout(5, 5));
        infoPanel.setBorder(BorderFactory.createTitledBorder("Informacja"));
        
        infoLabel = new JLabel("Wybierz plik, aby wyświetlić szczegóły");
        infoLabel.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 14));
        infoPanel.add(infoLabel, BorderLayout.CENTER);

        JPanel navPanel = new JPanel();
        prevButton = new JButton("< Poprzedni");
        nextButton = new JButton("Następny >");
        deleteButton = new JButton("Usuń");

        prevButton.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        nextButton.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        deleteButton.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        
        prevButton.addActionListener(e -> navigateList(-1));
        nextButton.addActionListener(e -> navigateList(1));
        deleteButton.addActionListener(e -> deleteSelectedFile());
        
        navPanel.add(prevButton);
        navPanel.add(deleteButton);
        navPanel.add(nextButton);
        
        JPanel bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.add(infoPanel, BorderLayout.CENTER);
        bottomPanel.add(navPanel, BorderLayout.SOUTH);

        JPanel rightPanel = new JPanel(new BorderLayout());
        rightPanel.add(cardPanel, BorderLayout.CENTER);
        rightPanel.add(bottomPanel, BorderLayout.SOUTH);
        rightPanel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        
        add(leftPanel, BorderLayout.WEST);
        add(rightPanel, BorderLayout.CENTER);
    }
    
    private void changeDirectory(String dir) {
        currentDirectory = dir;
        cachedSamples.clear();
        loadFiles();
    }
    
    private void loadFiles() {
        File dir = new File(currentDirectory);
        if (!dir.exists() || !dir.isDirectory()) {
            fileList.setListData(new String[]{"Brak plików"});
            return;
        }
        
        File[] files = dir.listFiles(file -> 
            file.isFile() && file.getName().toLowerCase().endsWith(".csv") &&
            LETTER_PATTERN.matcher(file.getName()).matches()
        );
        
        if (files == null || files.length == 0) {
            fileList.setListData(new String[]{"Brak plików"});
            return;
        }
        
        Arrays.sort(files, Comparator.comparing(File::getName));
        
        DefaultListModel<String> model = new DefaultListModel<>();
        for (File file : files) {
            model.addElement(file.getName());
        }
        
        fileList.setModel(model);
        
        if (model.getSize() > 0) {
            fileList.setSelectedIndex(0);
        }
        
        updateNavButtons();
    }
    
    private void displaySelectedFile() {
        String selectedFile = fileList.getSelectedValue();
        if (selectedFile == null || selectedFile.equals("Brak plików")) {
            cardLayout.show(cardPanel, "empty");
            infoLabel.setText("Brak wybranego pliku");
            return;
        }
        
        try {
            Matcher matcher = LETTER_PATTERN.matcher(selectedFile);
            if (matcher.matches()) {
                char letter = matcher.group(1).charAt(0);
                String number = matcher.group(2);
                
                double[] imageData = getImageData(selectedFile);
                
                cardLayout.show(cardPanel, "image");
                infoLabel.setText(String.format("Litera: %c, Numer próbki: %s", letter, number));
                
                imagePanel.repaint();
                updateNavButtons();
            } else {
                cardLayout.show(cardPanel, "empty");
                infoLabel.setText("Nieprawidłowy format pliku");
            }
        } catch (Exception e) {
            cardLayout.show(cardPanel, "empty");
            infoLabel.setText("Błąd odczytu: " + e.getMessage());
        }
    }
    
    private double[] getImageData(String fileName) throws IOException {
        if (cachedSamples.containsKey(fileName)) {
            return cachedSamples.get(fileName);
        }
        
        File file = new File(currentDirectory, fileName);
        String content = new String(Files.readAllBytes(file.toPath()));
        String[] values = content.trim().split(",");
        
        double[] data = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            data[i] = Double.parseDouble(values[i]);
        }
        
        cachedSamples.put(fileName, data);
        return data;
    }
    
    private void drawSelectedSample(Graphics g) {
        String selectedFile = fileList.getSelectedValue();
        if (selectedFile == null || selectedFile.equals("Brak plików")) {
            return;
        }
        
        try {
            double[] imageData = getImageData(selectedFile);
            
            int cellSize = CANVAS_SIZE / PIXEL_SIZE;
            for (int y = 0; y < PIXEL_SIZE; y++) {
                for (int x = 0; x < PIXEL_SIZE; x++) {
                    int index = y * PIXEL_SIZE + x;
                    if (index < imageData.length) {
                        int grayValue = (int)(255 * (1 - imageData[index]));
                        g.setColor(new Color(grayValue, grayValue, grayValue));
                        g.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                    }
                }
            }

            g.setColor(new Color(220, 220, 220));
            for (int i = 0; i <= PIXEL_SIZE; i++) {
                int pos = i * cellSize;
                g.drawLine(0, pos, CANVAS_SIZE, pos);
                g.drawLine(pos, 0, pos, CANVAS_SIZE);
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    private void navigateList(int direction) {
        int selectedIndex = fileList.getSelectedIndex();
        int listSize = fileList.getModel().getSize();
        
        if (listSize <= 1) return;
        
        int newIndex = selectedIndex + direction;
        if (newIndex < 0) newIndex = listSize - 1;
        if (newIndex >= listSize) newIndex = 0;
        
        fileList.setSelectedIndex(newIndex);
        fileList.ensureIndexIsVisible(newIndex);
    }
    
    private void deleteSelectedFile() {
        String selectedFile = fileList.getSelectedValue();
        if (selectedFile == null || selectedFile.equals("Brak plików")) {
            return;
        }
        
        int confirm = JOptionPane.showConfirmDialog(this,
            "Czy na pewno chcesz usunąć plik " + selectedFile + "?",
            "Potwierdzenie usunięcia", JOptionPane.YES_NO_OPTION);
        
        if (confirm == JOptionPane.YES_OPTION) {
            try {
                Files.delete(new File(currentDirectory, selectedFile).toPath());
                cachedSamples.remove(selectedFile);
                loadFiles();
            } catch (IOException e) {
                JOptionPane.showMessageDialog(this, 
                    "Błąd podczas usuwania pliku: " + e.getMessage(),
                    "Błąd", JOptionPane.ERROR_MESSAGE);
            }
        }
    }
    
    private void updateNavButtons() {
        int selectedIndex = fileList.getSelectedIndex();
        int listSize = fileList.getModel().getSize();
        
        prevButton.setEnabled(listSize > 1);
        nextButton.setEnabled(listSize > 1);
        deleteButton.setEnabled(selectedIndex >= 0 && !fileList.getSelectedValue().equals("Brak plików"));
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(SampleViewer::new);
    }
}
