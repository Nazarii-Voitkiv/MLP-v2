import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class ResizeScriptTrainTest {

    private static final int TARGET_WIDTH = 32;
    private static final int TARGET_HEIGHT = 32;

    public static void main(String[] args) {
        // --- 1. Куди зберігаємо ---
        // Папки для збереження зображень train і test
        File outputTrainM = new File("resized_data/train/M");
        File outputTrainN = new File("resized_data/train/N");
        File outputTrainO = new File("resized_data/train/O");

        File outputTestM = new File("resized_data/test/M");
        File outputTestN = new File("resized_data/test/N");
        File outputTestO = new File("resized_data/test/O");

        // Створюємо усі потрібні папки, якщо їх немає
        outputTrainM.mkdirs();
        outputTrainN.mkdirs();
        outputTrainO.mkdirs();
        outputTestM.mkdirs();
        outputTestN.mkdirs();
        outputTestO.mkdirs();

        // --- 2. Звідки зчитуємо ---
        // Припустимо, у вас структура:
        // train/m, train/M_caps, train/n, train/N_caps, train/o, train/O_caps
        // test/m, test/M_caps, test/n, test/N_caps, test/o, test/O_caps

        // a) Для train
        File inputTrainM1 = new File("train/m");
        File inputTrainM2 = new File("train/M_caps");

        File inputTrainN1 = new File("train/n");
        File inputTrainN2 = new File("train/N_caps");

        File inputTrainO1 = new File("train/o");
        File inputTrainO2 = new File("train/O_caps");

        // b) Для test
        File inputTestM1 = new File("test/m");
        File inputTestM2 = new File("test/M_caps");

        File inputTestN1 = new File("test/n");
        File inputTestN2 = new File("test/N_caps");

        File inputTestO1 = new File("test/o");
        File inputTestO2 = new File("test/O_caps");

        try {
            // 3. Для TRAIN
            // M
            resizeAllImages(inputTrainM1, outputTrainM, TARGET_WIDTH, TARGET_HEIGHT);
            resizeAllImages(inputTrainM2, outputTrainM, TARGET_WIDTH, TARGET_HEIGHT);
            // N
            resizeAllImages(inputTrainN1, outputTrainN, TARGET_WIDTH, TARGET_HEIGHT);
            resizeAllImages(inputTrainN2, outputTrainN, TARGET_WIDTH, TARGET_HEIGHT);
            // O
            resizeAllImages(inputTrainO1, outputTrainO, TARGET_WIDTH, TARGET_HEIGHT);
            resizeAllImages(inputTrainO2, outputTrainO, TARGET_WIDTH, TARGET_HEIGHT);

            // 4. Для TEST
            // M
            resizeAllImages(inputTestM1, outputTestM, TARGET_WIDTH, TARGET_HEIGHT);
            resizeAllImages(inputTestM2, outputTestM, TARGET_WIDTH, TARGET_HEIGHT);
            // N
            resizeAllImages(inputTestN1, outputTestN, TARGET_WIDTH, TARGET_HEIGHT);
            resizeAllImages(inputTestN2, outputTestN, TARGET_WIDTH, TARGET_HEIGHT);
            // O
            resizeAllImages(inputTestO1, outputTestO, TARGET_WIDTH, TARGET_HEIGHT);
            resizeAllImages(inputTestO2, outputTestO, TARGET_WIDTH, TARGET_HEIGHT);

            System.out.println("Готово! Зображення з train і test перетворені у 32×32 і збережені у \"resized_data/\".");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Пройшовшись по файлах у папці inputDir,
     * зчитуємо зображення, зменшуємо розмір до (w×h)
     * і зберігаємо у папку outputDir із тим самим ім'ям файлу.
     */
    private static void resizeAllImages(File inputDir, File outputDir, int w, int h) throws IOException {
        if (!inputDir.exists() || !inputDir.isDirectory()) {
            System.err.println("Пропускаємо: " + inputDir + " (не існує або не папка)");
            return;
        }

        File[] files = inputDir.listFiles();
        if (files == null) {
            System.err.println("Папка порожня або недоступна: " + inputDir);
            return;
        }

        for (File f : files) {
            if (!f.isFile()) {
                continue;
            }

            BufferedImage original = ImageIO.read(f);
            if (original == null) {
                System.err.println("Не вдалося прочитати (не зображення?): " + f.getName());
                continue;
            }

            BufferedImage resized = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
            Graphics2D g2 = resized.createGraphics();
            g2.drawImage(original, 0, 0, w, h, null);
            g2.dispose();

            File outFile = new File(outputDir, f.getName());
            ImageIO.write(resized, "png", outFile);
        }
    }
}
