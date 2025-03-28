public class ImageProcessor {

    public static double[] centerImage(double[] flatInput) {
        int size = 28;
        double[][] img = new double[size][size];

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                img[y][x] = flatInput[y * size + x];
            }
        }

        int minX = size, minY = size, maxX = 0, maxY = 0;
        boolean foundPixels = false;
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                if (img[y][x] > 0.1) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    foundPixels = true;
                }
            }
        }

        if (!foundPixels) return flatInput;
        
        int boxWidth = maxX - minX + 1;
        int boxHeight = maxY - minY + 1;

        if (boxWidth <= 0 || boxHeight <= 0) return flatInput;

        double[][] centered = new double[size][size];

        double scale = 1.0;
        int padding = 2;
        int maxDim = Math.max(boxWidth, boxHeight);
        if (maxDim > size - 2 * padding) {
            scale = (double)(size - 2 * padding) / maxDim;
        }
        
        int newWidth = (int)(boxWidth * scale);
        int newHeight = (int)(boxHeight * scale);
        
        int offsetX = (size - newWidth) / 2;
        int offsetY = (size - newHeight) / 2;

        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                int srcX = minX + (int)(x / scale);
                int srcY = minY + (int)(y / scale);
                if (srcX >= minX && srcX <= maxX && srcY >= minY && srcY <= maxY) {
                    centered[offsetY + y][offsetX + x] = img[srcY][srcX];
                }
            }
        }

        double[] result = new double[size * size];
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                result[y * size + x] = centered[y][x];
            }
        }

        return result;
    }
    
    public static double[] binarize(double[] input, double threshold) {
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = input[i] > threshold ? 1.0 : 0.0;
        }
        return result;
    }
    
    public static double[] processImageData(int[] rawPixels) {
        double[] normalized = new double[rawPixels.length];
        
        for (int i = 0; i < rawPixels.length; i++) {
            normalized[i] = rawPixels[i] / 255.0;
        }
        
        double[] binarized = binarize(normalized, 0.5);
        
        return centerImage(binarized);
    }
}
