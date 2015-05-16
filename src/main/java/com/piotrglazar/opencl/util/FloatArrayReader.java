package com.piotrglazar.opencl.util;

import java.io.InputStream;
import java.util.Scanner;

public class FloatArrayReader {

    public FloatArray read(String filePath) {
        Scanner scanner = getScanner(filePath);
        float[] floats = readRawInput(scanner);
        return new FloatArray(floats);
    }

    private Scanner getScanner(String filePath) {
        InputStream resourceAsStream = getResourceInputStream(filePath);
        return new Scanner(resourceAsStream);
    }

    private InputStream getResourceInputStream(String filePath) {
        return Thread.currentThread().getContextClassLoader().getResourceAsStream(filePath);
    }

    private float[] readRawInput(Scanner sampleScanner) {
        int size = sampleScanner.nextInt();
        float[] result = new float[size];

        for (int i = 0; i < size; ++i) {
            String raw = sampleScanner.next();
            result[i] = Float.parseFloat(raw);
        }

        return result;
    }
}
