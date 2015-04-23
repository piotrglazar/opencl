package com.piotrglazar.opencl;

import com.google.common.base.Charsets;
import com.google.common.base.Joiner;
import com.google.common.base.Throwables;
import com.google.common.io.Files;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.List;

public class OpenClKernelSource {

    private final String[] kernelSourceCode;

    private OpenClKernelSource(String[] kernelSourceCode) {
        this.kernelSourceCode = kernelSourceCode;
    }

    public String[] getKernelSourceCode() {
        return kernelSourceCode;
    }

    public static OpenClKernelSource getKernelSourceCode(String kernelFilePath) {
        URL resource = getResource(kernelFilePath);

        List<String> kernelLines = getLines(resource);
        return new OpenClKernelSource(new String[]{ Joiner.on("").join(kernelLines) });
    }

    private static List<String> getLines(URL resource) {
        try {
            return Files.readLines(new File(resource.toURI()), Charsets.UTF_8);
        } catch (IOException | URISyntaxException e) {
            throw Throwables.propagate(e);
        }
    }

    private static URL getResource(String filePath) {
        return Thread.currentThread().getContextClassLoader().getResource(filePath);
    }
}
