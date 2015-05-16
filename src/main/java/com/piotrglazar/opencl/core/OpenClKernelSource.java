package com.piotrglazar.opencl.core;

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
    private final String name;

    private OpenClKernelSource(String[] kernelSourceCode, String name) {
        this.kernelSourceCode = kernelSourceCode;
        this.name = name;
    }

    public String[] getKernelSourceCode() {
        return kernelSourceCode;
    }

    public static OpenClKernelSource getKernelSourceCode(String root, String fileName) {
        URL resource = getResource(root + "/" + fileName + ".cl");

        List<String> kernelLines = getLines(resource);
        return new OpenClKernelSource(new String[]{ Joiner.on("").join(kernelLines) }, fileName);
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

    public String getName() {
        return name;
    }
}
