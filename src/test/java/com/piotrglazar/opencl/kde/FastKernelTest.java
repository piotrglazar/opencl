package com.piotrglazar.opencl.kde;

import com.google.common.base.Splitter;
import junitparams.JUnitParamsRunner;
import junitparams.Parameters;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.InOrder;
import org.mockito.Mockito;

import java.util.List;

import static java.util.stream.Collectors.toList;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;

@RunWith(JUnitParamsRunner.class)
public class FastKernelTest {

    private OpenClApi openClApi = mock(OpenClApi.class);

    @Test
    @Parameters({
            "3, 123, 1024, 256 | 768;0;256;512, 256;256;256;256",
            "0, 100, 1024, 256 | 0;256;512;768, 256;256;256;256",
            "0, 100, 1200, 256 | 0;256;512;768;1024, 256;256;256;256;176",
            "2, 100, 1200, 256 | 512;768;1024;0;256, 256;256;176;256;256"
    })
    public void shouldCopyToCache(int gid, int inputAddress, int inputSize, int maxLocalCopy, String rawOffsets,
                                  String rawCopySizes) {
        // given
        List<Integer> offsets = ints(rawOffsets);
        List<Integer> copySizes = ints(rawCopySizes);
        assertThat(offsets.size()).isEqualTo(copySizes.size());

        // when
        mainCalculationLoop(gid, inputAddress, inputSize, maxLocalCopy);

        // then
        InOrder inOrder = Mockito.inOrder(openClApi);
        for (int i = 0; i < offsets.size(); ++i) {
            inOrder.verify(openClApi).asyncWorkGroupCopy(offsets.get(i) + inputAddress, copySizes.get(i));
            inOrder.verify(openClApi, times(copySizes.get(i))).compute();
        }
    }

    public void mainCalculationLoop(int gid, int input, int inputSize, int maxLocalCopy) {
        int offset = gid * maxLocalCopy;

        for (int k = 0; k < (inputSize + maxLocalCopy - 1) / maxLocalCopy; ++k) {
            int copySize = (offset + maxLocalCopy < inputSize) ? maxLocalCopy : inputSize - offset;
            openClApi.asyncWorkGroupCopy(input + offset, copySize);

            for (int i = 0; i < copySize; ++i) {
                openClApi.compute();
            }

            offset = (offset + copySize) % inputSize;
        }
    }

    public List<Integer> ints(String rawOffsets) {
        return Splitter.on(";")
                .splitToList(rawOffsets).stream()
                .map(Integer::parseInt)
                .collect(toList());
    }
}
