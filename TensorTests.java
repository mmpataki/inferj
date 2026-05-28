import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class TensorTests {

    private static final Random RNG = new Random(1337);
    private static int testsRun = 0;
    private static int testsFailed = 0;

    public static void main(String[] args) {
        runAll();
    }

    public static void runAll() {
        List<String> failures = new ArrayList<>();

        run("fuzz slice", TensorTests::fuzzSlice, failures);
        run("fuzz split", TensorTests::fuzzSplit, failures);
        run("fuzz select", TensorTests::fuzzSelect, failures);
        run("fuzz view", TensorTests::fuzzView, failures);
        run("fuzz transpose", TensorTests::fuzzTranspose, failures);
        run("fuzz add broadcast", TensorTests::fuzzAddBroadcast, failures);
        run("fuzz matmul", TensorTests::fuzzMatmul, failures);
        run("fuzz scalarMul", TensorTests::fuzzScalarMul, failures);
        run("fuzz softMax", TensorTests::fuzzSoftmax, failures);
        run("fuzz maskedFill", TensorTests::fuzzMaskedFill, failures);
        run("fuzz layerNorm", TensorTests::fuzzLayerNorm, failures);
        run("fuzz gelu", TensorTests::fuzzGelu, failures);
        run("fuzz argMax", TensorTests::fuzzArgMax, failures);
        run("sort contract", TensorTests::testSortContract, failures);
        run("topK contract", TensorTests::testTopKContract, failures);
        run("sub contract", TensorTests::testSubContract, failures);
        run("multinomial contract", TensorTests::testMultinomialContract, failures);
        run("materialize ops", TensorTests::testMaterializeOps, failures);
        run("fuzz cross-op pipeline", TensorTests::fuzzCrossOpPipeline, failures);
        run("fuzz transformed add+softmax", TensorTests::fuzzTransformedAddSoftmax, failures);
        run("fuzz transformed matmul chain", TensorTests::fuzzTransformedMatmulChain, failures);
        run("fuzz transformed split/select/maskedFill", TensorTests::fuzzTransformedSplitSelectMaskedFill, failures);
        run("edge case ops", TensorTests::testEdgeCases, failures);
        run("attention 4d transpose layout", TensorTests::testAttention4dTransposeLayout, failures);
        run("attention contiguous flatten layout", TensorTests::testAttentionContiguousFlattenLayout, failures);
        run("split non-last dimension layout", TensorTests::testSplitNonLastDimensionLayout, failures);

        System.out.println("TensorTests: run=" + testsRun + ", failed=" + testsFailed);
        if (!failures.isEmpty()) {
            System.out.println("Failures:");
            for (String f : failures) {
                System.out.println("- " + f);
            }
            throw new AssertionError("Tensor tests failed");
        }
    }

    private static void run(String name, Runnable test, List<String> failures) {
        testsRun++;
        try {
            test.run();
            System.out.println("[PASS] " + name);
        } catch (Throwable t) {
            testsFailed++;
            System.out.println("[FAIL] " + name + " -> " + t.getMessage());
            failures.add(name + " -> " + t.getMessage());
        }
    }

    private static void fuzzSlice() {
        for (int it = 0; it < 250; it++) {
            int rows = rnd(2, 7), cols = rnd(2, 7);
            float[] src = randArr(rows * cols);
            InferJ.Tensor t = InferJ.Tensor.of(Arrays.copyOf(src, src.length), rows, cols);

            int r0 = rnd(0, rows - 1), r1 = rnd(r0 + 1, rows);
            int c0 = rnd(0, cols - 1), c1 = rnd(c0 + 1, cols);

            InferJ.Tensor s = t.slice(InferJ.$(r0, r1), InferJ.$(c0, c1));
            assertArrayEquals(new int[]{r1 - r0, c1 - c0}, s.dims, "slice dims");

            for (int r = 0; r < s.dims[0]; r++) {
                for (int c = 0; c < s.dims[1]; c++) {
                    float exp = src[(r0 + r) * cols + (c0 + c)];
                    assertFloatEquals(exp, s.get(r, c), 1e-4f, "slice value");
                }
            }
        }
    }

    private static void fuzzSplit() {
        for (int it = 0; it < 250; it++) {
            int rows = rnd(1, 5);
            int parts = rnd(2, 5);
            int chunk = rnd(1, 5);
            int cols = parts * chunk;
            float[] src = randArr(rows * cols);
            InferJ.Tensor t = InferJ.Tensor.of(Arrays.copyOf(src, src.length), rows, cols);

            InferJ.Tensor[] out = t.split(parts, 1);
            assertIntEquals(parts, out.length, "split part count");

            for (int p = 0; p < parts; p++) {
                InferJ.Tensor tp = out[p];
                assertArrayEquals(new int[]{rows, chunk}, tp.dims, "split dims");
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < chunk; c++) {
                        float exp = src[r * cols + (p * chunk + c)];
                        assertFloatEquals(exp, tp.get(r, c), 1e-4f, "split value");
                    }
                }
            }
        }
    }

    private static void fuzzSelect() {
        for (int it = 0; it < 250; it++) {
            int rows = rnd(2, 8), cols = rnd(1, 8), k = rnd(1, rows);
            float[] src = randArr(rows * cols);
            InferJ.Tensor x = InferJ.Tensor.of(Arrays.copyOf(src, src.length), rows, cols);

            float[] idxf = new float[k];
            int[] idxi = new int[k];
            for (int i = 0; i < k; i++) {
                idxi[i] = rnd(0, rows - 1);
                idxf[i] = idxi[i];
            }
            InferJ.Tensor idx = InferJ.Tensor.of(idxf, k);

            InferJ.Tensor out = x.select(idx);
            out.eval();

            assertArrayEquals(new int[]{k, cols}, out.dims, "select dims");
            for (int r = 0; r < k; r++) {
                for (int c = 0; c < cols; c++) {
                    float exp = src[idxi[r] * cols + c];
                    assertFloatEquals(exp, out.get(r, c), 1e-4f, "select value");
                }
            }
        }
    }

    private static void fuzzView() {
        for (int it = 0; it < 200; it++) {
            int a = rnd(1, 6), b = rnd(1, 6), c = rnd(1, 6);
            int sz = a * b * c;
            float[] src = randArr(sz);
            InferJ.Tensor t = InferJ.Tensor.of(Arrays.copyOf(src, src.length), a, b, c);

            InferJ.Tensor v = t.view(a * b, c);
            assertArrayEquals(new int[]{a * b, c}, v.dims, "view dims");

            int ri = rnd(0, a * b - 1), ci = rnd(0, c - 1);
            int flat = ri * c + ci;
            assertFloatEquals(src[flat], v.get(ri, ci), 1e-4f, "view mapping");

            float marker = 1000f + it;
            v.set(marker, ri, ci);
            assertFloatEquals(marker, t.arr[flat], 1e-4f, "view shared storage");
        }
    }

    private static void fuzzTranspose() {
        for (int it = 0; it < 250; it++) {
            int rows = rnd(1, 7), cols = rnd(1, 7);
            float[] src = randArr(rows * cols);
            InferJ.Tensor t = InferJ.Tensor.of(Arrays.copyOf(src, src.length), rows, cols);
            InferJ.Tensor tr = t.transpose(0, 1);

            assertArrayEquals(new int[]{cols, rows}, tr.dims, "transpose dims");
            for (int r = 0; r < cols; r++) {
                for (int c = 0; c < rows; c++) {
                    float exp = src[c * cols + r];
                    assertFloatEquals(exp, tr.get(r, c), 1e-4f, "transpose value");
                }
            }
        }
    }

    private static void fuzzAddBroadcast() {
        for (int it = 0; it < 300; it++) {
            int m = rnd(1, 8), n = rnd(1, 8);
            float[] a = randArr(m * n);
            float[] b = randArr(n);

            InferJ.Tensor ta = InferJ.Tensor.of(Arrays.copyOf(a, a.length), m, n);
            InferJ.Tensor tb = InferJ.Tensor.of(Arrays.copyOf(b, b.length), 1, n);
            InferJ.Tensor out = ta.add(tb);
            out.eval();

            assertArrayEquals(new int[]{m, n}, out.dims, "add dims");
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    assertFloatEquals(a[r * n + c] + b[c], out.get(r, c), 1e-4f, "add value");
                }
            }
        }
    }

    private static void fuzzMatmul() {
        for (int it = 0; it < 250; it++) {
            int m = rnd(1, 6), k = rnd(1, 6), n = rnd(1, 6);
            float[] a = randArr(m * k);
            float[] b = randArr(k * n);

            InferJ.Tensor ta = InferJ.Tensor.of(Arrays.copyOf(a, a.length), m, k);
            InferJ.Tensor tb = InferJ.Tensor.of(Arrays.copyOf(b, b.length), k, n);
            InferJ.Tensor out = ta.matmul(tb);
            out.eval();

            assertArrayEquals(new int[]{m, n}, out.dims, "matmul dims");
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    float exp = 0;
                    for (int i = 0; i < k; i++) exp += a[r * k + i] * b[i * n + c];
                    assertFloatEquals(exp, out.get(r, c), 1e-3f, "matmul value");
                }
            }
        }
    }

    private static void fuzzScalarMul() {
        for (int it = 0; it < 300; it++) {
            int n = rnd(1, 30);
            float[] a = randArr(n);
            float s = randScalar();

            InferJ.Tensor x = InferJ.Tensor.of(Arrays.copyOf(a, a.length), n);
            InferJ.Tensor y = x.multiply(s);
            y.eval();

            for (int i = 0; i < n; i++) {
                assertFloatEquals(a[i] * s, y.get(i), 1e-4f, "scalarMul value");
            }
        }
    }

    private static void fuzzSoftmax() {
        for (int it = 0; it < 220; it++) {
            int rows = rnd(1, 7), cols = rnd(2, 9);
            float[] a = randArr(rows * cols);
            InferJ.Tensor x = InferJ.Tensor.of(Arrays.copyOf(a, a.length), rows, cols);
            InferJ.Tensor y = x.softMax(1);
            y.eval();

            for (int r = 0; r < rows; r++) {
                float sum = 0;
                float[] expv = new float[cols];
                for (int c = 0; c < cols; c++) {
                    expv[c] = (float) Math.exp(a[r * cols + c]);
                    sum += expv[c];
                }
                float outSum = 0;
                for (int c = 0; c < cols; c++) {
                    float exp = expv[c] / sum;
                    float got = y.get(r, c);
                    outSum += got;
                    assertFloatEquals(exp, got, 1e-3f, "softmax value");
                }
                assertFloatEquals(1f, outSum, 1e-3f, "softmax row sum");
            }
        }
    }

    private static void fuzzMaskedFill() {
        for (int it = 0; it < 220; it++) {
            int rows = rnd(1, 8), cols = rnd(1, 8);
            float[] a = randArr(rows * cols);
            float[] m = new float[rows * cols];
            for (int i = 0; i < m.length; i++) m[i] = RNG.nextBoolean() ? 1f : 0f;
            float fill = randScalar();

            InferJ.Tensor x = InferJ.Tensor.of(Arrays.copyOf(a, a.length), rows, cols);
            InferJ.Tensor mask = InferJ.Tensor.of(Arrays.copyOf(m, m.length), rows, cols);
            InferJ.Tensor y = x.maskedFill(mask, fill);
            y.eval();

            for (int i = 0; i < a.length; i++) {
                float exp = m[i] == 0f ? fill : a[i];
                assertFloatEquals(exp, y.arr[i], 1e-4f, "maskedFill value");
            }
        }
    }

    private static void fuzzLayerNorm() {
        for (int it = 0; it < 220; it++) {
            int rows = rnd(1, 8), cols = rnd(2, 10);
            float[] a = randArr(rows * cols);
            float[] w = randArr(cols);
            float eps = 1e-5f;

            InferJ.Tensor x = InferJ.Tensor.of(Arrays.copyOf(a, a.length), rows, cols);
            InferJ.Tensor tw = InferJ.Tensor.of(Arrays.copyOf(w, w.length), cols);
            InferJ.Tensor y = x.layerNorm(tw, eps);
            y.eval();

            for (int r = 0; r < rows; r++) {
                float mean = 0;
                for (int c = 0; c < cols; c++) mean += a[r * cols + c];
                mean /= cols;

                float var = 0;
                for (int c = 0; c < cols; c++) {
                    float d = a[r * cols + c] - mean;
                    var += d * d;
                }
                var /= cols;
                float denom = (float) Math.sqrt(var + eps);

                for (int c = 0; c < cols; c++) {
                    float exp = w[c] * ((a[r * cols + c] - mean) / denom);
                    assertFloatEquals(exp, y.get(r, c), 2e-3f, "layerNorm value");
                }
            }
        }
    }

    private static void fuzzGelu() {
        for (int it = 0; it < 280; it++) {
            int n = rnd(1, 50);
            float[] a = randArr(n);

            InferJ.Tensor x = InferJ.Tensor.of(Arrays.copyOf(a, a.length), n);
            InferJ.Tensor y = x.gelu();
            y.eval();

            for (int i = 0; i < n; i++) {
                float exp = geluRef(a[i]);
                assertFloatEquals(exp, y.get(i), 1e-4f, "gelu value");
            }
        }
    }

    private static void fuzzArgMax() {
        for (int it = 0; it < 240; it++) {
            int rows = rnd(1, 9), cols = rnd(1, 9);
            float[] a = randArr(rows * cols);

            InferJ.Tensor x = InferJ.Tensor.of(Arrays.copyOf(a, a.length), rows, cols);
            InferJ.Tensor y = x.argMax(1);
            y.eval();

            assertArrayEquals(new int[]{rows, 1}, y.dims, "argMax dims");
            for (int r = 0; r < rows; r++) {
                int bi = 0;
                float bv = a[r * cols];
                for (int c = 1; c < cols; c++) {
                    float v = a[r * cols + c];
                    if (v > bv) {
                        bv = v;
                        bi = c;
                    }
                }
                assertFloatEquals(bi, y.get(r, 0), 1e-4f, "argMax value");
            }
        }
    }

    private static void testSortContract() {
        InferJ.Tensor ascRows = InferJ.Tensor.of(new float[]{
                3, 1, 2,
                9, 7, 8
        }, 2, 3).sort(1, InferJ.Tensor.SortOrder.ASC);
        ascRows.eval();
        InferJ.Tensor ascVals = ascRows.sub(0);
        InferJ.Tensor ascIdx = ascRows.sub(1);

        assertArrayEquals(new int[]{2, 3}, ascVals.dims, "sort asc values dims");
        assertArrayEquals(new int[]{2, 3}, ascIdx.dims, "sort asc indices dims");
        assertFloatEquals(1f, ascVals.get(0, 0), 1e-6f, "sort asc rows value 0");
        assertFloatEquals(2f, ascVals.get(0, 1), 1e-6f, "sort asc rows value 1");
        assertFloatEquals(3f, ascVals.get(0, 2), 1e-6f, "sort asc rows value 2");
        assertFloatEquals(7f, ascVals.get(1, 0), 1e-6f, "sort asc rows value 3");
        assertFloatEquals(8f, ascVals.get(1, 1), 1e-6f, "sort asc rows value 4");
        assertFloatEquals(9f, ascVals.get(1, 2), 1e-6f, "sort asc rows value 5");
        assertFloatEquals(1f, ascIdx.get(0, 0), 1e-6f, "sort asc rows idx 0");
        assertFloatEquals(2f, ascIdx.get(0, 1), 1e-6f, "sort asc rows idx 1");
        assertFloatEquals(0f, ascIdx.get(0, 2), 1e-6f, "sort asc rows idx 2");
        assertFloatEquals(1f, ascIdx.get(1, 0), 1e-6f, "sort asc rows idx 3");
        assertFloatEquals(2f, ascIdx.get(1, 1), 1e-6f, "sort asc rows idx 4");
        assertFloatEquals(0f, ascIdx.get(1, 2), 1e-6f, "sort asc rows idx 5");

        InferJ.Tensor descCols = InferJ.Tensor.of(new float[]{
                3, 1, 2,
                9, 7, 8,
                6, 4, 5
        }, 3, 3).sort(0, InferJ.Tensor.SortOrder.DESC);
        descCols.eval();
        InferJ.Tensor descVals = descCols.sub(0);
        InferJ.Tensor descIdx = descCols.sub(1);

        assertArrayEquals(new int[]{3, 3}, descVals.dims, "sort desc values dims");
        assertArrayEquals(new int[]{3, 3}, descIdx.dims, "sort desc indices dims");
        assertFloatEquals(9f, descVals.get(0, 0), 1e-6f, "sort desc cols value 0");
        assertFloatEquals(7f, descVals.get(0, 1), 1e-6f, "sort desc cols value 1");
        assertFloatEquals(8f, descVals.get(0, 2), 1e-6f, "sort desc cols value 2");
        assertFloatEquals(6f, descVals.get(1, 0), 1e-6f, "sort desc cols value 3");
        assertFloatEquals(4f, descVals.get(1, 1), 1e-6f, "sort desc cols value 4");
        assertFloatEquals(5f, descVals.get(1, 2), 1e-6f, "sort desc cols value 5");
        assertFloatEquals(3f, descVals.get(2, 0), 1e-6f, "sort desc cols value 6");
        assertFloatEquals(1f, descVals.get(2, 1), 1e-6f, "sort desc cols value 7");
        assertFloatEquals(2f, descVals.get(2, 2), 1e-6f, "sort desc cols value 8");
        assertFloatEquals(1f, descIdx.get(0, 0), 1e-6f, "sort desc cols idx 0");
        assertFloatEquals(1f, descIdx.get(0, 1), 1e-6f, "sort desc cols idx 1");
        assertFloatEquals(1f, descIdx.get(0, 2), 1e-6f, "sort desc cols idx 2");
        assertFloatEquals(2f, descIdx.get(1, 0), 1e-6f, "sort desc cols idx 3");
        assertFloatEquals(2f, descIdx.get(1, 1), 1e-6f, "sort desc cols idx 4");
        assertFloatEquals(2f, descIdx.get(1, 2), 1e-6f, "sort desc cols idx 5");
        assertFloatEquals(0f, descIdx.get(2, 0), 1e-6f, "sort desc cols idx 6");
        assertFloatEquals(0f, descIdx.get(2, 1), 1e-6f, "sort desc cols idx 7");
        assertFloatEquals(0f, descIdx.get(2, 2), 1e-6f, "sort desc cols idx 8");
    }

    private static void testTopKContract() {
        InferJ.Tensor topk = InferJ.Tensor.of(new float[]{1, 7, 3, 9, 5}, 1, 5).topK(1, 3);
        topk.eval();
        InferJ.Tensor topkVals = topk.sub(0);
        InferJ.Tensor topkIdx = topk.sub(1);
        assertArrayEquals(new int[]{1, 5}, topkVals.dims, "topK values dims");
        assertFloatEquals(9f, topkVals.get(0, 0), 1e-6f, "topK value 0");
        assertFloatEquals(7f, topkVals.get(0, 1), 1e-6f, "topK value 1");
        assertFloatEquals(5f, topkVals.get(0, 2), 1e-6f, "topK value 2");
        assertFloatEquals(0f, topkVals.get(0, 3), 1e-6f, "topK value 3");
        assertFloatEquals(0f, topkVals.get(0, 4), 1e-6f, "topK value 4");
        assertFloatEquals(3f, topkIdx.get(0, 0), 1e-6f, "topK idx 0");
        assertFloatEquals(1f, topkIdx.get(0, 1), 1e-6f, "topK idx 1");
        assertFloatEquals(4f, topkIdx.get(0, 2), 1e-6f, "topK idx 2");
        assertFloatEquals(0f, topkIdx.get(0, 3), 1e-6f, "topK idx 3");
        assertFloatEquals(0f, topkIdx.get(0, 4), 1e-6f, "topK idx 4");
    }

    private static void testSubContract() {
        InferJ.Tensor sub3d = InferJ.Tensor.of(new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
                17, 18, 19, 20,
                21, 22, 23, 24
        }, 2, 3, 4);
        InferJ.Tensor row1 = sub3d.sub(1);
        InferJ.Tensor cell = sub3d.sub(1, 2);
        assertArrayEquals(new int[]{3, 4}, row1.dims, "sub dims");
        assertArrayEquals(new int[]{4}, cell.dims, "sub leaf dims");
        assertFloatEquals(13f, row1.get(0, 0), 1e-6f, "sub value 0");
        assertFloatEquals(16f, row1.get(0, 3), 1e-6f, "sub value 1");
        assertFloatEquals(21f, cell.get(0), 1e-6f, "sub leaf value 0");
        assertFloatEquals(24f, cell.get(3), 1e-6f, "sub leaf value 3");
    }

    private static void testMultinomialContract() {
        InferJ.Tensor probs = InferJ.Tensor.of(new float[]{
                0f, 1f, 0f,
                1f, 0f, 0f
        }, 2, 3);
        InferJ.Tensor choices = InferJ.Tensor.of(new float[]{
                10f, 20f, 30f,
                40f, 50f, 60f
        }, 2, 3);
        InferJ.Tensor sampled = probs.multinomial(choices);
        sampled.eval();
        assertArrayEquals(new int[]{2, 1}, sampled.dims, "multinomial dims");
        assertFloatEquals(20f, sampled.get(0, 0), 1e-6f, "multinomial row0");
        assertFloatEquals(40f, sampled.get(1, 0), 1e-6f, "multinomial row1");
    }

    private static void testMaterializeOps() {
        InferJ.Tensor base = InferJ.Tensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        InferJ.Tensor bias = InferJ.Tensor.of(new float[]{10, 20}, 1, 2);
        InferJ.Tensor out = base.add(bias).materialize();

        assertTrue(out.op == InferJ.Tensor.Op.NO_OP, "materialize should clear op");
        assertTrue(out.op1 == null && out.op2 == null, "materialize should clear parents");
        assertArrayEquals(new int[]{2, 2}, out.dims, "materialize dims");
        assertFloatEquals(11f, out.get(0, 0), 1e-6f, "materialize value 0");
        assertFloatEquals(22f, out.get(0, 1), 1e-6f, "materialize value 1");
        assertFloatEquals(13f, out.get(1, 0), 1e-6f, "materialize value 2");
        assertFloatEquals(24f, out.get(1, 1), 1e-6f, "materialize value 3");

        out.eval();
        assertFloatEquals(11f, out.get(0, 0), 1e-6f, "materialize idempotent value 0");
        assertFloatEquals(24f, out.get(1, 1), 1e-6f, "materialize idempotent value 3");
    }

    private static void fuzzCrossOpPipeline() {
        for (int it = 0; it < 120; it++) {
            int m = rnd(1, 5), k = rnd(1, 5), n = rnd(2, 6);
            float[] a = randArr(m * k);
            float[] b = randArr(k * n);
            float[] bias = randArr(n);

            InferJ.Tensor ta = InferJ.Tensor.of(Arrays.copyOf(a, a.length), m, k);
            InferJ.Tensor tb = InferJ.Tensor.of(Arrays.copyOf(b, b.length), k, n);
            InferJ.Tensor tBias = InferJ.Tensor.of(Arrays.copyOf(bias, bias.length), 1, n);

            InferJ.Tensor out = ta.matmul(tb).add(tBias).gelu().softMax(1);
            out.eval();

            float[] mat = new float[m * n];
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    float s = 0;
                    for (int i = 0; i < k; i++) s += a[r * k + i] * b[i * n + c];
                    s += bias[c];
                    s = geluRef(s);
                    mat[r * n + c] = s;
                }
            }

            for (int r = 0; r < m; r++) {
                float sum = 0;
                float[] ex = new float[n];
                for (int c = 0; c < n; c++) {
                    ex[c] = (float) Math.exp(mat[r * n + c]);
                    sum += ex[c];
                }
                float outSum = 0;
                for (int c = 0; c < n; c++) {
                    float exp = ex[c] / sum;
                    float got = out.get(r, c);
                    outSum += got;
                    assertFloatEquals(exp, got, 2e-3f, "cross-op value");
                }
                assertFloatEquals(1f, outSum, 2e-3f, "cross-op row sum");
            }
        }
    }

    private static void fuzzTransformedAddSoftmax() {
        for (int it = 0; it < 140; it++) {
            int rows = rnd(2, 7), cols = rnd(2, 7);
            float[] src = randArr(rows * cols);

            int r0 = rnd(0, rows - 2), r1 = rnd(r0 + 1, rows);
            InferJ.Tensor base = InferJ.Tensor.of(Arrays.copyOf(src, src.length), rows, cols);
            InferJ.Tensor sliced = base.slice(InferJ.$(r0, r1), InferJ.$(0, cols));
            InferJ.Tensor bias = InferJ.Tensor.of(randArr(cols), 1, cols);
            InferJ.Tensor out = sliced.add(bias).softMax(1);
            out.eval();

            int outRows = r1 - r0;
            for (int r = 0; r < outRows; r++) {
                float[] ex = new float[cols];
                float sum = 0;
                for (int c = 0; c < cols; c++) {
                    float v = src[(r0 + r) * cols + c] + bias.get(0, c);
                    ex[c] = (float) Math.exp(v);
                    sum += ex[c];
                }
                float rowSum = 0;
                for (int c = 0; c < cols; c++) {
                    float exp = ex[c] / sum;
                    float got = out.get(r, c);
                    rowSum += got;
                    assertFloatEquals(exp, got, 2e-3f, "transformed add+softmax value");
                }
                assertFloatEquals(1f, rowSum, 2e-3f, "transformed add+softmax row sum");
            }
        }
    }

    private static void fuzzTransformedMatmulChain() {
        for (int it = 0; it < 120; it++) {
            int m = rnd(2, 6), k = rnd(2, 6), n = rnd(2, 6);
            float[] a = randArr(m * k);
            float[] b = randArr(k * n);

            InferJ.Tensor ta = InferJ.Tensor.of(Arrays.copyOf(a, a.length), m, k);
            InferJ.Tensor tb = InferJ.Tensor.of(Arrays.copyOf(b, b.length), k, n);

            InferJ.Tensor at = ta.transpose(0, 1);
            InferJ.Tensor out = at.transpose(0, 1).matmul(tb.view(k, n));
            out.eval();

            for (int r = 0; r < m; r++) {
                for (int c = 0; c < n; c++) {
                    float exp = 0;
                    for (int i = 0; i < k; i++) exp += a[r * k + i] * b[i * n + c];
                    assertFloatEquals(exp, out.get(r, c), 1e-3f, "transformed matmul chain value");
                }
            }
        }
    }

    private static void fuzzTransformedSplitSelectMaskedFill() {
        for (int it = 0; it < 120; it++) {
            int rows = rnd(2, 6), half = rnd(1, 5), cols = half * 2;
            float[] src = randArr(rows * cols);
            InferJ.Tensor t = InferJ.Tensor.of(Arrays.copyOf(src, src.length), rows, cols);

            InferJ.Tensor[] sp = t.split(2, 1);
            InferJ.Tensor right = sp[1];
            float[] idxf = new float[]{rnd(0, rows - 1), rnd(0, rows - 1)};
            InferJ.Tensor idx = InferJ.Tensor.of(idxf, 2);
            InferJ.Tensor sel = right.select(idx);

            float[] maskArr = new float[2 * half];
            for (int i = 0; i < maskArr.length; i++) maskArr[i] = RNG.nextBoolean() ? 1f : 0f;
            InferJ.Tensor mask = InferJ.Tensor.of(maskArr, 2, half);
            float fill = randScalar();

            InferJ.Tensor out = sel.maskedFill(mask, fill);
            out.eval();

            for (int r = 0; r < 2; r++) {
                int srcRow = (int) idxf[r];
                for (int c = 0; c < half; c++) {
                    float orig = src[srcRow * cols + (half + c)];
                    float exp = maskArr[r * half + c] == 0f ? fill : orig;
                    assertFloatEquals(exp, out.get(r, c), 1e-4f, "transformed split/select/maskedFill value");
                }
            }
        }
    }

    private static void testEdgeCases() {
        testEdgeViewInferSingleNegativeDimension();
        testEdgeViewRejectsMultipleNegativeDimensions();
        testEdgeSplitRejectsUnevenPartitions();
        testEdgeMatmulRejectsMismatchedDimensions();
        testEdgeSelectWithRepeatedIndices();
        testEdgeMaskedFillAllMasked();
        testEdgeContiguousAfterTranspose();
        testEdgeSoftmaxLargeFiniteInputs();
    }

    private static void testEdgeViewInferSingleNegativeDimension() {
        InferJ.Tensor t = InferJ.Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        InferJ.Tensor v = t.view(-1, 3);
        assertArrayEquals(new int[]{2, 3}, v.dims, "view(-1,3) dims");
    }

    private static void testEdgeViewRejectsMultipleNegativeDimensions() {
        InferJ.Tensor t = InferJ.Tensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        assertThrows(() -> t.view(-1, -1), IllegalArgumentException.class, "view should reject multiple -1 dims");
    }

    private static void testEdgeSplitRejectsUnevenPartitions() {
        InferJ.Tensor t = InferJ.Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertThrows(() -> t.split(2, 1), IllegalArgumentException.class, "split should reject uneven partitions");
    }

    private static void testEdgeMatmulRejectsMismatchedDimensions() {
        InferJ.Tensor a = InferJ.Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        InferJ.Tensor b = InferJ.Tensor.of(new float[]{1, 2, 3, 4}, 2, 2);
        assertThrows(() -> a.matmul(b), IllegalArgumentException.class, "matmul mismatch should throw");
    }

    private static void testEdgeSelectWithRepeatedIndices() {
        InferJ.Tensor src = InferJ.Tensor.of(new float[]{
                1, 2, 3,
                4, 5, 6
        }, 2, 3);
        InferJ.Tensor idx = InferJ.Tensor.of(new float[]{1, 1, 0}, 3);
        InferJ.Tensor out = src.select(idx);
        out.eval();

        assertArrayEquals(new int[]{3, 3}, out.dims, "select repeated dims");
        assertFloatEquals(4f, out.get(0, 0), 1e-5f, "select repeated row0");
        assertFloatEquals(4f, out.get(1, 0), 1e-5f, "select repeated row1");
        assertFloatEquals(1f, out.get(2, 0), 1e-5f, "select repeated row2");
    }

    private static void testEdgeMaskedFillAllMasked() {
        InferJ.Tensor x = InferJ.Tensor.of(new float[]{9, 8, 7, 6}, 2, 2);
        InferJ.Tensor mask = InferJ.Tensor.of(new float[]{0, 0, 0, 0}, 2, 2);
        InferJ.Tensor y = x.maskedFill(mask, -123.5f);
        y.eval();
        for (int i = 0; i < 4; i++) {
            assertFloatEquals(-123.5f, y.arr[i], 1e-6f, "maskedFill all-masked");
        }
    }

    private static void testEdgeContiguousAfterTranspose() {
        InferJ.Tensor x = InferJ.Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        InferJ.Tensor y = x.transpose(0, 1).contiguous().view(6);
        y.eval();
        assertArrayEquals(new int[]{6}, y.dims, "contiguous view dims");
        assertFloatEquals(1f, y.get(0), 1e-6f, "contiguous first");
        assertFloatEquals(4f, y.get(1), 1e-6f, "contiguous second");
        assertFloatEquals(6f, y.get(5), 1e-6f, "contiguous last");
    }

    private static void testEdgeSoftmaxLargeFiniteInputs() {
        InferJ.Tensor x = InferJ.Tensor.of(new float[]{30f, 31f, 32f, 29f}, 1, 4);
        InferJ.Tensor y = x.softMax(1);
        y.eval();
        float sum = 0f;
        for (int i = 0; i < 4; i++) {
            float v = y.get(0, i);
            if (!Float.isFinite(v)) {
                throw new AssertionError("softmax produced non-finite value");
            }
            sum += v;
        }
        assertFloatEquals(1f, sum, 1e-3f, "softmax large finite sum");
    }

    private static void testAttentionContiguousFlattenLayout() {
        int bsz = 1, heads = 2, time = 3, headDim = 4;
        InferJ.Tensor x = attentionLayoutTensor(bsz, heads, time, headDim);
        InferJ.Tensor flat = x.transpose(1, 2).contiguous().view(bsz, time, heads * headDim);
        flat.eval();

        for (int b = 0; b < bsz; b++) {
            for (int t = 0; t < time; t++) {
                for (int h = 0; h < heads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        float exp = 1000 * b + 100 * h + 10 * t + d;
                        int c = h * headDim + d;
                        assertFloatEquals(exp, flat.get(b, t, c), 1e-6f, "attention contiguous flatten value");
                    }
                }
            }
        }
    }

    private static void testAttention4dTransposeLayout() {
        int bsz = 1, heads = 2, time = 3, headDim = 4;
        InferJ.Tensor x = attentionLayoutTensor(bsz, heads, time, headDim);
        InferJ.Tensor tr = x.transpose(1, 2);

        assertArrayEquals(new int[]{bsz, time, heads, headDim}, tr.dims, "attention transpose dims");
        for (int b = 0; b < bsz; b++) {
            for (int t = 0; t < time; t++) {
                for (int h = 0; h < heads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        float exp = 1000 * b + 100 * h + 10 * t + d;
                        assertFloatEquals(exp, tr.get(b, t, h, d), 1e-6f, "attention transpose value");
                    }
                }
            }
        }
    }

    private static InferJ.Tensor attentionLayoutTensor(int bsz, int heads, int time, int headDim) {
        float[] src = new float[bsz * heads * time * headDim];
        int p = 0;
        for (int b = 0; b < bsz; b++) {
            for (int h = 0; h < heads; h++) {
                for (int t = 0; t < time; t++) {
                    for (int d = 0; d < headDim; d++) {
                        src[p++] = 1000 * b + 100 * h + 10 * t + d;
                    }
                }
            }
        }

        return InferJ.Tensor.of(src, bsz, heads, time, headDim);
    }

    private static void testSplitNonLastDimensionLayout() {
        int a = 4, b = 3, c = 2;
        float[] src = new float[a * b * c];
        int p = 0;
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                for (int k = 0; k < c; k++) {
                    src[p++] = 100 * i + 10 * j + k;
                }
            }
        }

        InferJ.Tensor x = InferJ.Tensor.of(src, a, b, c);
        InferJ.Tensor[] parts = x.split(2, 0);
        assertArrayEquals(new int[]{2, b, c}, parts[0].dims, "split dim0 part0 dims");
        assertArrayEquals(new int[]{2, b, c}, parts[1].dims, "split dim0 part1 dims");

        for (int part = 0; part < 2; part++) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < b; j++) {
                    for (int k = 0; k < c; k++) {
                        int srcI = part * 2 + i;
                        float exp = 100 * srcI + 10 * j + k;
                        assertFloatEquals(exp, parts[part].get(i, j, k), 1e-6f, "split non-last value");
                    }
                }
            }
        }
    }

    private static float geluRef(float x) {
        float c = (float) Math.sqrt(2.0 / Math.PI);
        return (float) (0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x))));
    }

    private static float[] randArr(int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = randScalar();
        return a;
    }

    private static float randScalar() {
        return (RNG.nextFloat() - 0.5f) * 6f;
    }

    private static int rnd(int lo, int hi) {
        return lo + RNG.nextInt(hi - lo + 1);
    }

    private static void assertFloatEquals(float expected, float actual, float eps, String message) {
        if (Math.abs(expected - actual) > eps) {
            throw new AssertionError(message + ": expected=" + expected + ", actual=" + actual);
        }
    }

    private static void assertIntEquals(int expected, int actual, String message) {
        if (expected != actual) {
            throw new AssertionError(message + ": expected=" + expected + ", actual=" + actual);
        }
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void assertArrayEquals(int[] expected, int[] actual, String message) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError(message + ": expected=" + Arrays.toString(expected)
                    + ", actual=" + Arrays.toString(actual));
        }
    }

    private static void assertThrows(Runnable call, Class<? extends Throwable> expectedType, String message) {
        try {
            call.run();
        } catch (Throwable t) {
            if (expectedType.isInstance(t)) {
                return;
            }
            throw new AssertionError(message + ": expected exception " + expectedType.getSimpleName()
                    + ", got " + t.getClass().getSimpleName(), t);
        }
        throw new AssertionError(message + ": expected exception but nothing was thrown");
    }
}
