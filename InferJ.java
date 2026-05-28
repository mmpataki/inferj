import java.io.*;
import java.lang.reflect.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.Function;

public class InferJ {


    public static class Tensor {

        public enum SortOrder {ASC, DESC}

        record Pair<A, B>(A a, B b) {
        }

        enum Op {
            NO_OP,
            MAT_MUL,
            MAT_ADD,
            SCALAR_MUL,
            SELECT,
            LAYER_NORM,
            MASKED_FILL,
            SOFTMAX,
            GELU,
            ARGMAX,
            VIEW,
            TRANSPOSE,
            SPLIT,
            CONTIGUOUS,
            SLICE,
            SORT,
            MULTINOMIAL;
        }

        boolean continuous = true;
        Op op;
        Tensor op1, op2;
        String name;

        float arr[];
        int[] dims;
        int[] strides;
        int offset = 0;
        int _size = -1;

        Object cookie;
        boolean evaled = false;

        private Tensor(Tensor a, Tensor b, Op oper, float[] arr, int offset, Object cookie, boolean continuous,
                       int[] strides, int[] dims) {
            op1 = a;
            op2 = b;
            this.op = oper;
            this.dims = dims;
            this.arr = arr == null ? new float[size()] : arr;
            this.continuous = continuous;
            this.offset = offset;
            this.strides = strides;
            this.cookie = cookie;
            if (strides == null) {
                this.strides = new int[dims.length];
                for (int i = dims.length - 1, j = 1; i >= 0; j *= dims[i], i--) {
                    this.strides[i] = j;
                }
            }
        }

        public Tensor(float[] arr, int... dims) {
            this(null, null, Op.NO_OP, arr, 0, null, true, null, dims);
        }

        public static Tensor of(float[] arr, int... dims) {
            arr = arr != null ? arr : new float[Arrays.stream(dims).reduce(1, (a, b) -> a * b)];
            return new Tensor(arr, dims);
        }

        public static Tensor of(int... dims) {
            return Tensor.of(null, dims);
        }

        TensorIt it(int iter_rank) {
            return new TensorIt(this, iter_rank);
        }

        TensorIt it() {
            return it(continousDims());
        }

        public static Tensor ltm(int side) {
            float[] arr = new float[side * side];
            for (int i = 0; i < side; i++) {
                int row = i * side;
                for (int j = row; j <= row + i; j++) arr[j] = 1;
                for (int j = row + i + 1; j < row + side; j++) arr[j] = 0;
            }
            return Tensor.of(arr, side, side);
        }

        public Tensor withName(String name) {
            this.name = name;
            return this;
        }

        public int index(int... indices) {
            int idx = offset;
            for (int i = 0; i < indices.length; i++) {
                idx += indices[i] * strides[i];
            }
            return idx;
        }

        public float get(int... idx) {
            return arr[index(idx)];
        }

        public void set(float val, int... idx) {
            arr[index(idx)] = val;
        }

        public Tensor eval() {
            if (evaled) return this;
            evaled = true;
            if (op1 != null)
                op1.eval();
            if (op2 != null)
                op2.eval();
            switch (op) {
                case MAT_MUL -> _matmul(op1, op2);
                case MAT_ADD -> _add(op1, op2, 0);
                case SCALAR_MUL -> _apply(op1, x -> x * (float) cookie);
                case SELECT -> _select(op1, op2);
                case LAYER_NORM -> _layer_norm(op1, op2, (float) cookie);
                case MASKED_FILL -> _masked_fill(op1, op2, (float) cookie);
                case SOFTMAX -> _softmax(op1, (int) cookie);
                case GELU -> _apply(op1, this::_gelu);
                case ARGMAX -> _argmax(op1, (int) cookie);
                case CONTIGUOUS -> _contiguous(op1);
                case SORT -> _top_k(op1, (SortParams) cookie);
                case MULTINOMIAL -> _multinomial(op1, op2);
            }
            return this;
        }

        public int dim(int dIdx) {
            if (dIdx < 0) dIdx += dims.length;
            return dims[dIdx];
        }

        public int size() {
            return _size != -1 ? _size : (_size = Arrays.stream(dims).reduce(1, (a, b) -> a * b));
        }

        private Tensor[] broadcast_dims(Tensor x, Tensor y, int r_offset) {

            int res_ndim = Math.max(x.dims.length, y.dims.length);
            int[] rdim1 = new int[res_ndim];
            int[] rdim2 = new int[res_ndim];
            int[] strd1 = new int[res_ndim];
            int[] strd2 = new int[res_ndim];
            boolean con1 = x.continuous;
            boolean con2 = y.continuous;

            // validate
            for (int d1i = x.dims.length - 1, d2i = y.dims.length - 1, ri = res_ndim - 1; d1i >= 0
                    || d2i >= 0; d1i--, d2i--, ri--) {

                if (r_offset > 0) {
                    strd1[ri] = d1i >= 0 ? x.strides[d1i] : 0;
                    strd2[ri] = d2i >= 0 ? y.strides[d2i] : 0;
                    rdim1[ri] = x.dims[d1i];
                    rdim2[ri] = y.dims[d2i];
                    r_offset--;
                    continue;
                }

                int xd = d1i >= 0 ? x.dims[d1i] : 1;
                int yd = d2i >= 0 ? y.dims[d2i] : 1;
                int xs = d1i >= 0 ? x.strides[d1i] : 0;
                int ys = d2i >= 0 ? y.strides[d2i] : 0;

                // either equal or one of them 1
                if (xd != yd && xd != 1 && yd != 1)
                    throw new IllegalArgumentException("matrix dimension don't allow broadcast");

                rdim1[ri] = rdim2[ri] = Math.max(xd, yd);

                strd1[ri] = xd == yd ? xs : (xd == 1 ? 0 : xs);
                strd2[ri] = xd == yd ? ys : (yd == 1 ? 0 : ys);
                con1 &= strd1[ri] == 0;
                con2 &= strd2[ri] == 0;
            }

            return new Tensor[]{
                    new Tensor(x.op1, x.op2, x.op, x.arr, x.offset, x.cookie, con1, strd1, rdim1),
                    new Tensor(y.op1, y.op2, y.op, y.arr, y.offset, y.cookie, con2, strd2, rdim2)
            };
        }

        private int continousDims() {
            for (int i = dims.length - 1, c = 0; i > 0; i--, c++) {
                if (strides[i - 1] != strides[i] * dims[i])
                    return c;
            }
            return dims.length;
        }

        private int[] addDims(int[] dims, boolean left, int... new_dims) {
            int[] n_dims = new int[dims.length + new_dims.length];
            System.arraycopy(dims, 0, n_dims, left ? new_dims.length : 0, dims.length);
            System.arraycopy(new_dims, 0, n_dims, left ? 0 : dims.length, new_dims.length);
            return n_dims;
        }

        private static boolean isContinuous(Tensor t, int last_n) {
            if (last_n == 0) return t.strides[t.dims.length - 1] == 1;
            if (last_n == t.dims.length) return t.continuous;
            return t.strides[t.dims.length - 1 - last_n] == t.strides[t.dims.length - last_n] * t.dims[t.dims.length - last_n];
        }

        static class TensorIt implements Iterator<Tensor> {

            boolean _isContinuous;
            int view_rank;
            int[] idx, _strides, _dims;
            Tensor t;
            int offset = -1;

            /**
             * pass 1 to iterate on vectors, 2 for matrices and so on..
             */
            public TensorIt(Tensor t, int view_rank) {
                this.t = t;
                this.view_rank = view_rank;
                idx = new int[t.dims.length];
                Arrays.fill(idx, 0);

                _strides = Arrays.copyOfRange(t.strides, t.dims.length - view_rank, t.dims.length);
                _dims = Arrays.copyOfRange(t.dims, t.dims.length - view_rank, t.dims.length);
                _isContinuous = isContinuous(t, view_rank);

                make();
            }

            @Override
            public boolean hasNext() {
                if (offset != -1) return true;
                for (int i = idx.length - 1 - view_rank; i >= 0; i--) {
                    idx[i]++;
                    if (idx[i] < t.dims[i]) {
                        make();
                        return true;
                    }
                    idx[i] = 0;
                }
                return false;
            }

            @Override
            public Tensor next() {
                if (offset == -1)
                    throw new NoSuchElementException();
                Tensor ret = new Tensor(null, null, Op.NO_OP, t.arr, offset, null, _isContinuous, _strides, _dims);
                offset = -1;
                return ret;
            }

            public int nextIdx() {
                int off = offset;
                offset = -1;
                return off;
            }

            private void make() {
                offset = t.index(idx);
            }
        }

        // PRINTING
        private StringBuilder toString(int level) {
            return (new StringBuilder("\n"))
                    .repeat("|   ", level).append("`---")
                    .append(name == null ? op.toString() : name).append(" ").append(Arrays.toString(dims))
                    .append(op1 != null ? op1.toString(level + 1) : "")
                    .append(op2 != null ? op2.toString(level + 1) : "");
        }

        public String asTree() {
            return toString(0).toString();
        }

        public String asString() {
            int[] idx = new int[dims.length];
            Arrays.fill(idx, 0);
            StringBuilder sb = new StringBuilder();
            sb.repeat("[", idx.length);
            for (int op = 0; op < size(); op++) {
                sb.append(get(idx)).append(" ");
                int updates = 0;
                for (int i = idx.length - 1; i >= 0; i--, updates++) {
                    idx[i]++;
                    if (idx[i] < dims[i])
                        break;
                    idx[i] = 0;
                }
                if (updates != idx.length)
                    sb.repeat("]\n", updates).repeat("[ ", updates);
            }
            sb.repeat("]", idx.length);
            return sb.toString();
        }

        @Override
        public String toString() {
            return (name == null ? op.toString() : name) + " " + Arrays.toString(dims);
        }


        // OPERATORS
        public Tensor slice(int[]... new_dims) {

            if (new_dims.length != dims.length)
                throw new IllegalArgumentException("ranges for all dimensions should be specified");

            int new_offset = offset;
            int[] _new_dims = new int[dims.length];

            for (int i = 0; i < new_dims.length; i++) {
                int[] rng = new_dims[i];
                switch (rng.length) {
                    case 0 -> rng = new int[]{0, dims[i]};
                    case 1 -> rng = new int[]{rng[0], rng[0] + 1};
                }
                _new_dims[i] = rng[1] - rng[0];
                new_offset += rng[0] * strides[i];
            }

            return new Tensor(this, null, Op.SLICE, arr, new_offset, null, true, strides, _new_dims);
        }

        public Tensor[] split(int n, int dim) {
            if (dims[dim] % n != 0)
                throw new IllegalArgumentException("can't split evenly");
            int sz = strides[dim] * (dims[dim] / n);
            if (sz == 0)
                throw new IllegalArgumentException("zero size splits");
            Tensor[] t = new Tensor[n];
            int[] c_dims = Arrays.copyOf(dims, dims.length);
            c_dims[dim] = dims[dim] / n;
            for (int i = 0, n_offset = offset; i < n; i++, n_offset += sz) {
                t[i] = new Tensor(this, null, Op.SPLIT, arr, n_offset, null, continuous, strides, c_dims);
            }
            return t;
        }

        public Tensor transpose(int axisA, int axisB) {
            int[] n_dims = Arrays.copyOf(dims, dims.length);
            n_dims[axisA] = dims[axisB];
            n_dims[axisB] = dims[axisA];
            int[] n_strides = Arrays.copyOf(strides, strides.length);
            n_strides[axisA] = strides[axisB];
            n_strides[axisB] = strides[axisA];
            return new Tensor(this, null, Op.TRANSPOSE, arr, offset, null, false, n_strides, n_dims);
        }

        public Tensor view(int... new_dims) {
            int curSize = size(), newSize = 1, neg = -1;
            for (int i = 0; i < new_dims.length; i++) {
                if (new_dims[i] != -1)
                    newSize *= new_dims[i];
                else if (neg == -1)
                    neg = i;
                else
                    throw new IllegalArgumentException("multiple dimensions provided as -1");
            }

            if ((neg == -1 && newSize != curSize) || (curSize % newSize != 0)) {
                throw new IllegalArgumentException("new size > curSize or dimensions are not compatible for view");
            }

            if (neg != -1) {
                new_dims[neg] = curSize / newSize;
            }

            int[] new_strides = new int[new_dims.length];
            int odims = dims.length, ndims = new_dims.length;
            int oi = odims - 1, ni = ndims - 1;
            for (; oi >= 0 && ni >= 0; oi--) {

                // first stride is same
                new_strides[ni] = strides[oi];

                // find largest affine block in old dims
                int old_affine_block_size = dims[oi];
                while (oi > 0 && strides[oi - 1] == strides[oi] * dims[oi]) {
                    old_affine_block_size *= dims[oi - 1];
                    oi--;
                }

                // now try to match using new dims, exact match
                int new_affine_block_size = 1;

                while (ni >= 0) {
                    if (new_dims[ni] * new_affine_block_size > old_affine_block_size)
                        break;
                    new_affine_block_size *= new_dims[ni];
                    ni--;
                    if (ni >= 0)
                        new_strides[ni] = new_strides[ni + 1] * new_dims[ni + 1];
                }

                if (old_affine_block_size > new_affine_block_size) {
                    throw new IllegalArgumentException("can't create view");
                }
            }

            if (ni != -1)
                throw new IllegalArgumentException("can't create view");

            return new Tensor(this, null, Op.VIEW, arr, offset, null, continuous, new_strides, new_dims);
        }

        public Tensor select(Tensor indices) {
            Tensor x = this;
            if (x.dims.length != 2) {
                throw new IllegalArgumentException("selection can be done on a 2d tensor, found " + x.dims.length);
            }
            return new Tensor(
                    x,
                    indices,
                    Op.SELECT,
                    new float[x.dim(-1) * indices.size()],
                    0,
                    null,
                    true,
                    null,
                    addDims(indices.dims, false, x.dim(-1))
            );
        }

        private void _select(Tensor x, Tensor indices) {

            TensorIt it = indices.it(1);
            TensorIt rIt = this.it(2);
            while (it.hasNext() && rIt.hasNext()) {
                Tensor xx = it.next(), rr = rIt.next();

                int n_dims = x.dims.length;
                int vec_len = x.dims[n_dims - 1];

                for (int i = 0; i < xx.size(); i++) {
                    int idx = (int) xx.get(i);
                    System.arraycopy(x.arr, x.index(idx, 0), rr.arr, rr.offset + i * vec_len, vec_len);
                }
            }
        }

        public Tensor sub(int... idx) {
            int[][] indices = new int[this.dims.length][];
            for (int i = 0; i < dims.length; i++) {
                if (i < idx.length) {
                    if (Math.abs(idx[i]) >= dims[i])
                        throw new IllegalArgumentException("idx " + Arrays.toString(idx) + " is out of bounds for dims " + Arrays.toString(dims));
                    indices[i] = $(idx[i]);
                } else {
                    indices[i] = $();
                }
            }
            return slice(indices).view(Arrays.copyOfRange(dims, idx.length, dims.length));
        }

        record SortParams(int dim, SortOrder order, int topK) {
        }

        public Tensor sort(int dim, SortOrder order) {
            return topK(dim, -1, order);
        }

        public Tensor topK(int dim, int topK) {
            return topK(dim, topK, SortOrder.DESC);
        }

        private Tensor topK(int dim, int topK, SortOrder order) {
            int[] n_dims = addDims(dims, true, 2);
            n_dims[dim + 1] = topK;
            int size = Arrays.stream(n_dims).reduce(1, (a, b) -> a * b);
            return new Tensor(this, null, Op.SORT, new float[size], 0, new SortParams(dim, order, topK), true, null, n_dims);
        }

        private void _top_k(Tensor t, SortParams sParams) {

            final int dim = sParams.dim;
            final int k = sParams.topK;
            SortOrder order = sParams.order;

            Tensor values = this.sub(0), index = this.sub(1);
            for (int i = dim; i < t.dims.length - 1; i++) {
                t = t.transpose(i, i + 1);
                values = values.transpose(i, i + 1);
                index = index.transpose(i, i + 1);
            }

            int maxCDims = t.continousDims();
            TensorIt sIt = t.it(1), vIt = values.it(1), iIt = index.it(1);
            float[] buf = new float[t.dim(-1)];
            while (sIt.hasNext() && vIt.hasNext() && iIt.hasNext()) {

                PriorityQueue<Pair<Float, Integer>> pq =
                        new PriorityQueue<>(
                                order == SortOrder.DESC
                                        ? (x, y) -> Float.compare(y.a(), x.a())
                                        : (x, y) -> Float.compare(x.a(), y.a()));

                Tensor ss = sIt.next(), vv = vIt.next(), ii = iIt.next();
                if (maxCDims > 0) {
                    System.arraycopy(ss.arr, ss.offset, buf, 0, buf.length);
                    for (int i = 0; i < buf.length; i++)
                        pq.offer(new Pair<>(buf[i], i));
                } else {
                    for (int i = 0; i < buf.length; i++)
                        pq.offer(new Pair<>(ss.get(i), i));
                }

                for (int i = 0; (k == -1 || i < k) && !pq.isEmpty(); i++) {
                    Pair<Float, Integer> v_n_i = pq.poll();
                    vv.set(v_n_i.a, i);
                    ii.set(v_n_i.b, i);
                }
            }
        }

        public Tensor contiguous() {
            return new Tensor(this, null, Op.CONTIGUOUS, new float[size()], 0, null, true, null, dims);
        }

        private void _contiguous(Tensor x) {
            int maxContDims = x.continousDims();
            TensorIt xIt = x.it(maxContDims), rIt = this.it(maxContDims);
            while (xIt.hasNext() && rIt.hasNext()) {
                Tensor xx = xIt.next(), rr = rIt.next();
                System.arraycopy(xx.arr, xx.offset, rr.arr, rr.offset, xx.size());
            }
        }

        public Tensor materialize() {
            eval();
            op1 = op2 = null;
            op = Op.NO_OP;
            return this;
        }

        public void copyFrom(Tensor src) {
            eval();
            src.eval();
            Tensor[] t = broadcast_dims(this, src, 0);
            int maxCont = Math.min(t[0].continousDims(), t[1].continousDims());
            TensorIt tgtIt = t[0].it(maxCont), srcIt = t[1].it(maxCont);
            while (tgtIt.hasNext() && srcIt.hasNext()) {
                Tensor ss = srcIt.next(), tt = tgtIt.next();
                System.arraycopy(ss.arr, ss.offset, tt.arr, tt.offset, tt.size());
            }
        }

        public Tensor add(Tensor other) {
            Tensor[] t = broadcast_dims(this, other, 0);
            Tensor a = t[0], b = t[1];
            return new Tensor(a, b, Op.MAT_ADD, new float[a.size()], 0, null, true, null, a.dims);
        }

        private void _add(Tensor x, Tensor y, int dim) {
            int contDims = Math.min(x.continousDims(), y.continousDims());
            TensorIt xIt = x.it(contDims), yIt = y.it(contDims), rIt = this.it(contDims);
            if (contDims > 0) {
                while (xIt.hasNext() && yIt.hasNext() && rIt.hasNext()) {
                    Tensor xx = xIt.next(), yy = yIt.next(), rr = rIt.next();
                    for (int i = xx.offset, j = yy.offset, k = rr.offset; i < xx.offset + xx.size(); i++, j++, k++) {
                        rr.arr[k] = xx.arr[i] + yy.arr[j];
                    }
                }
            } else {
                while (xIt.hasNext() && yIt.hasNext() && rIt.hasNext()) {
                    int ri = rIt.nextIdx(), xi = xIt.nextIdx(), yi = yIt.nextIdx();
                    arr[ri] = x.arr[xi] + y.arr[yi];
                }
            }
        }

        public Tensor maskedFill(Tensor mask, float val) {
            int[] dims = this.dims;
            Tensor[] t = broadcast_dims(this, mask, 2);
            Tensor x = t[0];
            mask = t[1];
            return new Tensor(x, mask, Op.MASKED_FILL, new float[x.size()], 0, val, true, null, dims);
        }

        private void _masked_fill(Tensor x, Tensor mask, float val) {
            TensorIt it = x.it(2), rIt = this.it(2);
            TensorIt mIt = mask.it(2);
            mask = mIt.next();
            while (it.hasNext() && rIt.hasNext()) {
                Tensor t = it.next(), rr = rIt.next();
                for (int i = 0; i < t.dims[0]; i++) {
                    for (int j = 0; j < t.dims[1]; j++) {
                        rr.set((mask.get(i, j) == 0) ? val : t.get(i, j), i, j);
                    }
                }
            }
        }

        public Tensor softMax(int dim) {
            return new Tensor(this, null, Op.SOFTMAX, new float[size()], 0, dim, true, null, dims);
        }

        public void _softmax(Tensor x, int dim) {

            Tensor r = this;
            for (int i = dim; i < x.dims.length - 1; i++) {
                x = x.transpose(i, i + 1);
                r = r.transpose(i, i + 1);
            }

            int ldim = x.dim(-1);
            float[] buf = new float[ldim];

            TensorIt it = x.it(1), rIt = r.it(1);
            while (it.hasNext() && rIt.hasNext()) {
                Tensor xx = it.next(), rr = rIt.next();

                float max = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < ldim; i++) {
                    buf[i] = xx.get(i);
                    max = Math.max(max, buf[i]);
                }

                float sum = 0;
                for (int i = 0; i < ldim; i++) {
                    buf[i] = (float) Math.exp(buf[i] - max);
                    sum += buf[i];
                }

                for (int i = 0; i < ldim; i++) {
                    rr.set(buf[i] / sum, i);
                }
            }
        }

        public Tensor argMax(int dim) {
            int[] ndims = Arrays.copyOf(dims, dims.length);
            ndims[dim] = 1;
            return new Tensor(this, null, Op.ARGMAX, null, 0, dim, true, null, ndims);
        }

        private void _argmax(Tensor x, int dim) {
            Tensor rdash = this;
            for (int i = dim; i < dims.length - 1; i++) {
                x = x.transpose(i, i + 1);
                rdash = rdash.transpose(i, i + 1);
            }
            TensorIt rIt = rdash.it(0);
            TensorIt it = x.it(1);
            while (it.hasNext() && rIt.hasNext()) {
                Tensor t = it.next(), rr = rIt.next();
                float max = Float.NEGATIVE_INFINITY;
                int idx = -1;
                for (int i = 0; i < t.size(); i++) {
                    if (max < t.get(i)) {
                        max = t.get(i);
                        idx = i;
                    }
                }
                rr.set(idx);
            }
        }

        public Tensor layerNorm(Tensor w, float epsilon) {
            return new Tensor(
                    this,
                    w,
                    Op.LAYER_NORM,
                    new float[size()],
                    0,
                    epsilon,
                    true,
                    null,
                    dims
            );
        }

        private void _layer_norm(Tensor x, Tensor w, float epsilon) {
            int ldim = x.dim(-1);
            float[] x_min_mu = new float[ldim];
            float[] buf = new float[ldim];

            TensorIt it = x.it(1), rIt = this.it(1);
            while (it.hasNext() && rIt.hasNext()) {
                Tensor xx = it.next(), rr = rIt.next();

                float[] arr = xx.continuous ? xx.arr : buf;
                int off = xx.continuous ? xx.offset : 0;

                if (!xx.continuous) {
                    for (int i = 0; i < ldim; i++) arr[i] = xx.get(i);
                }

                float mean = 0;
                for (int j = off; j < off + ldim; j++) mean += arr[j];
                mean /= ldim;

                float sig_sq = 0;
                for (int i = 0, j = off; i < ldim; j++, i++) {
                    x_min_mu[i] = arr[j] - mean;
                    sig_sq += x_min_mu[i] * x_min_mu[i];
                }
                sig_sq /= ldim;
                float denominator = (float) Math.sqrt(sig_sq + epsilon);

                for (int i = 0; i < ldim; i++) {
                    rr.set(w.arr[i] * (x_min_mu[i] / denominator), i);
                }
            }
        }

        public Tensor matmul(Tensor y) {
            Tensor x = this;
            if (x.dim(-1) != y.dim(-2)) {
                throw new IllegalArgumentException(String.format(
                        "matrix multiplication requires [M x K] @ [K x N], found (%d, %d) @ (%d, %d)",
                        x.dim(-2), x.dim(-1),
                        y.dim(-2), y.dim(-1)));
            }

            Tensor[] t = broadcast_dims(x, y, 2);
            x = t[0];
            y = t[1];

            int[] new_dims = Arrays.copyOf(x.dims, x.dims.length);
            new_dims[new_dims.length - 2] = x.dim(-2);
            new_dims[new_dims.length - 1] = y.dim(-1);
            int size = Arrays.stream(new_dims).reduce(1, (a, b) -> a * b);

            return new Tensor(x, y, Op.MAT_MUL, new float[size], 0, null, true, null, new_dims);
        }

        private void _matmul(Tensor x, Tensor y) {

            Tensor[] t = broadcast_dims(x, y, 2);
            TensorIt xIt = t[0].it(2);
            TensorIt yIt = t[1].it(2);
            TensorIt rIt = this.it(2);

            while (xIt.hasNext() && yIt.hasNext() && rIt.hasNext()) {
                Tensor xx = xIt.next();
                Tensor yy = yIt.next();
                Tensor rr = rIt.next();

                for (int i = 0; i < xx.dims[0]; i++) {
                    for (int k = 0; k < yy.dims[1]; k++) {
                        int rri = rr.index(i, k);
                        rr.arr[rri] = 0f;
                        for (int j = 0; j < xx.dims[1]; j++) {
                            int xxi = xx.index(i, j), yyi = yy.index(j, k);
                            rr.arr[rri] += xx.arr[xxi] * yy.arr[yyi];
                        }
                    }
                }
            }
        }

        public Tensor multinomial(Tensor indices) {
            int[] n_dims = Arrays.copyOf(dims, dims.length);
            n_dims[n_dims.length - 1] = 1;
            return new Tensor(this, indices, Op.MULTINOMIAL, new float[size() / dim(-1)], 0, null, true, null, n_dims);
        }

        public void _multinomial(Tensor values, Tensor indices) {
            Random R = new Random();
            TensorIt vIt = values.it(1), iIt = indices.it(1), rIt = this.it(0);
            while (vIt.hasNext() && iIt.hasNext() && rIt.hasNext()) {
                Tensor vv = vIt.next(), ii = iIt.next(), rr = rIt.next();
                float r = R.nextFloat(), cdf = 0;
                for (int i = 0; i < vv.size(); i++) {
                    cdf += vv.get(i);
                    if (r < cdf) {
                        rr.set(ii.get(i));
                        break;
                    }
                }
            }
        }

        private void _apply(Tensor x, Function<Float, Float> f) {
            int contDims = x.continousDims();
            TensorIt xIt = x.it(contDims), rIt = this.it(contDims);
            if (contDims == 0) {
                while (xIt.hasNext()) {
                    int ix = xIt.nextIdx();
                    arr[ix] = f.apply(x.arr[ix]);
                }
            } else {
                while (xIt.hasNext()) {
                    Tensor xx = xIt.next(), rr = rIt.next();
                    for (int i = xx.offset, j = rr.offset; i < xx.size(); i++, j++)
                        rr.arr[j] = f.apply(x.arr[i]);
                }
            }
        }

        public Tensor multiply(float scalar) {
            return new Tensor(this, null, Op.SCALAR_MUL, new float[size()], 0, scalar, true, null, dims);
        }

        float sqrt_2_by_pi = (float) Math.sqrt(2.0 / Math.PI);

        public Tensor gelu() {
            return new Tensor(this, null, Op.GELU, new float[size()], offset, null, true, null, dims);
        }

        private float _gelu(float x) {
            return (float) (0.5 * x * (1 + Math.tanh(sqrt_2_by_pi * (x + (0.044715 * (x * x * x))))));
        }

    }

    public static int[] $(int... rng) {
        if (rng.length > 2)
            throw new IllegalArgumentException("invalid range, eg - range(), range(2), range(1, 3)");
        return rng;
    }

    static class Meta {
        float attn_pdrop;
        int bos_token_id;
        float embd_pdrop;
        int eos_token_id;
        float initializer_range;
        float layer_norm_epsilon;
        int n_ctx;
        int n_embd;
        int n_head;
        int n_layer;
        int n_positions;
        float resid_pdrop;
        int vocab_size;
    }

    Meta meta;
    char[] b2u = new char[256];
    char[] u2b = new char[512];
    Map<String, Integer> vocab = new HashMap<>();
    Map<Integer, String> rvocab = new HashMap<>();
    Map<Long, Integer> mergesOrder = new HashMap<>();
    Map<String, Tensor> tensors = new HashMap<>();

    InferJ(String modelDir) throws IOException {
        tokenizerInit(modelDir + "/merges.txt", modelDir + "/vocab.json");
        loadTensors(modelDir + "/model.safetensors");
        loadMeta(modelDir + "/config.json");
    }


    int[] byte2unicode(byte[] bytes) {
        int[] ret = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++)
            ret[i] = b2u[bytes[i]];
        return ret;
    }

    String unicode2bytes(char[] bytes) {
        bytes = Arrays.copyOf(bytes, bytes.length);
        for (int i = 0; i < bytes.length; i++)
            bytes[i] = u2b[bytes[i]];
        return new String(bytes);
    }

    long merge(int a, int b) {
        return ((((long) a) << 32) | b);
    }

    static class MiniJson {
        private final Reader reader;
        private int ch;

        private MiniJson(Reader reader) {
            this.reader = reader;
        }

        @SuppressWarnings("unchecked")
        public static <T> T parse(Reader reader, Class<T> type) {
            MiniJson parser = new MiniJson(reader);
            parser.next();
            Object result = parser.parseValue();
            parser.skipWhitespace();
            if (parser.ch != -1) throw new RuntimeException("Unexpected trailing data");

            if (type == Map.class || result == null) {
                return (T) result;
            }

            if (result instanceof Map) {
                return convertToPojo((Map<String, Object>) result, type);
            }

            throw new IllegalArgumentException("JSON structure does not match target class " + type.getName());
        }

        public static <T> T parse(String json, Class<T> type) {
            return parse(new StringReader(json), type);
        }

        @SuppressWarnings("unchecked")
        private static <T> T convertToPojo(Map<String, Object> map, Class<T> type) {
            try {
                // Instantiate the target POJO using its default no-arg constructor
                T pojo = type.getDeclaredConstructor().newInstance();

                for (Map.Entry<String, Object> entry : map.entrySet()) {
                    try {
                        Field field = type.getDeclaredField(entry.getKey());
                        field.setAccessible(true);
                        Object value = entry.getValue();

                        if (value == null) {
                            field.set(pojo, null);
                            continue;
                        }

                        Class<?> fieldType = field.getType();

                        // Case 1: Nested POJO handling
                        if (value instanceof Map && !fieldType.isAssignableFrom(Map.class)) {
                            field.set(pojo, convertToPojo((Map<String, Object>) value, fieldType));
                        }
                        // Case 2: Nested Collection/List handling
                        else if (value instanceof List && List.class.isAssignableFrom(fieldType)) {
                            Type genericType = field.getGenericType();
                            if (genericType instanceof ParameterizedType pt) {
                                Class<?> itemType = (Class<?>) pt.getActualTypeArguments()[0];
                                List<Object> rawList = (List<Object>) value;
                                List<Object> typedList = new ArrayList<>();

                                for (Object item : rawList) {
                                    if (item instanceof Map && !itemType.isAssignableFrom(Map.class)) {
                                        typedList.add(convertToPojo((Map<String, Object>) item, itemType));
                                    } else {
                                        typedList.add(castNumeric(item, itemType));
                                    }
                                }
                                field.set(pojo, typedList);
                            } else {
                                field.set(pojo, value); // Fallback for raw lists
                            }
                        }
                        // Case 3: Standard Field or Type Casting
                        else {
                            field.set(pojo, castNumeric(value, fieldType));
                        }
                    } catch (NoSuchFieldException e) {
                        // Ignore unknown JSON keys that don't exist in your POJO
                    }
                }
                return pojo;
            } catch (Exception e) {
                throw new RuntimeException("Failed to map JSON data to POJO class " + type.getName(), e);
            }
        }

        // Safely coerces the custom Long/Double numbers into the expected POJO primitive types
        private static Object castNumeric(Object value, Class<?> targetType) {
            if (value instanceof Number num) {
                if (targetType == int.class || targetType == Integer.class) return num.intValue();
                if (targetType == long.class || targetType == Long.class) return num.longValue();
                if (targetType == double.class || targetType == Double.class) return num.doubleValue();
                if (targetType == float.class || targetType == Float.class) return num.floatValue();
            }
            return value;
        }

        // --- Internal Parsing Logic ---

        private void next() {
            try {
                ch = reader.read();
            } catch (IOException e) {
                throw new RuntimeException("Error reading JSON input", e);
            }
        }

        private void skipWhitespace() {
            while (Character.isWhitespace(ch)) next();
        }

        private boolean eat(int c) {
            skipWhitespace();
            if (ch == c) {
                next();
                return true;
            }
            return false;
        }

        private Object parseValue() {
            if (eat('{')) return parseObject();
            if (eat('[')) return parseArray();
            if (eat('"')) return parseString();
            return parseLiteral();
        }

        private Map<String, Object> parseObject() {
            Map<String, Object> map = new LinkedHashMap<>();
            if (!eat('}')) {
                do {
                    if (!eat('"')) throw new RuntimeException("Expected string key");
                    String key = parseString();
                    if (!eat(':')) throw new RuntimeException("Expected ':' separator");
                    map.put(key, parseValue());
                } while (eat(','));
                if (!eat('}')) throw new RuntimeException("Expected '}' closing object");
            }
            return map;
        }

        private List<Object> parseArray() {
            List<Object> list = new ArrayList<>();
            if (!eat(']')) {
                do {
                    list.add(parseValue());
                } while (eat(','));
                if (!eat(']')) throw new RuntimeException("Expected ']' closing array");
            }
            return list;
        }

        private String parseString() {
            StringBuilder sb = new StringBuilder();
            while (ch != '"' && ch != -1) {
                if (ch == '\\') {
                    next();
                    switch (ch) {
                        case 'b' -> sb.append('\b');
                        case 'f' -> sb.append('\f');
                        case 'n' -> sb.append('\n');
                        case 'r' -> sb.append('\r');
                        case 't' -> sb.append('\t');
                        case '"', '\\', '/' -> sb.append((char) ch);
                        case 'u' -> {
                            int code = 0;
                            for (int i = 0; i < 4; i++) {
                                next();
                                int digit = Character.digit(ch, 16);
                                if (digit == -1) throw new RuntimeException("Invalid hex character in \\u escape");
                                code = (code << 4) | digit;
                            }
                            sb.append((char) code);
                        }
                        default -> throw new RuntimeException("Unsupported escape character: \\" + (char) ch);
                    }
                } else {
                    sb.append((char) ch);
                }
                next();
            }
            if (ch == -1) throw new RuntimeException("Unterminated string");
            next();
            return sb.toString();
        }

        private Object parseLiteral() {
            StringBuilder sb = new StringBuilder();
            skipWhitespace();
            while (Character.isLetterOrDigit(ch) || ch == '.' || ch == '-' || ch == '+') {
                sb.append((char) ch);
                next();
            }
            String s = sb.toString();
            if (s.isEmpty()) throw new RuntimeException("Unexpected character: " + (char) ch);
            if (s.equals("true")) return Boolean.TRUE;
            if (s.equals("false")) return Boolean.FALSE;
            if (s.equals("null")) return null;

            try {
                return Long.parseLong(s);
            } catch (NumberFormatException e1) {
                try {
                    return Double.parseDouble(s);
                } catch (NumberFormatException e2) {
                    return s;
                }
            }
        }
    }

    void tokenizerInit(String mergesFile, String vocabFile) throws IOException {

        // step 1 : reverse the byte to unicode
        boolean[] used = new boolean[256];
        for (int i = '!'; i <= '~'; i++)
            used[i] = true;
        for (int i = '¡'; i <= '¬'; i++)
            used[i] = true;
        for (int i = '®'; i <= 'ÿ'; i++)
            used[i] = true;

        for (int i = 0, j = 0; i < 256; i++) {
            b2u[i] = (char) (used[i] ? i : (256 + j++));
            u2b[b2u[i]] = (char) i;
        }

        // read vocab.json
        MiniJson.parse(new FileReader(vocabFile), Map.class).forEach((k, v) -> {
            vocab.put((String) k, ((Long) v).intValue());
            rvocab.put(((Long) v).intValue(), (String) k);
        });

        // read the merges.txt and vocab.txt
        BufferedReader br = new BufferedReader(new FileReader(mergesFile));
        int i = 0;
        for (String line = br.readLine(); line != null; line = br.readLine()) {
            if (line.startsWith("#"))
                continue;
            String[] splits = line.split(" ");
            mergesOrder.put(merge(vocab.get(splits[0]), vocab.get(splits[1])), i++);
        }
    }

    // simple, stupid tokenizer
    float[] tokenize(String str) {
        int[] tokens = byte2unicode(str.getBytes(StandardCharsets.UTF_8));

        for (int i = 0; i < tokens.length; i++)
            tokens[i] = vocab.get("" + (char) tokens[i]);

        int seqLen = tokens.length;
        while (true) {
            int curBest = Integer.MAX_VALUE;
            int best1 = 0, best2 = 0;
            for (int i = 1; i < seqLen; i++) {
                Integer mergeOrder = mergesOrder.get(merge(tokens[i - 1], tokens[i]));
                if (mergeOrder != null && mergeOrder < curBest) {
                    curBest = mergeOrder;
                    best1 = tokens[i - 1];
                    best2 = tokens[i];
                }
            }
            if (curBest == Integer.MAX_VALUE)
                break;
            int wi = 0;
            for (int i = 0; i < seqLen; i++, wi++) {
                if ((i + 1 < seqLen) && best1 == tokens[i] && best2 == tokens[i + 1]) {
                    tokens[wi] = vocab.get(rvocab.get(best1) + rvocab.get(best2));
                    i++;
                } else {
                    tokens[wi] = tokens[i];
                }
            }
            seqLen = wi;
        }

        float[] ret = new float[seqLen];
        for (int i = 0; i < seqLen; i++) ret[i] = tokens[i];
        return ret;
    }

    void loadTensors(String modelFile) throws IOException {

        try (RandomAccessFile rf = new RandomAccessFile(modelFile, "r");
             FileChannel fc = rf.getChannel()) {
            MappedByteBuffer map = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
            ByteBuffer buf = map.order(ByteOrder.LITTLE_ENDIAN);
            long headerLen = map.asLongBuffer().get();
            byte hbuf[] = new byte[(int) headerLen];
            map.get(8, hbuf);
            Map metadata = MiniJson.parse(new String(hbuf), Map.class);

            Map<String, Integer> dTypeSz = new HashMap<>() {
                {
                    put("FP32", 4);
                }
            };

            int dataBase = (int) (8 + headerLen);

            metadata.forEach((k, v) -> {
                String key = (String) k;
                if (key.equals("__metadata__"))
                    return;

                String dType = (String) ((Map) v).get("dtype");
                ArrayList<Long> shape = (ArrayList<Long>) ((Map) v).get("shape");
                ArrayList<Long> range = (ArrayList<Long>) ((Map) v).get("data_offsets");

                int start = dataBase + range.get(0).intValue();
                int len = (int) (range.get(1) - range.get(0));
                FloatBuffer fb = map.slice(start, len).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                float[] fbuf = new float[fb.remaining()];
                fb.get(fbuf);

                int[] shapes = new int[shape.size()];
                for (int i = 0; i < shape.size(); i++)
                    shapes[i] = shape.get(i).intValue();
                tensors.put(key, Tensor.of(fbuf, shapes).withName(key));
            });

        }

    }

    void loadMeta(String metaFile) throws FileNotFoundException {
        meta = MiniJson.parse(new FileReader(metaFile), Meta.class);
    }

    Tensor TG(String name) {
        return tensors.get(name);
    }

    Tensor makeGpt2Graph() {
        return null;
    }

    interface TokenReporter {
        void onToken(int batch, int token, float avgTokPerS);
    }

    // for now batch = 1
    void infer(String prompt, int maxTokens, float temp, float topP, int topK, TokenReporter reporter) {

        int MAX_BATCHES = 4;
        var mask = Tensor.ltm(meta.n_ctx);
        var allEmbeddings = TG("wte.weight");
        var inputs = Tensor.of(MAX_BATCHES, meta.n_ctx, meta.n_embd);

        float[] tokens = tokenize(prompt);
        int B = 1, T = tokens.length, C = meta.n_embd;
        var indices = Tensor.of(tokens, B, T).withName("input_tokens");

        long time = 0;

        for (int processedT = 0, it = 0; T < maxTokens + tokens.length; T++, it++) {

            long start = System.currentTimeMillis();
            Tensor add = allEmbeddings.select(indices).add(TG("wpe.weight").slice($(processedT, T), $()));
            inputs.slice($(0, B), $(processedT, T), $()).copyFrom(add);
            var cur = inputs.slice($(0, B), $(0, T), $());
            processedT = T;

            for (int i = 0; i < meta.n_layer; i++) {

                var ln1_wb = cur.layerNorm(TG("h." + i + ".ln_1.weight"), meta.layer_norm_epsilon);
                var ln1 = ln1_wb.add(TG("h." + i + ".ln_1.bias").view(1, -1));

                var c_attn = ln1.matmul(TG("h." + i + ".attn.c_attn.weight")).add(TG("h." + i + ".attn.c_attn.bias"));

                Tensor[] qkv = c_attn.split(3, 2);
                Tensor q = qkv[0], k = qkv[1], v = qkv[2];

                q = q.view(B, T, meta.n_head, C / meta.n_head).transpose(1, 2);
                k = k.view(B, T, meta.n_head, C / meta.n_head).transpose(1, 2);
                v = v.view(B, T, meta.n_head, C / meta.n_head).transpose(1, 2);

                var qkt = q.matmul(k.transpose(2, 3));
                qkt = qkt.multiply((float) (1f / Math.sqrt((double) C / meta.n_head)));
                qkt = qkt.maskedFill(mask, Float.NEGATIVE_INFINITY);

                var oWeights = qkt.softMax(3);

                var x = oWeights.matmul(v).transpose(1, 2).contiguous().view(B, T, C);

                var proj = x.matmul(TG("h." + i + ".attn.c_proj.weight"))
                        .add(TG("h." + i + ".attn.c_proj.bias"));

                cur = cur.add(proj);

                var ln2 = cur.layerNorm(TG("h." + i + ".ln_2.weight"), meta.layer_norm_epsilon)
                        .add(TG("h." + i + ".ln_2.bias").view(1, -1));

                var cfc = ln2.matmul(TG("h." + i + ".mlp.c_fc.weight"))
                        .add(TG("h." + i + ".mlp.c_fc.bias"));

                var gelu = cfc.gelu();

                var cproj = gelu.matmul(TG("h." + i + ".mlp.c_proj.weight"))
                        .add(TG("h." + i + ".mlp.c_proj.bias"));

                cur = cur.add(cproj);
            }

            var lnf = cur.slice($(), $(T - 1), $()).layerNorm(TG("ln_f.weight"), meta.layer_norm_epsilon)
                    .add(TG("ln_f.bias").view(1, -1));

            var top_k_i = lnf
                    .matmul(allEmbeddings.transpose(0, 1))
                    .view(B, -1)
                    .multiply(1f / temp)        // temperature
                    .topK(1, topK);

            var top_k = top_k_i.sub(0);
            var top_i = top_k_i.sub(1);

            indices = top_k.softMax(1).multinomial(top_i).eval().materialize();

            time += System.currentTimeMillis() - start;
            for (int i = 0; i < B; i++) {
                int next = (int) indices.get(0, 0);
                reporter.onToken(i, next, (float) it * 1000 / time);
            }
        }
    }

    static String findArg(String[] args, String name, String def) {
        for (int i = 0; i < args.length; i += 2)
            if (args[i].substring(2).equals(name)) return args[i + 1];
        if (def != null) return def;
        throw new IllegalArgumentException(name + " not found");
    }

    public static void main(String[] args) throws IOException {

        String modelDir = findArg(args, "model-dir", ".");
        String prompt = findArg(args, "prompt", "once upon a time there");
        int maxTokens = Integer.parseInt(findArg(args, "max-tokens", "100"));
        float topP = Float.parseFloat(findArg(args, "top-p", "1.0"));
        int topK = Integer.parseInt(findArg(args, "top-k", "5"));
        float temp = Float.parseFloat(findArg(args, "temp", "1.3"));

        final StringBuilder sb = new StringBuilder(prompt);
        InferJ gpt2 = new InferJ(modelDir);
        gpt2.infer(prompt, maxTokens, temp, topP, topK, (b, token, avgTokPerS) -> {
            sb.append(gpt2.unicode2bytes(gpt2.rvocab.get(token).toCharArray()));
            System.out.printf("\r[%f tok/s] %s", avgTokPerS, sb);
        });
    }
}
