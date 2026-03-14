/*
 * tafra/_accel.c — Minimal C extension for hot-path acceleration.
 *
 * Provides single-pass grouped aggregation and hash-based join index
 * construction, eliminating multiple numpy passes and temporary arrays.
 *
 * All functions accept and return numpy arrays. Falls back gracefully
 * if this module fails to compile — pure-Python paths remain available.
 *
 * All input arrays are coerced to contiguous layout via PyArray_FROM_OTF.
 * Callers guarantee n_groups > 0 and all label values in [0, n_groups).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>


/* Helper: coerce to contiguous array of given type. Caller must Py_DECREF. */
static PyArrayObject *
as_contig(PyObject *obj, int typenum)
{
    return (PyArrayObject *)PyArray_FROM_OTF(
        obj, typenum, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}


/* ================================================================
 * GroupBy aggregations: single pass over (labels, data)
 *
 * All groupby functions expect:
 *   labels: int64 array of group indices in [0, n_groups)
 *   data:   float64 array of values (same length as labels)
 *   n_groups: number of groups (Py_ssize_t)
 * ================================================================ */

static PyObject *
accel_groupby_sum(PyObject *self, PyObject *args)
{
    PyObject *labels_obj, *data_obj;
    Py_ssize_t n_groups;

    if (!PyArg_ParseTuple(args, "OOn", &labels_obj, &data_obj, &n_groups))
        return NULL;

    PyArrayObject *labels_arr = as_contig(labels_obj, NPY_INT64);
    PyArrayObject *data_arr = as_contig(data_obj, NPY_FLOAT64);
    if (!labels_arr || !data_arr) {
        Py_XDECREF(labels_arr); Py_XDECREF(data_arr);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(labels_arr);
    npy_intp dims[1] = {(npy_intp)n_groups};
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    if (!out) { Py_DECREF(labels_arr); Py_DECREF(data_arr); return NULL; }

    const npy_int64 *labels = (const npy_int64 *)PyArray_DATA(labels_arr);
    const npy_float64 *data = (const npy_float64 *)PyArray_DATA(data_arr);
    npy_float64 *result = (npy_float64 *)PyArray_DATA(out);

    for (npy_intp i = 0; i < n; i++)
        result[labels[i]] += data[i];

    Py_DECREF(labels_arr);
    Py_DECREF(data_arr);
    return (PyObject *)out;
}


static PyObject *
accel_groupby_count(PyObject *self, PyObject *args)
{
    PyObject *labels_obj;
    Py_ssize_t n_groups;

    if (!PyArg_ParseTuple(args, "On", &labels_obj, &n_groups))
        return NULL;

    PyArrayObject *labels_arr = as_contig(labels_obj, NPY_INT64);
    if (!labels_arr) return NULL;

    npy_intp n = PyArray_SIZE(labels_arr);
    npy_intp dims[1] = {(npy_intp)n_groups};
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    if (!out) { Py_DECREF(labels_arr); return NULL; }

    const npy_int64 *labels = (const npy_int64 *)PyArray_DATA(labels_arr);
    npy_int64 *result = (npy_int64 *)PyArray_DATA(out);

    for (npy_intp i = 0; i < n; i++)
        result[labels[i]]++;

    Py_DECREF(labels_arr);
    return (PyObject *)out;
}


static PyObject *
accel_groupby_mean(PyObject *self, PyObject *args)
{
    PyObject *labels_obj, *data_obj;
    Py_ssize_t n_groups;

    if (!PyArg_ParseTuple(args, "OOn", &labels_obj, &data_obj, &n_groups))
        return NULL;

    PyArrayObject *labels_arr = as_contig(labels_obj, NPY_INT64);
    PyArrayObject *data_arr = as_contig(data_obj, NPY_FLOAT64);
    if (!labels_arr || !data_arr) {
        Py_XDECREF(labels_arr); Py_XDECREF(data_arr);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(labels_arr);
    npy_intp dims[1] = {(npy_intp)n_groups};
    PyArrayObject *sums = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    PyArrayObject *counts = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    if (!sums || !counts) {
        Py_XDECREF(sums); Py_XDECREF(counts);
        Py_DECREF(labels_arr); Py_DECREF(data_arr);
        return NULL;
    }

    const npy_int64 *labels = (const npy_int64 *)PyArray_DATA(labels_arr);
    const npy_float64 *data = (const npy_float64 *)PyArray_DATA(data_arr);
    npy_float64 *s = (npy_float64 *)PyArray_DATA(sums);
    npy_int64 *c = (npy_int64 *)PyArray_DATA(counts);

    for (npy_intp i = 0; i < n; i++) {
        npy_int64 g = labels[i];
        s[g] += data[i];
        c[g]++;
    }

    for (Py_ssize_t g = 0; g < n_groups; g++) {
        if (c[g] > 0)
            s[g] /= (npy_float64)c[g];
        else
            s[g] = NPY_NAN;
    }

    Py_DECREF(counts);
    Py_DECREF(labels_arr);
    Py_DECREF(data_arr);
    return (PyObject *)sums;
}


/* Welford's online algorithm for numerically stable variance */
static PyObject *
accel_groupby_var(PyObject *self, PyObject *args)
{
    PyObject *labels_obj, *data_obj;
    Py_ssize_t n_groups;

    if (!PyArg_ParseTuple(args, "OOn", &labels_obj, &data_obj, &n_groups))
        return NULL;

    PyArrayObject *labels_arr = as_contig(labels_obj, NPY_INT64);
    PyArrayObject *data_arr = as_contig(data_obj, NPY_FLOAT64);
    if (!labels_arr || !data_arr) {
        Py_XDECREF(labels_arr); Py_XDECREF(data_arr);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(labels_arr);
    npy_intp dims[1] = {(npy_intp)n_groups};

    /* Welford accumulators: count, mean, M2 */
    PyArrayObject *counts = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    PyArrayObject *means = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    PyArrayObject *m2s = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    if (!counts || !means || !m2s) {
        Py_XDECREF(counts); Py_XDECREF(means); Py_XDECREF(m2s);
        Py_DECREF(labels_arr); Py_DECREF(data_arr);
        return NULL;
    }

    const npy_int64 *labels = (const npy_int64 *)PyArray_DATA(labels_arr);
    const npy_float64 *data = (const npy_float64 *)PyArray_DATA(data_arr);
    npy_int64 *cnt = (npy_int64 *)PyArray_DATA(counts);
    npy_float64 *mean = (npy_float64 *)PyArray_DATA(means);
    npy_float64 *m2 = (npy_float64 *)PyArray_DATA(m2s);

    for (npy_intp i = 0; i < n; i++) {
        npy_int64 g = labels[i];
        npy_float64 x = data[i];
        cnt[g]++;
        npy_float64 delta = x - mean[g];
        mean[g] += delta / (npy_float64)cnt[g];
        npy_float64 delta2 = x - mean[g];
        m2[g] += delta * delta2;
    }

    /* Convert M2 to population variance, reuse means array for output */
    for (Py_ssize_t g = 0; g < n_groups; g++) {
        if (cnt[g] > 0)
            mean[g] = m2[g] / (npy_float64)cnt[g];
        else
            mean[g] = NPY_NAN;
    }

    Py_DECREF(counts);
    Py_DECREF(m2s);
    Py_DECREF(labels_arr);
    Py_DECREF(data_arr);
    return (PyObject *)means;
}


static PyObject *
accel_groupby_min(PyObject *self, PyObject *args)
{
    PyObject *labels_obj, *data_obj;
    Py_ssize_t n_groups;

    if (!PyArg_ParseTuple(args, "OOn", &labels_obj, &data_obj, &n_groups))
        return NULL;

    PyArrayObject *labels_arr = as_contig(labels_obj, NPY_INT64);
    PyArrayObject *data_arr = as_contig(data_obj, NPY_FLOAT64);
    if (!labels_arr || !data_arr) {
        Py_XDECREF(labels_arr); Py_XDECREF(data_arr);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(labels_arr);
    npy_intp dims[1] = {(npy_intp)n_groups};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!out) { Py_DECREF(labels_arr); Py_DECREF(data_arr); return NULL; }

    npy_float64 *result = (npy_float64 *)PyArray_DATA(out);
    for (Py_ssize_t g = 0; g < n_groups; g++)
        result[g] = NPY_INFINITY;

    const npy_int64 *labels = (const npy_int64 *)PyArray_DATA(labels_arr);
    const npy_float64 *data = (const npy_float64 *)PyArray_DATA(data_arr);

    for (npy_intp i = 0; i < n; i++) {
        npy_int64 g = labels[i];
        if (data[i] < result[g]) result[g] = data[i];
    }

    Py_DECREF(labels_arr);
    Py_DECREF(data_arr);
    return (PyObject *)out;
}


static PyObject *
accel_groupby_max(PyObject *self, PyObject *args)
{
    PyObject *labels_obj, *data_obj;
    Py_ssize_t n_groups;

    if (!PyArg_ParseTuple(args, "OOn", &labels_obj, &data_obj, &n_groups))
        return NULL;

    PyArrayObject *labels_arr = as_contig(labels_obj, NPY_INT64);
    PyArrayObject *data_arr = as_contig(data_obj, NPY_FLOAT64);
    if (!labels_arr || !data_arr) {
        Py_XDECREF(labels_arr); Py_XDECREF(data_arr);
        return NULL;
    }

    npy_intp n = PyArray_SIZE(labels_arr);
    npy_intp dims[1] = {(npy_intp)n_groups};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!out) { Py_DECREF(labels_arr); Py_DECREF(data_arr); return NULL; }

    npy_float64 *result = (npy_float64 *)PyArray_DATA(out);
    for (Py_ssize_t g = 0; g < n_groups; g++)
        result[g] = -NPY_INFINITY;

    const npy_int64 *labels = (const npy_int64 *)PyArray_DATA(labels_arr);
    const npy_float64 *data = (const npy_float64 *)PyArray_DATA(data_arr);

    for (npy_intp i = 0; i < n; i++) {
        npy_int64 g = labels[i];
        if (data[i] > result[g]) result[g] = data[i];
    }

    Py_DECREF(labels_arr);
    Py_DECREF(data_arr);
    return (PyObject *)out;
}


/* ================================================================
 * Hash join: build (left_indices, right_indices) for equi-join
 *
 * Uses a simple open-addressing hash table on the right key,
 * then probes with each left key.
 *
 * Input arrays are coerced to contiguous int64.
 * ================================================================ */

typedef struct {
    npy_int64 key;
    npy_intp first;  /* index into chain array */
    npy_intp count;
} HashEntry;

typedef struct {
    npy_intp row;
    npy_intp next;  /* -1 = end */
} ChainNode;

#define HASH_KEY(k, sz) ((npy_intp)(((npy_uint64)(k) * 0x9E3779B97F4A7C15ULL) >> 1) % (sz))


/* Build hash table on right-side keys. Returns 0 on success, -1 on error. */
static int
build_hash_table(const npy_int64 *right, npy_intp right_n,
                 HashEntry **out_table, npy_intp *out_table_size,
                 ChainNode **out_chain)
{
    npy_intp table_size = right_n * 2;
    if (table_size < 16) table_size = 16;

    HashEntry *table = (HashEntry *)calloc(table_size, sizeof(HashEntry));
    ChainNode *chain = (ChainNode *)malloc(right_n * sizeof(ChainNode));
    if (!table || !chain) {
        free(table); free(chain);
        PyErr_NoMemory();
        return -1;
    }

    for (npy_intp i = 0; i < table_size; i++) {
        table[i].count = 0;
        table[i].first = -1;
    }

    for (npy_intp i = 0; i < right_n; i++) {
        npy_int64 k = right[i];
        npy_intp h = HASH_KEY(k, table_size);
        while (table[h].count > 0 && table[h].key != k)
            h = (h + 1) % table_size;
        chain[i].row = i;
        chain[i].next = table[h].first;
        table[h].key = k;
        table[h].first = i;
        table[h].count++;
    }

    *out_table = table;
    *out_table_size = table_size;
    *out_chain = chain;
    return 0;
}


static PyObject *
accel_inner_join(PyObject *self, PyObject *args)
{
    PyObject *left_obj, *right_obj;
    if (!PyArg_ParseTuple(args, "OO", &left_obj, &right_obj))
        return NULL;

    PyArrayObject *left_arr = as_contig(left_obj, NPY_INT64);
    PyArrayObject *right_arr = as_contig(right_obj, NPY_INT64);
    if (!left_arr || !right_arr) {
        Py_XDECREF(left_arr); Py_XDECREF(right_arr);
        return NULL;
    }

    npy_intp left_n = PyArray_SIZE(left_arr);
    npy_intp right_n = PyArray_SIZE(right_arr);
    const npy_int64 *left = (const npy_int64 *)PyArray_DATA(left_arr);
    const npy_int64 *right = (const npy_int64 *)PyArray_DATA(right_arr);

    HashEntry *table; npy_intp table_size; ChainNode *chain;
    if (build_hash_table(right, right_n, &table, &table_size, &chain) < 0) {
        Py_DECREF(left_arr); Py_DECREF(right_arr);
        return NULL;
    }

    /* Count total output rows */
    npy_intp total = 0;
    for (npy_intp i = 0; i < left_n; i++) {
        npy_intp h = HASH_KEY(left[i], table_size);
        while (table[h].count > 0 && table[h].key != left[i])
            h = (h + 1) % table_size;
        if (table[h].count > 0 && table[h].key == left[i])
            total += table[h].count;
    }

    npy_intp dims[1] = {total};
    PyArrayObject *li_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    PyArrayObject *ri_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    if (!li_arr || !ri_arr) {
        Py_XDECREF(li_arr); Py_XDECREF(ri_arr);
        free(table); free(chain);
        Py_DECREF(left_arr); Py_DECREF(right_arr);
        return NULL;
    }

    npy_intp *li = (npy_intp *)PyArray_DATA(li_arr);
    npy_intp *ri = (npy_intp *)PyArray_DATA(ri_arr);
    npy_intp pos = 0;

    for (npy_intp i = 0; i < left_n; i++) {
        npy_intp h = HASH_KEY(left[i], table_size);
        while (table[h].count > 0 && table[h].key != left[i])
            h = (h + 1) % table_size;
        if (table[h].count > 0 && table[h].key == left[i]) {
            npy_intp ci = table[h].first;
            while (ci >= 0) {
                li[pos] = i;
                ri[pos] = chain[ci].row;
                pos++;
                ci = chain[ci].next;
            }
        }
    }

    free(table); free(chain);
    Py_DECREF(left_arr); Py_DECREF(right_arr);

    PyObject *result = PyTuple_Pack(2, (PyObject *)li_arr, (PyObject *)ri_arr);
    Py_DECREF(li_arr); Py_DECREF(ri_arr);
    return result;
}


static PyObject *
accel_left_join(PyObject *self, PyObject *args)
{
    PyObject *left_obj, *right_obj;
    if (!PyArg_ParseTuple(args, "OO", &left_obj, &right_obj))
        return NULL;

    PyArrayObject *left_arr = as_contig(left_obj, NPY_INT64);
    PyArrayObject *right_arr = as_contig(right_obj, NPY_INT64);
    if (!left_arr || !right_arr) {
        Py_XDECREF(left_arr); Py_XDECREF(right_arr);
        return NULL;
    }

    npy_intp left_n = PyArray_SIZE(left_arr);
    npy_intp right_n = PyArray_SIZE(right_arr);
    const npy_int64 *left = (const npy_int64 *)PyArray_DATA(left_arr);
    const npy_int64 *right = (const npy_int64 *)PyArray_DATA(right_arr);

    HashEntry *table; npy_intp table_size; ChainNode *chain;
    if (build_hash_table(right, right_n, &table, &table_size, &chain) < 0) {
        Py_DECREF(left_arr); Py_DECREF(right_arr);
        return NULL;
    }

    npy_intp total = 0;
    int has_null = 0;
    for (npy_intp i = 0; i < left_n; i++) {
        npy_intp h = HASH_KEY(left[i], table_size);
        while (table[h].count > 0 && table[h].key != left[i])
            h = (h + 1) % table_size;
        if (table[h].count > 0 && table[h].key == left[i])
            total += table[h].count;
        else {
            total++;
            has_null = 1;
        }
    }

    npy_intp dims[1] = {total};
    PyArrayObject *li_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    PyArrayObject *ri_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    if (!li_arr || !ri_arr) {
        Py_XDECREF(li_arr); Py_XDECREF(ri_arr);
        free(table); free(chain);
        Py_DECREF(left_arr); Py_DECREF(right_arr);
        return NULL;
    }

    npy_intp *li = (npy_intp *)PyArray_DATA(li_arr);
    npy_intp *ri = (npy_intp *)PyArray_DATA(ri_arr);
    npy_intp pos = 0;

    for (npy_intp i = 0; i < left_n; i++) {
        npy_intp h = HASH_KEY(left[i], table_size);
        while (table[h].count > 0 && table[h].key != left[i])
            h = (h + 1) % table_size;
        if (table[h].count > 0 && table[h].key == left[i]) {
            npy_intp ci = table[h].first;
            while (ci >= 0) {
                li[pos] = i;
                ri[pos] = chain[ci].row;
                pos++;
                ci = chain[ci].next;
            }
        } else {
            li[pos] = i;
            ri[pos] = -1;
            pos++;
        }
    }

    free(table); free(chain);
    Py_DECREF(left_arr); Py_DECREF(right_arr);

    PyObject *result = PyTuple_Pack(3,
        (PyObject *)li_arr, (PyObject *)ri_arr,
        has_null ? Py_True : Py_False);
    Py_DECREF(li_arr); Py_DECREF(ri_arr);
    return result;
}


/* ================================================================
 * Module definition
 * ================================================================ */

static PyMethodDef AccelMethods[] = {
    {"groupby_sum", accel_groupby_sum, METH_VARARGS,
     "Single-pass grouped sum: groupby_sum(labels, data, n_groups)"},
    {"groupby_count", accel_groupby_count, METH_VARARGS,
     "Single-pass grouped count: groupby_count(labels, n_groups)"},
    {"groupby_mean", accel_groupby_mean, METH_VARARGS,
     "Single-pass grouped mean: groupby_mean(labels, data, n_groups)"},
    {"groupby_var", accel_groupby_var, METH_VARARGS,
     "Single-pass grouped variance (Welford): groupby_var(labels, data, n_groups)"},
    {"groupby_min", accel_groupby_min, METH_VARARGS,
     "Single-pass grouped min: groupby_min(labels, data, n_groups)"},
    {"groupby_max", accel_groupby_max, METH_VARARGS,
     "Single-pass grouped max: groupby_max(labels, data, n_groups)"},
    {"inner_join", accel_inner_join, METH_VARARGS,
     "Hash inner join: inner_join(left_key, right_key) -> (li, ri)"},
    {"left_join", accel_left_join, METH_VARARGS,
     "Hash left join: left_join(left_key, right_key) -> (li, ri, has_null)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef accelmodule = {
    PyModuleDef_HEAD_INIT,
    "_accel",
    "C acceleration for tafra hot paths",
    -1,
    AccelMethods
};

PyMODINIT_FUNC
PyInit__accel(void)
{
    import_array();
    return PyModule_Create(&accelmodule);
}
