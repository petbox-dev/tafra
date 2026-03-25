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

/* Golden ratio constant for multiplicative hashing (Knuth).
 * phi = (sqrt(5)-1)/2 * 2^64 ≈ 0x9E3779B97F4A7C15 */
#define GOLDEN_HASH 0x9E3779B97F4A7C15ULL
#define HASH_KEY(k, sz) ((npy_intp)(((npy_uint64)(k) * GOLDEN_HASH) >> 1) % (sz))


/* Helper: coerce to contiguous array of given type. Caller must Py_DECREF. */
static PyArrayObject *
as_contig(PyObject *obj, int typenum)
{
    return (PyArrayObject *)PyArray_FROM_OTF(
        obj, typenum, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}


/* ================================================================
 * Generic hash table probing
 *
 * Callbacks return: 1 = yes, 0 = no, -1 = error (for fallible comparisons).
 * The probe function uses void* so a single implementation covers all
 * entry types; per-type static callbacks cast back to the concrete struct.
 * ================================================================ */

typedef int (*ht_occupied_fn)(const void *entry);
typedef int (*ht_equal_fn)(const void *entry, npy_int64 hash_key, void *ctx);

/*
 * Linear-probe to find either the slot matching hash_key or the first
 * empty slot.  On return:
 *   *found = 1 if an occupied matching slot was found, 0 if empty slot
 *   *err   = 1 if keys_equal returned -1 (comparison error), else 0
 */
static inline npy_intp
ht_probe(const void *table, npy_intp entry_size, npy_intp table_size,
         npy_int64 hash_key, void *ctx, int *found, int *err,
         ht_occupied_fn is_occupied, ht_equal_fn keys_equal)
{
    npy_intp slot = HASH_KEY(hash_key, table_size);
    const char *base = (const char *)table;
    *err = 0;
    for (;;) {
        const void *entry = base + slot * entry_size;
        if (!is_occupied(entry)) {
            *found = 0;
            return slot;
        }
        int eq = keys_equal(entry, hash_key, ctx);
        if (eq < 0) { *found = 0; *err = 1; return slot; }
        if (eq) { *found = 1; return slot; }
        slot = (slot + 1) % table_size;
    }
}

/* Convenience: alloc a zero-initialized hash table with 2x headroom. */
static inline void *
ht_alloc(npy_intp n_items, npy_intp entry_size, npy_intp *out_table_size)
{
    npy_intp sz = n_items * 2;
    if (sz < 16) sz = 16;
    *out_table_size = sz;
    return calloc(sz, entry_size);
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
 * Composite key: positional encoding of multiple int64 columns
 *
 * composite_key(arrays, cardinalities) -> int64 array
 *   arrays:        tuple of int64 arrays (encoded columns)
 *   cardinalities: tuple of ints (max value + 1 for each column)
 *
 * Computes: key[i] = col0[i]*card1*card2*... + col1[i]*card2*... + ...
 * Single pass, no temporaries.
 * ================================================================ */

static PyObject *
accel_composite_key(PyObject *self, PyObject *args)
{
    PyObject *arrays_obj, *cards_obj;
    if (!PyArg_ParseTuple(args, "OO", &arrays_obj, &cards_obj))
        return NULL;

    if (!PyTuple_Check(arrays_obj) || !PyTuple_Check(cards_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected two tuples");
        return NULL;
    }

    Py_ssize_t n_cols = PyTuple_GET_SIZE(arrays_obj);
    if (n_cols != PyTuple_GET_SIZE(cards_obj) || n_cols < 1) {
        PyErr_SetString(PyExc_ValueError, "arrays and cardinalities must have same positive length");
        return NULL;
    }

    /* Coerce all arrays to contiguous int64 */
    PyArrayObject **col_arrs = (PyArrayObject **)malloc(n_cols * sizeof(PyArrayObject *));
    if (!col_arrs) return PyErr_NoMemory();

    npy_intp n_rows = 0;
    for (Py_ssize_t c = 0; c < n_cols; c++) {
        col_arrs[c] = as_contig(PyTuple_GET_ITEM(arrays_obj, c), NPY_INT64);
        if (!col_arrs[c]) {
            for (Py_ssize_t j = 0; j < c; j++) Py_DECREF(col_arrs[j]);
            free(col_arrs);
            return NULL;
        }
        npy_intp sz = PyArray_SIZE(col_arrs[c]);
        if (c == 0) n_rows = sz;
        else if (sz != n_rows) {
            PyErr_SetString(PyExc_ValueError, "All arrays must have the same length");
            for (Py_ssize_t j = 0; j <= c; j++) Py_DECREF(col_arrs[j]);
            free(col_arrs);
            return NULL;
        }
    }

    /* Read cardinalities */
    npy_int64 *cards = (npy_int64 *)malloc(n_cols * sizeof(npy_int64));
    if (!cards) {
        for (Py_ssize_t c = 0; c < n_cols; c++) Py_DECREF(col_arrs[c]);
        free(col_arrs);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t c = 0; c < n_cols; c++) {
        cards[c] = PyLong_AsLongLong(PyTuple_GET_ITEM(cards_obj, c));
        if (cards[c] == -1 && PyErr_Occurred()) {
            for (Py_ssize_t j = 0; j < n_cols; j++) Py_DECREF(col_arrs[j]);
            free(col_arrs); free(cards);
            return NULL;
        }
    }

    /* Precompute multipliers: mult[c] = product of cards[c+1..n-1] */
    npy_int64 *mult = (npy_int64 *)malloc(n_cols * sizeof(npy_int64));
    if (!mult) {
        for (Py_ssize_t c = 0; c < n_cols; c++) Py_DECREF(col_arrs[c]);
        free(col_arrs); free(cards);
        return PyErr_NoMemory();
    }
    mult[n_cols - 1] = 1;
    for (Py_ssize_t c = n_cols - 2; c >= 0; c--)
        mult[c] = mult[c + 1] * cards[c + 1];

    /* Build output in single pass */
    npy_intp dims[1] = {n_rows};
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    if (!out) {
        for (Py_ssize_t c = 0; c < n_cols; c++) Py_DECREF(col_arrs[c]);
        free(col_arrs); free(cards); free(mult);
        return NULL;
    }
    npy_int64 *result = (npy_int64 *)PyArray_DATA(out);

    /* Get data pointers */
    const npy_int64 **col_data = (const npy_int64 **)malloc(n_cols * sizeof(npy_int64 *));
    if (!col_data) {
        for (Py_ssize_t c = 0; c < n_cols; c++) Py_DECREF(col_arrs[c]);
        free(col_arrs); free(cards); free(mult);
        Py_DECREF(out);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t c = 0; c < n_cols; c++)
        col_data[c] = (const npy_int64 *)PyArray_DATA(col_arrs[c]);

    /* Single pass: result[i] = sum(col_data[c][i] * mult[c]) */
    for (npy_intp i = 0; i < n_rows; i++) {
        npy_int64 key = 0;
        for (Py_ssize_t c = 0; c < n_cols; c++)
            key += col_data[c][i] * mult[c];
        result[i] = key;
    }

    for (Py_ssize_t c = 0; c < n_cols; c++) Py_DECREF(col_arrs[c]);
    free(col_arrs); free(cards); free(mult); free((void *)col_data);
    return (PyObject *)out;
}


/* ================================================================
 * encode_strings: O(n) hash-based string -> integer encoding
 *
 * encode_strings(array) -> (codes, n_unique)
 *   array: numpy array of hashable objects (strings, etc.)
 *
 * Returns:
 *   codes:    int64 array of integer codes (0..n_unique-1), first-seen order
 *   n_unique: int
 *
 * Replaces np.unique(return_inverse=True) which is O(n log n).
 * ================================================================ */

typedef struct {
    Py_hash_t hash;
    PyObject *obj;      /* borrowed ref; NULL = unoccupied */
    npy_intp code;
} StrHashEntry;

static int str_occupied(const void *entry)
    { return ((const StrHashEntry *)entry)->obj != NULL; }

/* ctx carries the PyObject* being looked up.  Returns -1/0/1
 * matching PyObject_RichCompareBool's convention. */
static int str_equal(const void *entry, npy_int64 hash_key, void *ctx)
{
    const StrHashEntry *e = (const StrHashEntry *)entry;
    if (e->hash != (Py_hash_t)hash_key) return 0;
    return PyObject_RichCompareBool(e->obj, (PyObject *)ctx, Py_EQ);
}

static PyObject *
accel_encode_strings(PyObject *self, PyObject *args)
{
    PyObject *arr_obj;
    if (!PyArg_ParseTuple(args, "O", &arr_obj))
        return NULL;

    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(
        arr_obj, NPY_OBJECT, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr) return NULL;

    npy_intp n = PyArray_SIZE(arr);
    PyObject **data = (PyObject **)PyArray_DATA(arr);

    /* Hash table */
    npy_intp table_size;
    StrHashEntry *table = (StrHashEntry *)ht_alloc(n, sizeof(StrHashEntry),
                                                   &table_size);
    if (!table) { Py_DECREF(arr); return PyErr_NoMemory(); }

    /* Output codes */
    npy_intp dims[1] = {n};
    PyArrayObject *codes_out = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INT64, 0);
    if (!codes_out) { free(table); Py_DECREF(arr); return NULL; }
    npy_int64 *codes = (npy_int64 *)PyArray_DATA(codes_out);

    npy_intp n_unique = 0;

    for (npy_intp i = 0; i < n; i++) {
        PyObject *obj = data[i];
        Py_hash_t h = PyObject_Hash(obj);
        if (h == -1 && PyErr_Occurred()) {
            free(table); Py_DECREF(arr); Py_DECREF(codes_out);
            return NULL;
        }

        int found, err;
        npy_intp slot = ht_probe(table, sizeof(StrHashEntry), table_size,
                                 (npy_int64)h, obj, &found, &err,
                                 str_occupied, str_equal);
        if (err) {
            free(table); Py_DECREF(arr); Py_DECREF(codes_out);
            return NULL;
        }

        if (!found) {
            table[slot].hash = h;
            table[slot].obj = obj;  /* borrowed ref, arr keeps it alive */
            table[slot].code = n_unique;
            n_unique++;
        }
        codes[i] = table[slot].code;
    }

    free(table);
    Py_DECREF(arr);

    PyObject *result = Py_BuildValue("(On)", codes_out, (Py_ssize_t)n_unique);
    Py_DECREF(codes_out);
    return result;
}


/* ================================================================
 * group_indices: O(n) hash-based group index construction
 *
 * group_indices(key_array) -> (first_seen_indices, group_row_lists, n_groups)
 *   key_array:  int64 array (composite or single-column key)
 *
 * Returns:
 *   first_seen_indices: int64 array of length n_groups
 *   group_row_lists:    Python list of int64 arrays (row indices per group)
 *   n_groups:           int
 *
 * Replaces np.unique + argsort + split with a single O(n) pass.
 * ================================================================ */

typedef struct {
    npy_int64 key;
    npy_intp label;     /* assigned group label */
    int occupied;
} GroupHashEntry;

static int group_occupied(const void *entry)
    { return ((const GroupHashEntry *)entry)->occupied; }
static int group_equal(const void *entry, npy_int64 hash_key, void *ctx)
    { (void)ctx; return ((const GroupHashEntry *)entry)->key == hash_key; }

static PyObject *
accel_group_indices(PyObject *self, PyObject *args)
{
    PyObject *key_obj;
    if (!PyArg_ParseTuple(args, "O", &key_obj))
        return NULL;

    PyArrayObject *key_arr = as_contig(key_obj, NPY_INT64);
    if (!key_arr) return NULL;

    npy_intp n = PyArray_SIZE(key_arr);
    const npy_int64 *keys = (const npy_int64 *)PyArray_DATA(key_arr);

    /* Allocate hash table */
    npy_intp table_size;
    GroupHashEntry *table = (GroupHashEntry *)ht_alloc(
        n, sizeof(GroupHashEntry), &table_size);
    if (!table) { Py_DECREF(key_arr); return PyErr_NoMemory(); }

    /* Pass 1: assign labels, count per group */
    npy_intp *labels = (npy_intp *)malloc(n * sizeof(npy_intp));
    npy_intp groups_cap = 256;
    npy_intp *first_seen = (npy_intp *)malloc(groups_cap * sizeof(npy_intp));
    npy_intp *counts = (npy_intp *)calloc(groups_cap, sizeof(npy_intp));
    if (!labels || !first_seen || !counts) {
        free(labels); free(first_seen); free(counts); free(table);
        Py_DECREF(key_arr);
        return PyErr_NoMemory();
    }
    npy_intp n_groups = 0;

    int found, err;  /* err always 0 — group callbacks are infallible */
    for (npy_intp i = 0; i < n; i++) {
        npy_int64 k = keys[i];
        npy_intp h = ht_probe(table, sizeof(GroupHashEntry), table_size,
                               k, NULL, &found, &err,
                               group_occupied, group_equal);

        if (!found) {
            table[h].key = k;
            table[h].occupied = 1;
            table[h].label = n_groups;
            if (n_groups >= groups_cap) {
                npy_intp new_cap = groups_cap * 2;
                npy_intp *tmp_fs = (npy_intp *)realloc(first_seen, new_cap * sizeof(npy_intp));
                npy_intp *tmp_ct = (npy_intp *)realloc(counts, new_cap * sizeof(npy_intp));
                if (!tmp_fs || !tmp_ct) {
                    free(tmp_fs ? tmp_fs : first_seen);
                    free(tmp_ct ? tmp_ct : counts);
                    free(labels); free(table); Py_DECREF(key_arr);
                    return PyErr_NoMemory();
                }
                first_seen = tmp_fs;
                counts = tmp_ct;
                memset(counts + groups_cap, 0, (new_cap - groups_cap) * sizeof(npy_intp));
                groups_cap = new_cap;
            }
            first_seen[n_groups] = i;
            n_groups++;
        }
        npy_intp lbl = table[h].label;
        labels[i] = lbl;
        counts[lbl]++;
    }
    free(table);
    Py_DECREF(key_arr);

    /* Allocate per-group output arrays */
    PyObject *group_list = PyList_New(n_groups);
    if (!group_list) {
        free(labels); free(first_seen); free(counts);
        return NULL;
    }

    npy_intp **group_data = (npy_intp **)malloc(n_groups * sizeof(npy_intp *));
    npy_intp *group_pos = (npy_intp *)calloc(n_groups, sizeof(npy_intp));
    if (!group_data || !group_pos) {
        free(group_data); free(group_pos);
        free(labels); free(first_seen); free(counts);
        Py_DECREF(group_list);
        return PyErr_NoMemory();
    }

    for (npy_intp g = 0; g < n_groups; g++) {
        npy_intp dims[1] = {counts[g]};
        PyArrayObject *arr = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INTP, 0);
        if (!arr) {
            free(group_data); free(group_pos);
            free(labels); free(first_seen); free(counts);
            Py_DECREF(group_list);
            return NULL;
        }
        group_data[g] = (npy_intp *)PyArray_DATA(arr);
        PyList_SET_ITEM(group_list, g, (PyObject *)arr);  /* steals ref */
    }
    free(counts);

    /* Pass 2: scatter row indices into per-group arrays */
    for (npy_intp i = 0; i < n; i++) {
        npy_intp lbl = labels[i];
        group_data[lbl][group_pos[lbl]++] = i;
    }
    free(labels);
    free(group_data);
    free(group_pos);

    /* Build first_seen output array */
    npy_intp fs_dims[1] = {n_groups};
    PyArrayObject *fs_out = (PyArrayObject *)PyArray_EMPTY(1, fs_dims, NPY_INTP, 0);
    if (!fs_out) { free(first_seen); Py_DECREF(group_list); return NULL; }
    npy_intp *fs_data = (npy_intp *)PyArray_DATA(fs_out);
    for (npy_intp g = 0; g < n_groups; g++)
        fs_data[g] = first_seen[g];
    free(first_seen);

    PyObject *result = Py_BuildValue("(OOn)", fs_out, group_list, (Py_ssize_t)n_groups);
    Py_DECREF(fs_out);
    Py_DECREF(group_list);
    return result;
}


/* ================================================================
 * Hash join: build (left_indices, right_indices) for equi-join
 *
 * CSR layout: two-pass build produces contiguous per-key storage.
 * Pass 1 counts right-side rows per key, prefix sum computes offsets,
 * Pass 2 scatters row indices into a flat rows[] array.
 * Emit phase does sequential reads — no pointer chasing.
 *
 * Input arrays are coerced to contiguous int64.
 * ================================================================ */

typedef struct {
    npy_int64 key;
    npy_intp count;     /* number of right-side rows for this key */
    npy_intp offset;    /* index into flat rows[] array */
} JoinHashEntry;

static int join_occupied(const void *entry)
    { return ((const JoinHashEntry *)entry)->count > 0; }
static int join_equal(const void *entry, npy_int64 hash_key, void *ctx)
    { (void)ctx; return ((const JoinHashEntry *)entry)->key == hash_key; }


/* Build CSR hash table on right-side keys. Returns 0 on success, -1 on error.
 * Caller must free(*out_table) and free(*out_rows). */
static int
build_join_table(const npy_int64 *right, npy_intp right_n,
                 JoinHashEntry **out_table, npy_intp *out_table_size,
                 npy_intp **out_rows)
{
    npy_intp table_size;
    JoinHashEntry *table = (JoinHashEntry *)ht_alloc(
        right_n, sizeof(JoinHashEntry), &table_size);
    if (!table) { PyErr_NoMemory(); return -1; }

    /* Pass 1: probe + count */
    int found, err;
    for (npy_intp i = 0; i < right_n; i++) {
        npy_intp h = ht_probe(table, sizeof(JoinHashEntry), table_size,
                               right[i], NULL, &found, &err,
                               join_occupied, join_equal);
        if (!found)
            table[h].key = right[i];
        table[h].count++;
    }

    /* Exclusive prefix sum over occupied slots to fill offset */
    npy_intp running = 0;
    for (npy_intp i = 0; i < table_size; i++) {
        if (table[i].count > 0) {
            table[i].offset = running;
            running += table[i].count;
        }
    }

    /* Allocate flat rows array */
    npy_intp *rows = (npy_intp *)malloc(right_n * sizeof(npy_intp));
    /* Temporary write positions (copy of offsets, bumped during scatter) */
    npy_intp *wpos = (npy_intp *)malloc(table_size * sizeof(npy_intp));
    if (!rows || !wpos) {
        free(table); free(rows); free(wpos);
        PyErr_NoMemory();
        return -1;
    }
    for (npy_intp i = 0; i < table_size; i++)
        wpos[i] = table[i].offset;

    /* Pass 2: probe + scatter row indices */
    for (npy_intp i = 0; i < right_n; i++) {
        npy_intp h = ht_probe(table, sizeof(JoinHashEntry), table_size,
                               right[i], NULL, &found, &err,
                               join_occupied, join_equal);
        rows[wpos[h]++] = i;
    }
    free(wpos);

    *out_table = table;
    *out_table_size = table_size;
    *out_rows = rows;
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
    const npy_int64 *right_data = (const npy_int64 *)PyArray_DATA(right_arr);

    JoinHashEntry *table; npy_intp table_size; npy_intp *rows;
    if (build_join_table(right_data, right_n, &table, &table_size, &rows) < 0) {
        Py_DECREF(left_arr); Py_DECREF(right_arr);
        return NULL;
    }

    /* Count pass: probe each left key, cache slot, sum counts */
    npy_intp *slots = (npy_intp *)malloc(left_n * sizeof(npy_intp));
    if (!slots) {
        free(table); free(rows);
        Py_DECREF(left_arr); Py_DECREF(right_arr);
        return PyErr_NoMemory();
    }

    int found, err;
    npy_intp total = 0;
    for (npy_intp i = 0; i < left_n; i++) {
        npy_intp h = ht_probe(table, sizeof(JoinHashEntry), table_size,
                               left[i], NULL, &found, &err,
                               join_occupied, join_equal);
        if (found) {
            slots[i] = h;
            total += table[h].count;
        } else {
            slots[i] = -1;
        }
    }

    npy_intp dims[1] = {total};
    PyArrayObject *li_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    PyArrayObject *ri_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    if (!li_arr || !ri_arr) {
        Py_XDECREF(li_arr); Py_XDECREF(ri_arr);
        free(slots); free(table); free(rows);
        Py_DECREF(left_arr); Py_DECREF(right_arr);
        return NULL;
    }

    /* Emit pass: sequential scan of rows[offset..offset+count] */
    npy_intp *li = (npy_intp *)PyArray_DATA(li_arr);
    npy_intp *ri = (npy_intp *)PyArray_DATA(ri_arr);
    npy_intp pos = 0;

    for (npy_intp i = 0; i < left_n; i++) {
        npy_intp h = slots[i];
        if (h >= 0) {
            npy_intp off = table[h].offset;
            npy_intp cnt = table[h].count;
            for (npy_intp j = 0; j < cnt; j++) {
                li[pos] = i;
                ri[pos] = rows[off + j];
                pos++;
            }
        }
    }

    free(slots); free(table); free(rows);
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
    const npy_int64 *right_data = (const npy_int64 *)PyArray_DATA(right_arr);

    JoinHashEntry *table; npy_intp table_size; npy_intp *rows;
    if (build_join_table(right_data, right_n, &table, &table_size, &rows) < 0) {
        Py_DECREF(left_arr); Py_DECREF(right_arr);
        return NULL;
    }

    /* Count pass: probe each left key, cache slot, sum counts */
    npy_intp *slots = (npy_intp *)malloc(left_n * sizeof(npy_intp));
    if (!slots) {
        free(table); free(rows);
        Py_DECREF(left_arr); Py_DECREF(right_arr);
        return PyErr_NoMemory();
    }

    int found, err;
    npy_intp total = 0;
    int has_null = 0;
    for (npy_intp i = 0; i < left_n; i++) {
        npy_intp h = ht_probe(table, sizeof(JoinHashEntry), table_size,
                               left[i], NULL, &found, &err,
                               join_occupied, join_equal);
        if (found) {
            slots[i] = h;
            total += table[h].count;
        } else {
            slots[i] = -1;
            total++;
            has_null = 1;
        }
    }

    npy_intp dims[1] = {total};
    PyArrayObject *li_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    PyArrayObject *ri_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    if (!li_arr || !ri_arr) {
        Py_XDECREF(li_arr); Py_XDECREF(ri_arr);
        free(slots); free(table); free(rows);
        Py_DECREF(left_arr); Py_DECREF(right_arr);
        return NULL;
    }

    /* Emit pass: sequential scan of rows[offset..offset+count] */
    npy_intp *li = (npy_intp *)PyArray_DATA(li_arr);
    npy_intp *ri = (npy_intp *)PyArray_DATA(ri_arr);
    npy_intp pos = 0;

    for (npy_intp i = 0; i < left_n; i++) {
        npy_intp h = slots[i];
        if (h >= 0) {
            npy_intp off = table[h].offset;
            npy_intp cnt = table[h].count;
            for (npy_intp j = 0; j < cnt; j++) {
                li[pos] = i;
                ri[pos] = rows[off + j];
                pos++;
            }
        } else {
            li[pos] = i;
            ri[pos] = -1;
            pos++;
        }
    }

    free(slots); free(table); free(rows);
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
    {"composite_key", accel_composite_key, METH_VARARGS,
     "Positional encoding: composite_key(arrays, cardinalities) -> int64 key"},
    {"encode_strings", accel_encode_strings, METH_VARARGS,
     "O(n) hash-based string encoding: encode_strings(array) -> (codes, n_unique)"},
    {"group_indices", accel_group_indices, METH_VARARGS,
     "O(n) hash-based group labeling: group_indices(key) -> (labels, first_seen, n_groups)"},
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
