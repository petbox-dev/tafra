/*
 * tafra/_accel.c — C extension for hot-path acceleration (v4).
 *
 *   - Power-of-2 hash tables, Fibonacci hashing on the HIGH product bits
 *     (keys that share trailing zero bits, such as datetime64[ns] dates,
 *     still spread over the whole table)
 *   - CSR (flat contiguous) layout for join hash tables
 *   - Each key is read once: slots are cached between passes, so a key
 *     array that another thread writes cannot corrupt the table
 *   - The GIL is released around every loop that touches no Python object
 *
 * All functions accept and return numpy arrays. Falls back gracefully
 * if this module fails to compile — pure-Python paths remain available.
 *
 * Every array argument must be one-dimensional; it is coerced to a
 * contiguous array of the expected type. Group labels are validated to be
 * in [0, group_count); a label outside that range raises ValueError.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>

/* Golden ratio constant for multiplicative hashing (Knuth).
 * phi = (sqrt(5)-1)/2 * 2^64 ~ 0x9E3779B97F4A7C15 */
#define GOLDEN_HASH 0x9E3779B97F4A7C15ULL

/* Fibonacci hashing: the top log2(table size) bits of key * GOLDEN_HASH.
 * The low bits of a product depend only on the low bits of the key, so
 * masking them clusters keys that end in zero bits; the high bits mix
 * every key bit. */
#define HASH_SLOT(key, shift) ((npy_intp)(((npy_uint64)(key) * GOLDEN_HASH) >> (shift)))


/* Size, probe mask and hash shift of a power-of-2 open-addressing table. */
typedef struct {
    npy_intp size;
    npy_intp mask;
    int shift;
} HashGeometry;


/* Table for key_count keys at a load factor of at most 50% (minimum 16). */
static HashGeometry
hash_geometry(npy_intp key_count)
{
    HashGeometry geometry = {16, 15, 60};
    while (geometry.size < key_count * 2) {
        geometry.size <<= 1;
        geometry.shift--;
    }
    geometry.mask = geometry.size - 1;
    return geometry;
}


/* malloc that never asks for 0 bytes: malloc(0) may return NULL, which
 * would be reported as a false MemoryError. */
static void *
malloc_nonzero(size_t byte_count)
{
    return malloc(byte_count ? byte_count : 1);
}


/* calloc with the same guard: never asks for 0 elements. */
static void *
calloc_nonzero(size_t element_count, size_t element_size)
{
    return calloc(element_count ? element_count : 1, element_size);
}


/* Coerce to a contiguous one-dimensional array of the given type.
 * Caller must Py_DECREF. */
static PyArrayObject *
as_contiguous_vector(PyObject *obj, int typenum)
{
    PyArrayObject *array = (PyArrayObject *)PyArray_FROM_OTF(
        obj, typenum, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (array && PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "expected a one-dimensional array");
        Py_DECREF(array);
        return NULL;
    }
    return array;
}


/* ================================================================
 * GroupBy aggregations over (labels, data); one pass, two for variance
 *
 * All groupby functions take:
 *   labels:      int64 array of group indices in [0, group_count)
 *   data:        float64 array of values (int64 for the *_int64 kernels),
 *                the same length as labels
 *   group_count: number of groups
 *   option_flag: optional bool, read per kernel as
 *                skip_nan (float kernels): True ignores NaN (np.nansum and
 *                friends); a group with no non-NaN value gives NaN (0 for sum);
 *                unsigned_values (groupby_min_int64 / groupby_max_int64):
 *                compare the data as uint64. groupby_sum_int64 ignores it.
 * ================================================================ */

typedef struct {
    PyArrayObject *labels_array;
    PyArrayObject *data_array;
    const npy_int64 *labels;
    const void *data;
    npy_intp row_count;
    npy_intp group_count;
    int option_flag;    /* skip_nan or unsigned_values; see above */
} GroupbyInput;


static void
release_groupby_input(GroupbyInput *input)
{
    Py_XDECREF(input->labels_array);
    Py_XDECREF(input->data_array);
}


/* Parse (labels, data, group_count[, option_flag]). Returns 0, or -1 with
 * an exception set. */
static int
parse_groupby_input(PyObject *args, int data_typenum, GroupbyInput *input)
{
    PyObject *labels_obj = NULL, *data_obj = NULL;
    Py_ssize_t group_count = 0;
    int option_flag = 0;

    memset(input, 0, sizeof(*input));
    if (!PyArg_ParseTuple(args, "OOn|p", &labels_obj, &data_obj, &group_count, &option_flag))
        return -1;
    if (group_count < 0) {
        PyErr_SetString(PyExc_ValueError, "group_count must be non-negative");
        return -1;
    }

    input->labels_array = as_contiguous_vector(labels_obj, NPY_INT64);
    if (!input->labels_array) return -1;
    input->row_count = PyArray_SIZE(input->labels_array);
    input->labels = (const npy_int64 *)PyArray_DATA(input->labels_array);

    input->data_array = as_contiguous_vector(data_obj, data_typenum);
    if (!input->data_array) { release_groupby_input(input); return -1; }
    if (PyArray_SIZE(input->data_array) != input->row_count) {
        PyErr_SetString(PyExc_ValueError, "labels and data must have the same length");
        release_groupby_input(input);
        return -1;
    }
    input->data = PyArray_DATA(input->data_array);
    input->group_count = (npy_intp)group_count;
    input->option_flag = option_flag;
    return 0;
}


/* True when label is outside [0, group_count). */
#define LABEL_OUT_OF_RANGE(label, group_count) \
    ((npy_uint64)(label) >= (npy_uint64)(group_count))


static PyObject *
raise_label_out_of_range(void)
{
    PyErr_SetString(PyExc_ValueError, "group label outside [0, group_count)");
    return NULL;
}


static PyObject *
accel_groupby_sum(PyObject *self, PyObject *args)
{
    GroupbyInput input;
    if (parse_groupby_input(args, NPY_FLOAT64, &input) < 0) return NULL;

    npy_intp dims[1] = {input.group_count};
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    if (!out) { release_groupby_input(&input); return NULL; }

    const npy_int64 *labels = input.labels;
    const npy_float64 *data = (const npy_float64 *)input.data;
    npy_float64 *result = (npy_float64 *)PyArray_DATA(out);
    int invalid_label = 0;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < input.row_count; i++) {
        npy_int64 group = labels[i];
        if (LABEL_OUT_OF_RANGE(group, input.group_count)) { invalid_label = 1; break; }
        npy_float64 value = data[i];
        if (!input.option_flag || value == value)
            result[group] += value;
    }
    Py_END_ALLOW_THREADS

    release_groupby_input(&input);
    if (invalid_label) { Py_DECREF(out); return raise_label_out_of_range(); }
    return (PyObject *)out;
}


/* Exact integer sum. Wraps modulo 2^64 like numpy; a uint64 sum is the
 * same bit pattern, so callers pass uint64 data viewed as int64. */
static PyObject *
accel_groupby_sum_int64(PyObject *self, PyObject *args)
{
    GroupbyInput input;
    if (parse_groupby_input(args, NPY_INT64, &input) < 0) return NULL;

    npy_intp dims[1] = {input.group_count};
    PyArrayObject *out = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    if (!out) { release_groupby_input(&input); return NULL; }

    const npy_int64 *labels = input.labels;
    const npy_int64 *data = (const npy_int64 *)input.data;
    npy_uint64 *result = (npy_uint64 *)PyArray_DATA(out);
    int invalid_label = 0;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < input.row_count; i++) {
        npy_int64 group = labels[i];
        if (LABEL_OUT_OF_RANGE(group, input.group_count)) { invalid_label = 1; break; }
        /* unsigned arithmetic: overflow wraps instead of being undefined */
        result[group] += (npy_uint64)data[i];
    }
    Py_END_ALLOW_THREADS

    release_groupby_input(&input);
    if (invalid_label) { Py_DECREF(out); return raise_label_out_of_range(); }
    return (PyObject *)out;
}


static PyObject *
accel_groupby_mean(PyObject *self, PyObject *args)
{
    GroupbyInput input;
    if (parse_groupby_input(args, NPY_FLOAT64, &input) < 0) return NULL;

    npy_intp dims[1] = {input.group_count};
    PyArrayObject *sums = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    PyArrayObject *counts = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    if (!sums || !counts) {
        Py_XDECREF(sums); Py_XDECREF(counts);
        release_groupby_input(&input);
        return NULL;
    }

    const npy_int64 *labels = input.labels;
    const npy_float64 *data = (const npy_float64 *)input.data;
    npy_float64 *sum = (npy_float64 *)PyArray_DATA(sums);
    npy_int64 *count = (npy_int64 *)PyArray_DATA(counts);
    int invalid_label = 0;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < input.row_count; i++) {
        npy_int64 group = labels[i];
        if (LABEL_OUT_OF_RANGE(group, input.group_count)) { invalid_label = 1; break; }
        npy_float64 value = data[i];
        if (input.option_flag && value != value) continue;
        sum[group] += value;
        count[group]++;
    }
    for (npy_intp group = 0; group < input.group_count; group++)
        sum[group] = count[group] > 0 ? sum[group] / (npy_float64)count[group] : NPY_NAN;
    Py_END_ALLOW_THREADS

    Py_DECREF(counts);
    release_groupby_input(&input);
    if (invalid_label) { Py_DECREF(sums); return raise_label_out_of_range(); }
    return (PyObject *)sums;
}


/* Grouped population variance, two-pass as numpy computes it */
static PyObject *
accel_groupby_var(PyObject *self, PyObject *args)
{
    GroupbyInput input;
    if (parse_groupby_input(args, NPY_FLOAT64, &input) < 0) return NULL;

    npy_intp dims[1] = {input.group_count};
    PyArrayObject *counts = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT64, 0);
    PyArrayObject *means = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    PyArrayObject *variances = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    if (!counts || !means || !variances) {
        Py_XDECREF(counts); Py_XDECREF(means); Py_XDECREF(variances);
        release_groupby_input(&input);
        return NULL;
    }

    const npy_int64 *labels = input.labels;
    const npy_float64 *data = (const npy_float64 *)input.data;
    npy_int64 *count = (npy_int64 *)PyArray_DATA(counts);
    npy_float64 *mean = (npy_float64 *)PyArray_DATA(means);
    npy_float64 *variance = (npy_float64 *)PyArray_DATA(variances);
    int invalid_label = 0;

    /* Two passes, as numpy: the group mean first, then the mean squared
     * deviation from it. A one-pass update (Welford) loses about 1e-7 of
     * relative accuracy on values offset by 1e9; this matches numpy. */
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < input.row_count; i++) {
        npy_int64 group = labels[i];
        if (LABEL_OUT_OF_RANGE(group, input.group_count)) { invalid_label = 1; break; }
        npy_float64 value = data[i];
        if (input.option_flag && value != value) continue;
        count[group]++;
        mean[group] += value;  /* the sum, until divided below */
    }
    if (!invalid_label) {
        for (npy_intp group = 0; group < input.group_count; group++)
            mean[group] = count[group] > 0 ? mean[group] / (npy_float64)count[group] : NPY_NAN;
        for (npy_intp i = 0; i < input.row_count; i++) {
            /* read each label once and check it again: the array may be
             * shared with a thread that writes it between the passes */
            npy_int64 group = labels[i];
            if (LABEL_OUT_OF_RANGE(group, input.group_count)) { invalid_label = 1; break; }
            npy_float64 value = data[i];
            if (input.option_flag && value != value) continue;
            npy_float64 deviation = value - mean[group];
            variance[group] += deviation * deviation;
        }
        for (npy_intp group = 0; group < input.group_count; group++)
            variance[group] = count[group] > 0 ? variance[group] / (npy_float64)count[group] : NPY_NAN;
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(counts);
    Py_DECREF(means);
    release_groupby_input(&input);
    if (invalid_label) { Py_DECREF(variances); return raise_label_out_of_range(); }
    return (PyObject *)variances;
}


/* Grouped min or max with numpy NaN semantics.
 *   skip_nan false (np.min):    any NaN in a group makes the result NaN.
 *   skip_nan true  (np.nanmin): NaN is ignored; an all-NaN group gives NaN. */
static PyObject *
groupby_extreme(PyObject *args, int want_maximum)
{
    GroupbyInput input;
    if (parse_groupby_input(args, NPY_FLOAT64, &input) < 0) return NULL;

    npy_intp dims[1] = {input.group_count};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!out) { release_groupby_input(&input); return NULL; }

    const npy_int64 *labels = input.labels;
    const npy_float64 *data = (const npy_float64 *)input.data;
    npy_float64 *result = (npy_float64 *)PyArray_DATA(out);
    const int skip_nan = input.option_flag;
    int invalid_label = 0;

    Py_BEGIN_ALLOW_THREADS
    npy_float64 initial = skip_nan ? NPY_NAN : (want_maximum ? -NPY_INFINITY : NPY_INFINITY);
    for (npy_intp group = 0; group < input.group_count; group++)
        result[group] = initial;

    for (npy_intp i = 0; i < input.row_count; i++) {
        npy_int64 group = labels[i];
        if (LABEL_OUT_OF_RANGE(group, input.group_count)) { invalid_label = 1; break; }
        npy_float64 value = data[i];
        npy_float64 current = result[group];
        if (skip_nan) {
            /* current is NaN until the first non-NaN value arrives */
            if (value == value
                && (current != current || (want_maximum ? value > current : value < current)))
                result[group] = value;
        } else {
            /* once NaN, stay NaN; a NaN value fails every comparison */
            if (current == current
                && !(want_maximum ? value <= current : value >= current))
                result[group] = value;
        }
    }
    Py_END_ALLOW_THREADS

    release_groupby_input(&input);
    if (invalid_label) { Py_DECREF(out); return raise_label_out_of_range(); }
    return (PyObject *)out;
}


/* Grouped exact integer min or max. unsigned_values compares the int64
 * data as uint64 (callers pass uint64 data viewed as int64). Groups are
 * assumed non-empty; an empty group keeps the initial extreme. */
static PyObject *
groupby_integer_extreme(PyObject *args, int want_maximum)
{
    GroupbyInput input;
    if (parse_groupby_input(args, NPY_INT64, &input) < 0) return NULL;
    const int unsigned_values = input.option_flag;

    npy_intp dims[1] = {input.group_count};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
    if (!out) { release_groupby_input(&input); return NULL; }

    const npy_int64 *labels = input.labels;
    const npy_int64 *data = (const npy_int64 *)input.data;
    npy_int64 *result = (npy_int64 *)PyArray_DATA(out);
    int invalid_label = 0;

    Py_BEGIN_ALLOW_THREADS
    npy_int64 initial;
    if (unsigned_values)
        initial = want_maximum ? 0 : (npy_int64)NPY_MAX_UINT64;
    else
        initial = want_maximum ? NPY_MIN_INT64 : NPY_MAX_INT64;
    for (npy_intp group = 0; group < input.group_count; group++)
        result[group] = initial;

    for (npy_intp i = 0; i < input.row_count; i++) {
        npy_int64 group = labels[i];
        if (LABEL_OUT_OF_RANGE(group, input.group_count)) { invalid_label = 1; break; }
        npy_int64 value = data[i];
        int replace;
        if (unsigned_values) {
            npy_uint64 unsigned_value = (npy_uint64)value;
            npy_uint64 unsigned_current = (npy_uint64)result[group];
            replace = want_maximum ? unsigned_value > unsigned_current
                                   : unsigned_value < unsigned_current;
        } else {
            replace = want_maximum ? value > result[group] : value < result[group];
        }
        if (replace) result[group] = value;
    }
    Py_END_ALLOW_THREADS

    release_groupby_input(&input);
    if (invalid_label) { Py_DECREF(out); return raise_label_out_of_range(); }
    return (PyObject *)out;
}


static PyObject *
accel_groupby_min_int64(PyObject *self, PyObject *args)
{
    return groupby_integer_extreme(args, 0);
}


static PyObject *
accel_groupby_max_int64(PyObject *self, PyObject *args)
{
    return groupby_integer_extreme(args, 1);
}


static PyObject *
accel_groupby_min(PyObject *self, PyObject *args)
{
    return groupby_extreme(args, 0);
}


static PyObject *
accel_groupby_max(PyObject *self, PyObject *args)
{
    return groupby_extreme(args, 1);
}


/* ================================================================
 * Composite key: positional encoding of multiple int64 columns
 *
 * composite_key(arrays, cardinalities) -> int64 array
 *   arrays:        tuple of int64 arrays (codes in [0, cardinality))
 *   cardinalities: tuple of ints (max value + 1 for each column)
 *
 * Computes: key[i] = col0[i]*card1*card2*... + col1[i]*card2*... + ...
 * Raises ValueError if the product of the cardinalities exceeds int64.
 * ================================================================ */

static PyObject *
accel_composite_key(PyObject *self, PyObject *args)
{
    PyObject *arrays_obj, *cardinalities_obj;
    if (!PyArg_ParseTuple(args, "OO", &arrays_obj, &cardinalities_obj))
        return NULL;

    if (!PyTuple_Check(arrays_obj) || !PyTuple_Check(cardinalities_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected two tuples");
        return NULL;
    }

    Py_ssize_t column_count = PyTuple_GET_SIZE(arrays_obj);
    if (column_count != PyTuple_GET_SIZE(cardinalities_obj) || column_count < 1) {
        PyErr_SetString(PyExc_ValueError, "arrays and cardinalities must have same positive length");
        return NULL;
    }

    PyArrayObject **column_arrays = (PyArrayObject **)calloc(column_count, sizeof(PyArrayObject *));
    const npy_int64 **column_data = (const npy_int64 **)calloc(column_count, sizeof(npy_int64 *));
    npy_uint64 *multipliers = (npy_uint64 *)calloc(column_count, sizeof(npy_uint64));
    PyArrayObject *out = NULL;
    if (!column_arrays || !column_data || !multipliers) {
        PyErr_NoMemory();
        goto cleanup;
    }

    npy_intp row_count = 0;
    for (Py_ssize_t column = 0; column < column_count; column++) {
        column_arrays[column] = as_contiguous_vector(PyTuple_GET_ITEM(arrays_obj, column), NPY_INT64);
        if (!column_arrays[column]) goto cleanup;
        npy_intp size = PyArray_SIZE(column_arrays[column]);
        if (column == 0) row_count = size;
        else if (size != row_count) {
            PyErr_SetString(PyExc_ValueError, "All arrays must have the same length");
            goto cleanup;
        }
        column_data[column] = (const npy_int64 *)PyArray_DATA(column_arrays[column]);
    }

    /* multipliers[c] = product of cardinalities[c+1 .. n-1], checked for overflow */
    npy_uint64 product = 1;
    for (Py_ssize_t column = column_count - 1; column >= 0; column--) {
        multipliers[column] = product;
        long long cardinality = PyLong_AsLongLong(PyTuple_GET_ITEM(cardinalities_obj, column));
        if (cardinality == -1 && PyErr_Occurred()) goto cleanup;
        if (cardinality < 1 || product > (npy_uint64)NPY_MAX_INT64 / (npy_uint64)cardinality) {
            PyErr_SetString(PyExc_ValueError,
                            "cardinalities must be positive and their product must fit int64");
            goto cleanup;
        }
        product *= (npy_uint64)cardinality;
    }

    npy_intp dims[1] = {row_count};
    out = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INT64, 0);
    if (!out) goto cleanup;
    npy_uint64 *result = (npy_uint64 *)PyArray_DATA(out);

    /* unsigned arithmetic: a code outside its cardinality wraps instead of
     * being undefined behavior */
    Py_BEGIN_ALLOW_THREADS
    if (column_count == 1) {
        memcpy(result, column_data[0], row_count * sizeof(npy_int64));
    } else if (column_count == 2) {
        const npy_int64 *first = column_data[0], *second = column_data[1];
        npy_uint64 first_multiplier = multipliers[0];
        for (npy_intp i = 0; i < row_count; i++)
            result[i] = (npy_uint64)first[i] * first_multiplier + (npy_uint64)second[i];
    } else if (column_count == 3) {
        const npy_int64 *first = column_data[0], *second = column_data[1], *third = column_data[2];
        npy_uint64 first_multiplier = multipliers[0], second_multiplier = multipliers[1];
        for (npy_intp i = 0; i < row_count; i++)
            result[i] = (npy_uint64)first[i] * first_multiplier
                      + (npy_uint64)second[i] * second_multiplier
                      + (npy_uint64)third[i];
    } else {
        for (npy_intp i = 0; i < row_count; i++) {
            npy_uint64 key = 0;
            for (Py_ssize_t column = 0; column < column_count; column++)
                key += (npy_uint64)column_data[column][i] * multipliers[column];
            result[i] = key;
        }
    }
    Py_END_ALLOW_THREADS

cleanup:
    if (column_arrays) {
        for (Py_ssize_t column = 0; column < column_count; column++)
            Py_XDECREF(column_arrays[column]);
    }
    free(column_arrays);
    free((void *)column_data);
    free(multipliers);
    if (PyErr_Occurred()) {
        Py_XDECREF(out);
        return NULL;
    }
    return (PyObject *)out;
}


/* ================================================================
 * encode_strings: O(n) hash-based object -> integer encoding
 *
 * encode_strings(array) -> (codes, unique_count)
 *   array: numpy array of hashable objects (strings, None, etc.)
 *
 * Returns:
 *   codes:        int64 array of integer codes (0..unique_count-1), first-seen order
 *   unique_count: int
 *
 * Equal hashes are confirmed with PyObject_RichCompareBool, so a hash
 * collision never merges two different values.
 * ================================================================ */

typedef struct {
    /* Borrowed reference into the input array; NULL marks an empty slot.
     * Safe on free-threaded builds only because callers pass a private
     * copy (arr.astype(object)) that no other thread can mutate while the
     * table is alive. */
    PyObject *obj;
    Py_hash_t hash;
    npy_intp code;
} StringHashEntry;

static PyObject *
accel_encode_strings(PyObject *self, PyObject *args)
{
    PyObject *arr_obj;
    if (!PyArg_ParseTuple(args, "O", &arr_obj))
        return NULL;

    PyArrayObject *arr = as_contiguous_vector(arr_obj, NPY_OBJECT);
    if (!arr) return NULL;

    npy_intp row_count = PyArray_SIZE(arr);
    PyObject **data = (PyObject **)PyArray_DATA(arr);

    HashGeometry geometry = hash_geometry(row_count);
    StringHashEntry *table = (StringHashEntry *)calloc(geometry.size, sizeof(StringHashEntry));
    if (!table) { Py_DECREF(arr); return PyErr_NoMemory(); }

    npy_intp dims[1] = {row_count};
    PyArrayObject *codes_out = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INT64, 0);
    if (!codes_out) { free(table); Py_DECREF(arr); return NULL; }
    npy_int64 *codes = (npy_int64 *)PyArray_DATA(codes_out);

    npy_intp unique_count = 0;

    for (npy_intp i = 0; i < row_count; i++) {
        PyObject *obj = data[i];
        Py_hash_t hash = PyObject_Hash(obj);
        if (hash == -1 && PyErr_Occurred()) {
            free(table); Py_DECREF(arr); Py_DECREF(codes_out);
            return NULL;
        }

        npy_intp slot = HASH_SLOT(hash, geometry.shift);
        while (table[slot].obj) {
            if (table[slot].hash == hash) {
                int equal = PyObject_RichCompareBool(table[slot].obj, obj, Py_EQ);
                if (equal < 0) {
                    free(table); Py_DECREF(arr); Py_DECREF(codes_out);
                    return NULL;
                }
                if (equal) break;
            }
            slot = (slot + 1) & geometry.mask;
        }

        if (!table[slot].obj) {
            table[slot].obj = obj;
            table[slot].hash = hash;
            table[slot].code = unique_count++;
        }
        codes[i] = table[slot].code;
    }

    free(table);
    Py_DECREF(arr);

    PyObject *result = Py_BuildValue("(On)", codes_out, (Py_ssize_t)unique_count);
    Py_DECREF(codes_out);
    return result;
}


/* ================================================================
 * Integer key hash table (shared by encode_int64 and group_indices)
 * ================================================================ */

typedef struct {
    npy_int64 key;
    /* label + 1, so that 0 marks an empty slot and a calloc'd table is
     * empty without touching every page */
    npy_intp label_plus_one;
} IntegerHashEntry;


/* Assign first-seen labels 0..group_count-1 to keys. Pure C, no GIL.
 * table must hold geometry.size zeroed entries; first_seen (optional)
 * receives the first row of each group. Returns group_count. */
static npy_intp
label_integer_keys(const npy_int64 *keys, npy_intp row_count, HashGeometry geometry,
                   IntegerHashEntry *table, npy_int64 *labels, npy_intp *first_seen)
{
    npy_intp group_count = 0;
    for (npy_intp i = 0; i < row_count; i++) {
        npy_int64 key = keys[i];
        npy_intp slot = HASH_SLOT(key, geometry.shift);
        while (table[slot].label_plus_one != 0 && table[slot].key != key)
            slot = (slot + 1) & geometry.mask;
        if (table[slot].label_plus_one == 0) {
            table[slot].key = key;
            table[slot].label_plus_one = group_count + 1;
            if (first_seen) first_seen[group_count] = i;
            group_count++;
        }
        labels[i] = table[slot].label_plus_one - 1;
    }
    return group_count;
}


/* ================================================================
 * encode_int64: O(n) hash-based dense encoding of int64 keys
 *
 * encode_int64(keys) -> (codes, unique_count)
 *   codes: int64 array of first-seen codes 0..unique_count-1
 * ================================================================ */

static PyObject *
accel_encode_int64(PyObject *self, PyObject *args)
{
    PyObject *keys_obj;
    if (!PyArg_ParseTuple(args, "O", &keys_obj))
        return NULL;

    PyArrayObject *keys_array = as_contiguous_vector(keys_obj, NPY_INT64);
    if (!keys_array) return NULL;
    npy_intp row_count = PyArray_SIZE(keys_array);

    HashGeometry geometry = hash_geometry(row_count);
    IntegerHashEntry *table = (IntegerHashEntry *)calloc(geometry.size, sizeof(IntegerHashEntry));
    if (!table) { Py_DECREF(keys_array); return PyErr_NoMemory(); }

    npy_intp dims[1] = {row_count};
    PyArrayObject *codes_out = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INT64, 0);
    if (!codes_out) { free(table); Py_DECREF(keys_array); return NULL; }

    const npy_int64 *keys = (const npy_int64 *)PyArray_DATA(keys_array);
    npy_int64 *codes = (npy_int64 *)PyArray_DATA(codes_out);
    npy_intp unique_count;

    Py_BEGIN_ALLOW_THREADS
    unique_count = label_integer_keys(keys, row_count, geometry, table, codes, NULL);
    Py_END_ALLOW_THREADS

    free(table);
    Py_DECREF(keys_array);

    PyObject *result = Py_BuildValue("(On)", codes_out, (Py_ssize_t)unique_count);
    Py_DECREF(codes_out);
    return result;
}


/* ================================================================
 * group_indices: O(n) hash-based group index construction
 *
 * group_indices(key_array) -> (first_seen_indices, group_row_lists, group_count)
 *   key_array:  int64 array (composite or single-column key)
 * ================================================================ */

static PyObject *
accel_group_indices(PyObject *self, PyObject *args)
{
    PyObject *key_obj;
    if (!PyArg_ParseTuple(args, "O", &key_obj))
        return NULL;

    PyArrayObject *key_array = as_contiguous_vector(key_obj, NPY_INT64);
    if (!key_array) return NULL;

    npy_intp row_count = PyArray_SIZE(key_array);
    const npy_int64 *keys = (const npy_int64 *)PyArray_DATA(key_array);

    HashGeometry geometry = hash_geometry(row_count);
    IntegerHashEntry *table = (IntegerHashEntry *)calloc(geometry.size, sizeof(IntegerHashEntry));
    /* group_count <= row_count, so per-group buffers sized by rows never grow */
    npy_int64 *labels = (npy_int64 *)malloc_nonzero(row_count * sizeof(npy_int64));
    npy_intp *first_seen = (npy_intp *)malloc_nonzero(row_count * sizeof(npy_intp));
    npy_intp *group_sizes = (npy_intp *)calloc_nonzero(row_count, sizeof(npy_intp));
    npy_intp **group_data = NULL;
    PyObject *group_list = NULL;
    if (!table || !labels || !first_seen || !group_sizes) {
        PyErr_NoMemory();
        goto error;
    }

    npy_intp group_count;
    Py_BEGIN_ALLOW_THREADS
    group_count = label_integer_keys(keys, row_count, geometry, table, labels, first_seen);
    for (npy_intp i = 0; i < row_count; i++)
        group_sizes[labels[i]]++;
    Py_END_ALLOW_THREADS
    free(table); table = NULL;

    group_list = PyList_New(group_count);
    group_data = (npy_intp **)malloc_nonzero(group_count * sizeof(npy_intp *));
    if (!group_list || !group_data) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        goto error;
    }
    for (npy_intp group = 0; group < group_count; group++) {
        npy_intp dims[1] = {group_sizes[group]};
        PyArrayObject *rows = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_INTP, 0);
        if (!rows) goto error;
        /* group_list owns rows (and keeps it alive until return or error),
         * so this interior pointer stays valid for the scatter below */
        group_data[group] = (npy_intp *)PyArray_DATA(rows);
        PyList_SET_ITEM(group_list, group, (PyObject *)rows);
    }

    /* Scatter row indices into per-group arrays; group_sizes becomes the
     * write position of each group. */
    Py_BEGIN_ALLOW_THREADS
    memset(group_sizes, 0, group_count * sizeof(npy_intp));
    for (npy_intp i = 0; i < row_count; i++) {
        npy_int64 group = labels[i];
        group_data[group][group_sizes[group]++] = i;
    }
    Py_END_ALLOW_THREADS

    npy_intp first_seen_dims[1] = {group_count};
    PyArrayObject *first_seen_out = (PyArrayObject *)PyArray_EMPTY(1, first_seen_dims, NPY_INTP, 0);
    if (!first_seen_out) goto error;
    memcpy(PyArray_DATA(first_seen_out), first_seen, group_count * sizeof(npy_intp));

    free(labels); free(first_seen); free(group_sizes); free(group_data);
    Py_DECREF(key_array);

    PyObject *result = Py_BuildValue("(OOn)", first_seen_out, group_list, (Py_ssize_t)group_count);
    Py_DECREF(first_seen_out);
    Py_DECREF(group_list);
    return result;

error:
    free(table); free(labels); free(first_seen); free(group_sizes); free(group_data);
    Py_XDECREF(group_list);
    Py_DECREF(key_array);
    return NULL;
}


/* ================================================================
 * Hash join: CSR layout + slot caching
 *
 * Build phase (right side):
 *   1. Probe each right key once, count per bucket, cache its slot.
 *   2. Prefix sum -> end offset of each bucket in a flat row buffer.
 *   3. Scatter right rows in reverse through the cached slots; each
 *      bucket's offset decrements to its start and rows stay ascending.
 *
 * Probe phase (left side):
 *   Pass 1: probe each left key once, cache the slot, sum the counts
 *           for exact output pre-allocation.
 *   Pass 2: emit pairs through the cached slots.
 *
 * No key is read twice, so another thread writing a key array during the
 * join gives a wrong answer for that key, never an out-of-bounds write.
 * ================================================================ */

typedef struct {
    npy_int64 key;
    npy_intp offset;    /* start index into flat row buffer */
    npy_intp count;     /* number of right rows with this key; 0 marks empty */
} CSRHashEntry;


typedef struct {
    HashGeometry geometry;
    CSRHashEntry *entries;
    npy_intp *rows;         /* right row indices grouped by key */
} CSRHashTable;


static void
free_csr_hash_table(CSRHashTable *table)
{
    free(table->entries);
    free(table->rows);
    table->entries = NULL;
    table->rows = NULL;
}


/* Allocate (GIL held), then build without the GIL.
 * Returns 0, or -1 with MemoryError set. */
static int
build_csr_hash_table(const npy_int64 *right, npy_intp right_count, CSRHashTable *table)
{
    table->geometry = hash_geometry(right_count);
    table->entries = (CSRHashEntry *)calloc(table->geometry.size, sizeof(CSRHashEntry));
    table->rows = (npy_intp *)malloc_nonzero(right_count * sizeof(npy_intp));
    npy_intp *right_slots = (npy_intp *)malloc_nonzero(right_count * sizeof(npy_intp));
    if (!table->entries || !table->rows || !right_slots) {
        free(right_slots);
        free_csr_hash_table(table);
        PyErr_NoMemory();
        return -1;
    }

    const HashGeometry geometry = table->geometry;
    CSRHashEntry *entries = table->entries;
    npy_intp *rows = table->rows;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < right_count; i++) {
        npy_int64 key = right[i];
        npy_intp slot = HASH_SLOT(key, geometry.shift);
        while (entries[slot].count > 0 && entries[slot].key != key)
            slot = (slot + 1) & geometry.mask;
        entries[slot].key = key;
        entries[slot].count++;
        right_slots[i] = slot;
    }

    npy_intp running = 0;
    for (npy_intp slot = 0; slot < geometry.size; slot++) {
        if (entries[slot].count > 0) {
            running += entries[slot].count;
            entries[slot].offset = running;
        }
    }

    for (npy_intp i = right_count - 1; i >= 0; i--)
        rows[--entries[right_slots[i]].offset] = i;
    Py_END_ALLOW_THREADS

    free(right_slots);
    return 0;
}


/* Probe pass: cache each left key's slot (-1 if absent) and return the
 * number of output rows. An unmatched left row counts as one output row
 * when keep_unmatched is set (left join). Returns -1 when the count does
 * not fit in npy_intp. Pure C, no GIL. */
static npy_intp
probe_csr_hash_table(const CSRHashTable *table, const npy_int64 *left, npy_intp left_count,
                     npy_intp *slots, int keep_unmatched, int *has_unmatched)
{
    const HashGeometry geometry = table->geometry;
    const CSRHashEntry *entries = table->entries;
    npy_intp total = 0;
    for (npy_intp i = 0; i < left_count; i++) {
        npy_int64 key = left[i];
        npy_intp slot = HASH_SLOT(key, geometry.shift);
        while (entries[slot].count > 0 && entries[slot].key != key)
            slot = (slot + 1) & geometry.mask;
        if (entries[slot].count > 0) {
            slots[i] = slot;
            /* a many-to-many join can exceed npy_intp (32 bits on win32) */
            if (entries[slot].count > NPY_MAX_INTP - total) return -1;
            total += entries[slot].count;
        } else {
            slots[i] = -1;
            *has_unmatched = 1;
            if (keep_unmatched) {
                if (total == NPY_MAX_INTP) return -1;
                total++;
            }
        }
    }
    return total;
}


/* Emit pass: write (left, right) index pairs; right = -1 for an unmatched
 * left row when keep_unmatched is set. Pure C, no GIL. */
static void
emit_csr_pairs(const CSRHashTable *table, const npy_intp *slots, npy_intp left_count,
               int keep_unmatched, npy_intp *left_indices, npy_intp *right_indices)
{
    const CSRHashEntry *entries = table->entries;
    const npy_intp *rows = table->rows;
    npy_intp position = 0;
    for (npy_intp i = 0; i < left_count; i++) {
        npy_intp slot = slots[i];
        if (slot >= 0) {
            npy_intp offset = entries[slot].offset;
            npy_intp count = entries[slot].count;
            for (npy_intp j = 0; j < count; j++) {
                left_indices[position] = i;
                right_indices[position] = rows[offset + j];
                position++;
            }
        } else if (keep_unmatched) {
            left_indices[position] = i;
            right_indices[position] = -1;
            position++;
        }
    }
}


/* Shared body of inner_join and left_join. */
static PyObject *
hash_join(PyObject *args, int keep_unmatched)
{
    PyObject *left_obj, *right_obj;
    if (!PyArg_ParseTuple(args, "OO", &left_obj, &right_obj))
        return NULL;

    PyArrayObject *left_array = as_contiguous_vector(left_obj, NPY_INT64);
    PyArrayObject *right_array = left_array ? as_contiguous_vector(right_obj, NPY_INT64) : NULL;
    if (!left_array || !right_array) {
        Py_XDECREF(left_array); Py_XDECREF(right_array);
        return NULL;
    }

    npy_intp left_count = PyArray_SIZE(left_array);
    npy_intp right_count = PyArray_SIZE(right_array);
    const npy_int64 *left = (const npy_int64 *)PyArray_DATA(left_array);
    const npy_int64 *right = (const npy_int64 *)PyArray_DATA(right_array);

    CSRHashTable table = {{0, 0, 0}, NULL, NULL};
    npy_intp *slots = NULL;
    PyArrayObject *left_indices_array = NULL, *right_indices_array = NULL;
    PyObject *result = NULL;
    int has_unmatched = 0;

    if (build_csr_hash_table(right, right_count, &table) < 0) goto cleanup;
    slots = (npy_intp *)malloc_nonzero(left_count * sizeof(npy_intp));
    if (!slots) { PyErr_NoMemory(); goto cleanup; }

    npy_intp total;
    Py_BEGIN_ALLOW_THREADS
    total = probe_csr_hash_table(&table, left, left_count, slots, keep_unmatched, &has_unmatched);
    Py_END_ALLOW_THREADS
    if (total < 0) {
        PyErr_SetString(PyExc_MemoryError, "join result has more rows than an array can index");
        goto cleanup;
    }

    npy_intp dims[1] = {total};
    left_indices_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    right_indices_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    if (!left_indices_array || !right_indices_array) goto cleanup;

    npy_intp *left_indices = (npy_intp *)PyArray_DATA(left_indices_array);
    npy_intp *right_indices = (npy_intp *)PyArray_DATA(right_indices_array);
    Py_BEGIN_ALLOW_THREADS
    emit_csr_pairs(&table, slots, left_count, keep_unmatched, left_indices, right_indices);
    Py_END_ALLOW_THREADS

    if (keep_unmatched)
        result = PyTuple_Pack(3, (PyObject *)left_indices_array, (PyObject *)right_indices_array,
                              has_unmatched ? Py_True : Py_False);
    else
        result = PyTuple_Pack(2, (PyObject *)left_indices_array, (PyObject *)right_indices_array);

cleanup:
    free(slots);
    free_csr_hash_table(&table);
    Py_XDECREF(left_indices_array);
    Py_XDECREF(right_indices_array);
    Py_DECREF(left_array);
    Py_DECREF(right_array);
    return result;
}


static PyObject *
accel_inner_join(PyObject *self, PyObject *args)
{
    return hash_join(args, 0);
}


static PyObject *
accel_left_join(PyObject *self, PyObject *args)
{
    return hash_join(args, 1);
}


/* ================================================================
 * Module definition
 * ================================================================ */

static PyMethodDef AccelMethods[] = {
    {"groupby_sum", accel_groupby_sum, METH_VARARGS,
     "Grouped float sum: groupby_sum(labels, data, group_count, [skip_nan])"},
    {"groupby_sum_int64", accel_groupby_sum_int64, METH_VARARGS,
     "Grouped exact int64 sum, wrapping like numpy: groupby_sum_int64(labels, data, group_count)"},
    {"groupby_mean", accel_groupby_mean, METH_VARARGS,
     "Grouped mean: groupby_mean(labels, data, group_count, [skip_nan])"},
    {"groupby_var", accel_groupby_var, METH_VARARGS,
     "Grouped population variance (two-pass): groupby_var(labels, data, group_count, [skip_nan])"},
    {"groupby_min", accel_groupby_min, METH_VARARGS,
     "Grouped min, NaN as numpy: groupby_min(labels, data, group_count, [skip_nan])"},
    {"groupby_max", accel_groupby_max, METH_VARARGS,
     "Grouped max, NaN as numpy: groupby_max(labels, data, group_count, [skip_nan])"},
    {"groupby_min_int64", accel_groupby_min_int64, METH_VARARGS,
     "Grouped exact integer min: groupby_min_int64(labels, data, group_count, [unsigned_values])"},
    {"groupby_max_int64", accel_groupby_max_int64, METH_VARARGS,
     "Grouped exact integer max: groupby_max_int64(labels, data, group_count, [unsigned_values])"},
    {"composite_key", accel_composite_key, METH_VARARGS,
     "Positional encoding: composite_key(arrays, cardinalities) -> int64 key"},
    {"encode_strings", accel_encode_strings, METH_VARARGS,
     "O(n) hash-based object encoding: encode_strings(array) -> (codes, unique_count)"},
    {"encode_int64", accel_encode_int64, METH_VARARGS,
     "O(n) hash-based int64 encoding: encode_int64(keys) -> (codes, unique_count)"},
    {"group_indices", accel_group_indices, METH_VARARGS,
     "O(n) hash-based grouping: group_indices(key) -> (first_seen, group_rows, group_count)"},
    {"inner_join", accel_inner_join, METH_VARARGS,
     "Hash inner join: inner_join(left_key, right_key) -> (left_indices, right_indices)"},
    {"left_join", accel_left_join, METH_VARARGS,
     "Hash left join: left_join(left_key, right_key) -> (left_indices, right_indices, has_unmatched)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef accelmodule = {
    PyModuleDef_HEAD_INIT,
    "_accel",
    "C acceleration for tafra hot paths (v4: Fibonacci hashing, CSR joins, GIL released)",
    -1,
    AccelMethods
};

PyMODINIT_FUNC
PyInit__accel(void)
{
    import_array();
    PyObject *module = PyModule_Create(&accelmodule);
    if (!module) return NULL;
#ifdef Py_GIL_DISABLED
    /* No module-level mutable state: every buffer is allocated per call,
     * and each input key is read once, so a concurrent writer cannot
     * corrupt a table. */
    if (PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED) < 0) {
        Py_DECREF(module);
        return NULL;
    }
#endif
    return module;
}
