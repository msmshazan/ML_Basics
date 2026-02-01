typedef double f64;
typedef float f32;
typedef int i32;
typedef long i64;
typedef long long i128;

typedef struct {
    f64 linear_weight;
    f64 quadratic_weight;
    f64 bias;
} LinearModel;

typedef struct {
    f64* x_data;
    f64* y_data;
    i32 size;
} Dataset;

typedef struct {
    f64 decay;
    f64 initial_learning_rate;
    f64 learning_rate;
    i32 epochs;
} TrainingConfig;

typedef struct {
    LinearModel model;
    Dataset dataset;
    TrainingConfig config;
} Trainer;

typedef struct {
    f64 prediction;
    f64 loss;
} TrainingResult;

typedef struct {
    int rows;
    int cols;
    double* data;
} Matrix;

typedef struct {
    int number_of_dimensions;
    int* size_of_each_dimension;
    int* strides;
    double* data;
} Tensor;

int main();

#define matrix_index(m, r, c) ((m).data[(r) * (m).cols + (c)])

int tensor_index(const Tensor* t, const int* indices) {
    int idx = 0;
    for (int i = 0; i < t->number_of_dimensions; i++) {
        idx += indices[i] * t->strides[i];
    }
    return idx;
}

void linear_regression_linear_example();
void linear_regression_quadratic_example();
