typedef double f64;
typedef float f32;
typedef int i32;

typedef struct {
    f64 weight;
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

void linear_regression_example();
