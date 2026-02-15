#include "main.h"
#include <stdio.h>

int main()
{
    linear_regression_linear_example();
    linear_regression_quadratic_example();
	return 0;
}

void linear_regression_linear_example()
{
	// Simple Linear Regression using Gradient Descent
	// Feature size = 1
	// Sample size = 5
	// Input Dimension: 1D (x)
	// Output Dimension: 1D (y)
	// Target: y = 2x 
    
    enum { sample_size = 5 };

    // Training data
    f64 x[sample_size] = { 1, 2, 3, 4, 5 };
    f64 y[sample_size] = { 2, 4, 6, 8, 10 }; // y = 2x

    LinearModel model = { .bias = 0 , .linear_weight = 0 };
    TrainingConfig config = { .epochs = 100000 , .initial_learning_rate = 0.1, .decay = 0.01 };
    Dataset dataset;
    dataset.x_data = x;
    dataset.y_data = y;
    dataset.size = sample_size;

    for (i32 epoch = 0; epoch < config.epochs; epoch++) {
        config.learning_rate = config.initial_learning_rate / (1.0 + config.decay * epoch);  // Adaptive learning rate​ ​
        f64 delta_weight = 0.0;
        f64 delta_bais = 0.0;
        f64 loss = 0.0;

        for (i32 i = 0; i < dataset.size; i++) {
            f64 y_prediction = model.linear_weight * dataset.x_data[i] + model.bias;
            f64 error = y_prediction - dataset.y_data[i];

            loss += error * error;
            delta_weight += error * dataset.x_data[i];
            delta_bais += error;
        }

        // Average gradients
        delta_weight = (2.0 / dataset.size) * delta_weight;
        delta_bais = (2.0 / dataset.size) * delta_bais;

        // Update parameters
        model.linear_weight -= config.learning_rate * delta_weight;
        model.bias -= config.learning_rate * delta_bais;

        loss /= dataset.size;

        if (epoch % 100 == 0) {
            printf("Epoch %d | Loss: %f | w: %f | b: %f| lr: %f\n",
                epoch, loss, model.linear_weight, model.bias, config.learning_rate);
        }
        if (loss < 10e-20) {
            printf("Final Epoch %d | Loss: %f | w: %f | b: %f | lr: %f\n",
                epoch, loss, model.linear_weight, model.bias, config.learning_rate);
            break;
        }
    }

    printf("\nFinal Model: y = %.3fx + %.3f\n", model.linear_weight, model.bias);

    // Test
    f64 test_x = 12.0;
    printf("Prediction for x=12 y= %.2f\n", model.linear_weight* test_x + model.bias);
    test_x = -4.0;
    printf("Prediction for x=-4 y= %.2f\n", model.linear_weight* test_x + model.bias);
}



void linear_regression_quadratic_example()
{
    // Simple Linear Regression using Gradient Descent
    // Feature size = 2
    // Sample size = 5
    // Input Dimension: 1D (x)
    // Output Dimension: 1D (y)
	// Target: y = x^2 

    enum { sample_size = 5 };

    // Training data
    f64 x[sample_size] = { 1, 2, 3, 4, 5 };
    f64 y[sample_size] = { 1, 4, 9, 16, 25 }; // y = x^2 

    LinearModel model = { .bias = 0 , .linear_weight = 0, .quadratic_weight = 0 };
    Dataset dataset;
    dataset.x_data = x;
    dataset.y_data = y;
    dataset.size = sample_size;

    double max_x2 = 0.0;
    for (i32 i = 0; i < dataset.size; i++) {
        double x2 = dataset.x_data[i] * dataset.x_data[i];
        if (x2 > max_x2) max_x2 = x2;
    }

    TrainingConfig config = { .epochs = 500000 , .initial_learning_rate = 1.0 / (max_x2 + 1.0) };
    config.decay = config.initial_learning_rate * 0.01;

    for (i32 epoch = 0; epoch < config.epochs; epoch++) {
        config.learning_rate = config.initial_learning_rate / (1.0 + config.decay * epoch);  // Adaptive learning rate​ ​
        f64 delta_linear_weight = 0.0;
        f64 delta_quadratic_weight = 0.0;
        f64 delta_bais = 0.0;
        f64 loss = 0.0;

        for (i32 i = 0; i < dataset.size; i++) {
            f64 x = dataset.x_data[i];
            f64 y = dataset.y_data[i];

            f64 y_prediction = 
                (model.quadratic_weight * x * x) + 
                (model.linear_weight * x) + 
                model.bias;

            f64 error = y_prediction - y;

            loss += error * error;

            delta_quadratic_weight += error * x * x;
            delta_linear_weight += error * x;
            delta_bais += error;
        }

        // Mean Squared Error gradients
        f64 scale = (2.0 / dataset.size);
        delta_quadratic_weight *= scale;
        delta_linear_weight *= scale;
        delta_bais *= scale;

        // Update parameters
        model.quadratic_weight -= config.learning_rate * delta_quadratic_weight;
        model.linear_weight -= config.learning_rate * delta_linear_weight;
        model.bias -= config.learning_rate * delta_bais;

        loss /= dataset.size;

        if (epoch % 100 == 0) {
            printf("Epoch %d | Loss: %f | qw: %f | lw: %f | b: %f| lr: %f\n",
                epoch, loss, model.quadratic_weight, model.linear_weight, model.bias, config.learning_rate);
        }
        if (loss < 10e-20) {
            printf("Final Epoch %d | Loss: %f | qw: %f | lw: %f | b: %f | lr: %f\n",
                epoch, loss, model.quadratic_weight, model.linear_weight, model.bias, config.learning_rate);
            break;
        }
    }

    printf("\nFinal Model: y = %.3fx^2 + %.3fx + %.3f\n", model.quadratic_weight , model.linear_weight, model.bias);

    // Test
    f64 test_x = 12.0;
    printf("Prediction for x=12 y= %.2f\n", model.quadratic_weight* test_x* test_x + model.linear_weight * test_x + model.bias);
    test_x = -4.0;
    printf("Prediction for x=-4 y= %.2f\n", model.quadratic_weight* test_x* test_x + model.linear_weight * test_x + model.bias);
}
