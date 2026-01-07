#include "main.h"
#include <stdio.h>

int main()
{
    linear_regression_example();

	return 0;
}

void linear_regression_example()
{
    enum { N = 5 };

    // Training data
    f64 x[N] = { 1, 2, 3, 4, 5 };
    f64 y[N] = { 2, 4, 6, 8, 10 }; // y = 2x

    LinearModel model = { .bias = 0 , .weight = 0 };
    TrainingConfig config = { .epochs = 100000 , .initial_learning_rate = 0.1, .decay = 0.01 };
    Dataset dataset;
    dataset.x_data = x;
    dataset.y_data = y;
    dataset.size = N;

    for (i32 epoch = 0; epoch < config.epochs; epoch++) {
        config.learning_rate = config.initial_learning_rate / (1.0 + config.decay * epoch);  // Adaptive learning rate​ ​
        double delta_weight = 0.0;
        double delta_bais = 0.0;
        double loss = 0.0;

        for (i32 i = 0; i < N; i++) {
            double y_prediction = model.weight * dataset.x_data[i] + model.bias;
            double error = y_prediction - dataset.y_data[i];

            loss += error * error;
            delta_weight += error * dataset.x_data[i];
            delta_bais += error;
        }

        // Average gradients
        delta_weight = (2.0 / N) * delta_weight;
        delta_bais = (2.0 / N) * delta_bais;

        // Update parameters
        model.weight -= config.learning_rate * delta_weight;
        model.bias -= config.learning_rate * delta_bais;

        loss /= N;

        if (epoch % 100 == 0) {
            printf("Epoch %d | Loss: %f | w: %f | b: %f| lr: %f\n",
                epoch, loss, model.weight, model.bias, config.learning_rate);
        }
        if (loss < 10e-20) {
            printf("Final Epoch %d | Loss: %f | w: %f | b: %f | lr: %f\n",
                epoch, loss, model.weight, model.bias, config.learning_rate);
            break;
        }
    }

    printf("\nFinal Model: y = %.3fx + %.3f\n", model.weight, model.bias);

    // Test
    f64 test_x = 12.0;
    printf("Prediction for x=12 y= %.2f\n", model.weight * test_x + model.bias);
    test_x = -4.0;
    printf("Prediction for x=-4 y= %.2f\n", model.weight * test_x + model.bias);
}
