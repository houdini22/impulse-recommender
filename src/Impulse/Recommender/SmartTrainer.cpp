#include "include.h"

namespace Impulse {

    namespace Recommender {

        SmartTrainer::SmartTrainer(Model &model) : model(model) {
        }

        void SmartTrainer::setLearningRate(double value) {
            learningRate = value;
        }

        void SmartTrainer::setNumOfIterations(T_Size value) {
            this->numOfIterations = value;
        }

        void SmartTrainer::setVerbose(bool value) {
            this->verbose = value;
        }

        void SmartTrainer::setVerboseStep(T_Size value) {
            this->verboseStep = value;
        }

        void SmartTrainer::train() {
            this->model.calculatePredictions();

            T_Size step = 0;
            Math::T_Matrix y = this->model.getY();
            double error = this->model.getError();
            double prevError = error;
            double learningRate = this->learningRate;

            if (this->verbose) {
                std::cout << "Starting training ERROR:[" << error << "], NUMBER OF ITERATIONS:["
                          << this->numOfIterations
                          << "]." << std::endl;
            }

            while (step < this->numOfIterations) {
                std::chrono::high_resolution_clock::time_point timeStart = std::chrono::high_resolution_clock::now();

                Math::T_Matrix predictions = this->model.getPredictions();
                Math::T_Matrix theta = this->model.getTheta();
                Math::T_Matrix x = this->model.getX();

                Math::T_Matrix newX(x.rows(), x.cols());
                Math::T_Matrix newTheta(theta.rows(), theta.cols());

                Math::T_Matrix difference = predictions.array() - y.array();

#pragma omp parallel for collapse(2)
                for (T_Size i = 0; i < x.cols(); i++) {
                    for (T_Size j = 0; j < x.rows(); j++) {
                        double gradientSum = 0.0;

#pragma omp parallel
                        for (T_Size l = 0; l < y.cols(); l++) {
                            if (!std::isnan(difference(i, l))) {
                                gradientSum += (difference(i, l) * theta(j, l));
                            }
                        }

                        newX(j, i) = x(j, i) - (learningRate * gradientSum);
                    }
                }

#pragma omp parallel for collapse(2)
                for (T_Size i = 0; i < theta.cols(); i++) {
                    for (T_Size j = 0; j < theta.rows(); j++) {
                        double gradientSum = 0.0;

#pragma omp parallel
                        for (T_Size l = 0; l < y.rows(); l++) {
                            if (!std::isnan(difference(l, i))) {
                                gradientSum += (difference(l, i)) * newX(j, l);
                            }
                        }

                        newTheta(j, i) = theta(j, i) - (learningRate * gradientSum);
                    }
                }

                this->model.setX(newX);
                this->model.setTheta(newTheta);
                error = this->model.getError();

                if (error > prevError) {
                    this->model.setX(x);
                    this->model.setTheta(theta);
                    error = this->model.getError();

                    if (learningRate > 0.005) {
                        learningRate -= 0.001;
                    } else {
                        learningRate *= 0.95;
                    }


                    if (this->verbose) {
                        std::cout << "Decreasing learning rate. STEP:[" << (step + 1) << "] ERROR:[" << error
                                  << "] LEARNING RATE:[" << learningRate << "]." << std::endl;
                    }
                } else {
                    step++;
                    prevError = error;

                    this->model.calculatePredictions();

                    std::chrono::high_resolution_clock::time_point timeEnd = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();

                    if (this->verbose) {
                        if ((step + 1) % this->verboseStep == 0) {
                            std::cout << "STEP:[" << (step + 1) << "] ERROR:[" << error << "] TIME:[" << duration
                                      << "ms]" << std::endl;
                        }
                    }
                }

            }

            if (this->verbose) {
                std::cout << "Training ended with ERROR:[" << error << "]." << std::endl;
            }
        }
    }
}