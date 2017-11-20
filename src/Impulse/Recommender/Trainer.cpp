#include "include.h"

namespace Impulse {

    namespace Recommender {

        Trainer::Trainer(Model &model) : model(model) {
        }

        void Trainer::setLearningRate(double value) {
            this->learningRate = value;
        }

        void Trainer::setNumOfIterations(T_Size value) {
            this->numOfIterations = value;
        }

        void Trainer::setVerbose(bool value) {
            this->verbose = value;
        }

        void Trainer::setVerboseStep(T_Size value) {
            this->verboseStep = value;
        }

        void Trainer::train() {
            T_Size step = 0;
            Model model = this->model;
            Math::T_Matrix y = model.getY();

            if (this->verbose) {
                std::cout << "Starting training with [" << this->numOfIterations << "] iterations." << std::endl;
            }

            while (step < this->numOfIterations) {
                model.calculatePredictions();
                Math::T_Matrix predictions = model.getPredictions();
                Math::T_Matrix theta = model.getTheta();
                Math::T_Matrix x = model.getX();

                Math::T_Matrix newX(x.rows(), x.cols());
                Math::T_Matrix newTheta(theta.rows(), theta.cols());

                for (T_Size i = 0; i < x.cols(); i++) {
                    for (T_Size j = 0; j < x.rows(); j++) {
                        double gradientSum = 0.0;
                        T_Size k = 0;

                        for (T_Size l = 0; l < y.cols(); l++) {
                            if (!std::isnan(y(k, i))) {
                                gradientSum += (predictions(k, i) - y(k, i)) * theta(k, j);
                            }
                            k++;
                        }

                        newX(j, i) = x(j, i) - (this->learningRate * gradientSum);
                    }
                }

                for (T_Size i = 0; i < theta.cols(); i++) {
                    for (T_Size j = 0; j < theta.rows(); j++) {
                        double gradientSum = 0.0;
                        T_Size k = 0;

                        for (T_Size l = 0; l < y.cols(); l++) {
                            if (!std::isnan(y(k, i))) {
                                gradientSum += (predictions(k, i) - y(k, i)) * newX(k, j);
                            }
                            k++;
                        }

                        newTheta(j, i) = theta(j, i) - (this->learningRate * gradientSum);
                    }
                }

                model.setX(newX);
                model.setTheta(newTheta);

                double error = model.getError();

                if ((step + 1) % this->verboseStep == 0) {
                    std::cout << "Step: [" << (step + 1) << "] with error:[" << error << "]." << std::endl;
                }

                step++;
            }

            if (this->verbose) {
                std::cout << "Training ended." << std::endl;
            }
        }
    }
}