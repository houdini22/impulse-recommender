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
            Math::T_Matrix y = this->model.getY();
            double error = this->model.getError();

            if (this->verbose) {
                std::cout << "Starting training with error: [" << error << "], [" << this->numOfIterations
                          << "] iterations." << std::endl;
            }

            while (step < this->numOfIterations) {
                this->model.calculatePredictions();
                Math::T_Matrix predictions = this->model.getPredictions();
                Math::T_Matrix theta = this->model.getTheta();
                Math::T_Matrix x = this->model.getX();

                Math::T_Matrix newX(x.rows(), x.cols());
                Math::T_Matrix newTheta(theta.rows(), theta.cols());

                for (T_Size i = 0; i < x.cols(); i++) {
                    //std::cout << "TEST: " << i << std::endl;
                    for (T_Size j = 0; j < x.rows(); j++) {
                        //std::cout << "TEST2: " << j << std::endl;
                        double gradientSum = 0.0;

                        for (T_Size l = 0; l < y.cols(); l++) {
                            //std::cout << "TEST3: " << l << std::endl;
                            if (!std::isnan(y(i, l))) {
                                //std::cout << "RESULT2:" << (y(i, l)) << std::endl;
                                //std::cout << predictions << std::endl;
                                //return;
                                //std::cout << "THETA:" << theta(j, l) << std::endl;
                                gradientSum += ((predictions(i, l) - y(i, l)) * theta(j, l));

                                //std::cout << theta(j, l) << std::endl;
                            }
                        }

                        // std::cout << "TMP:" << gradientSum << std::endl;

                        newX(j, i) = x(j, i) - (this->learningRate * gradientSum);
                    }
                }
                /*std::cout << "NEW X: " << std::endl << newX << std::endl;
                return;*/

                for (T_Size i = 0; i < theta.cols(); i++) {
                    //std::cout << "TEST: " << i << std::endl;
                    for (T_Size j = 0; j < theta.rows(); j++) {
                        //std::cout << "TEST2: " << j << std::endl;
                        double gradientSum = 0.0;

                        for (T_Size l = 0; l < y.rows(); l++) {
                            if (!std::isnan(y(l, i))) {
                                //std::cout << "TEST3: " << l << std::endl;
                                //std::cout << "PREDICTIONS: " << (predictions(j, i) - y(l, i)) << std::endl;
                                //std::cout << "Y: " << y(l, i) << std::endl;
                                //std::cout << "NEW X:" << newX(j, l) << std::endl;
                                gradientSum += (predictions(l, i) - y(l, i)) * newX(j, l);
                                //std::cout << newX(j, l) << std::endl;
                            }
                        }

                        newTheta(j, i) = theta(j, i) - (this->learningRate * gradientSum);
                    }
                }

                this->model.setX(newX);
                this->model.setTheta(newTheta);

                error = this->model.getError();

                if ((step + 1) % this->verboseStep == 0) {
                    std::cout << "Step: [" << (step + 1) << "] with error:[" << error << "]." << std::endl;
                }

                step++;
            }

            if (this->verbose) {
                std::cout << "Training ended with error: [" << error << "]." << std::endl;
            }
        }
    }
}