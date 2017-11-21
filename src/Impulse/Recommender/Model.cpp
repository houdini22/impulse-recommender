#include <utility>

#include "include.h"

namespace Impulse {

    namespace Recommender {

        Model::Model(Impulse::Dataset::Dataset &dataset, T_Size numFeatures) : dataset(dataset) {
            this->numberOfFeatures = numFeatures;
            this->datasetMatrix = this->dataset.exportToEigen();

            this->initialize();
        }

        void Model::initialize() {
            T_Size uniqueItemsCount = this->datasetMatrix.col(0).maxCoeff() + 1;
            T_Size uniqueCategoriesCount = this->datasetMatrix.col(1).maxCoeff() + 1;

            this->means.resize(uniqueItemsCount);
            this->means.setZero();

            Math::T_Vector meanCounts(uniqueItemsCount);
            meanCounts.setZero();

            this->y.resize(uniqueItemsCount, uniqueCategoriesCount);
            this->y.setZero();

            for (T_Size i = 0; i < this->datasetMatrix.rows(); i++) {
                this->y(this->datasetMatrix(i, 0), this->datasetMatrix(i, 1)) = this->datasetMatrix(i, 2);
                if (!std::isnan(this->datasetMatrix(i, 2))) {
                    this->means(this->datasetMatrix(i, 0)) += this->datasetMatrix(i, 2);
                    meanCounts(this->datasetMatrix(i, 0)) += 1;
                }
            }

            this->means = this->means.array() / meanCounts.array();
            this->y = this->y.array() - this->means.replicate(1, uniqueCategoriesCount).array();

            this->x.resize(this->numberOfFeatures, uniqueItemsCount);
            this->x.setRandom();

            this->theta.resize(this->numberOfFeatures, uniqueCategoriesCount);
            this->theta.setRandom();

            this->predictions.resize(uniqueItemsCount, uniqueCategoriesCount);
            this->predictions.setZero();
        }

        void Model::calculatePredictions() {
            this->predictions = (this->theta.transpose() * this->x).transpose();
        }

        Math::T_Matrix &Model::getPredictions() {
            return this->predictions;
        }

        Math::T_Matrix &Model::getTheta() {
            return this->theta;
        }

        Math::T_Matrix &Model::getX() {
            return this->x;
        }

        Math::T_Matrix &Model::getY() {
            return this->y;
        }

        Math::T_Vector &Model::getMeans() {
            return this->means;
        }

        void Model::setX(Math::T_Matrix x) {
            this->x = std::move(x);
        }

        void Model::setTheta(Math::T_Matrix theta) {
            this->theta = std::move(theta);
        }

        double Model::getError() {
            this->calculatePredictions();

            double error = (this->predictions.array() - this->y.array()).unaryExpr([](const double x) {
                if (std::isnan(x)) {
                    return 0.0;
                }
                return pow(x, 2.0);
            }).sum();

            return error / 2;
        }

        double Model::predict(T_Size itemId, T_Size categoryId) {
            return this->predictions(itemId, categoryId) + this->means(itemId);
        }

        double Model::predict(T_Size itemId) {
            return this->means(itemId);
        }

        T_Size Model::getNumberOfFeatures() {
            return this->numberOfFeatures;
        }
    }
}
