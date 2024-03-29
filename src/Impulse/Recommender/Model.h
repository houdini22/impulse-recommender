#ifndef RECOMMENDER_MODEL_H
#define RECOMMENDER_MODEL_H

#include "include.h"

namespace Impulse {

    namespace Recommender {

        class Model {
        protected:
            Impulse::Dataset::Dataset &dataset;
            T_Size numberOfFeatures;
            Math::T_Matrix datasetMatrix;
            Math::T_Matrix x;
            Math::T_Matrix y;
            Math::T_Matrix theta;
            Math::T_Vector means;
            Math::T_Matrix predictions;

            void initialize();

        public:
            Model(Impulse::Dataset::Dataset &dataset, T_Size numFeatures, bool initialize = true);

            Model(T_Size numFeatures);

            void calculatePredictions();

            Math::T_Matrix &getPredictions();

            Math::T_Matrix &getTheta();

            Math::T_Matrix &getX();

            Math::T_Matrix &getY();

            Math::T_Vector &getMeans();

            void setPredictions(Math::T_Matrix predictions);

            void setTheta(Math::T_Matrix theta);

            void setX(Math::T_Matrix x);

            void setY(Math::T_Matrix y);

            void setMeans(Math::T_Matrix means);

            double getError();

            double predict(T_Size itemId, T_Size categoryId);

            double predict(T_Size itemId);

            T_Size getNumberOfFeatures();
        };
    }
}

#endif //RECOMMENDER_MODEL_H
