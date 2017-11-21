#ifndef RECOMMENDER_MODEL_H
#define RECOMMENDER_MODEL_H

#include "include.h"

namespace Impulse {

    namespace Recommender {

        class Model {
        protected:
            Impulse::Dataset::Dataset &dataset;
            T_Size numFeatures;
            Math::T_Matrix datasetMatrix;
            Math::T_Matrix x;
            Math::T_Matrix y;
            Math::T_Matrix theta;
            Math::T_Vector means;
            Math::T_Matrix predictions;

            void initialize();

        public:
            Model(Impulse::Dataset::Dataset &dataset, T_Size numFeatures);

            void calculatePredictions();

            Math::T_Matrix getPredictions();

            Math::T_Matrix getTheta();

            Math::T_Matrix getX();

            Math::T_Matrix getY();

            Math::T_Matrix getMeans();

            void setX(Math::T_Matrix x);

            void setTheta(Math::T_Matrix theta);

            double getError();

            double predict(T_Size itemId, T_Size categoryId);
        };
    }
}

#endif //RECOMMENDER_MODEL_H
