#ifndef RECOMMENDER_TRAINER_H
#define RECOMMENDER_TRAINER_H

#include "include.h"

namespace Impulse {

    namespace Recommender {

        class Trainer {
        protected:
            Model &model;
            double learningRate = 0.01;
            T_Size numOfIterations = 2000;
            bool verbose = true;
            T_Size verboseStep = 100;
        public:
            explicit Trainer(Model &model);

            void setLearningRate(double value);

            void setNumOfIterations(T_Size value);

            void setVerbose(bool value);

            void setVerboseStep(T_Size value);

            void train();
        };
    }
}

#endif //RECOMMENDER_TRAINER_H
