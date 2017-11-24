#ifndef RECOMMENDER_SMARTTRAINER_H
#define RECOMMENDER_SMARTTRAINER_H

#include "include.h"

namespace Impulse {

    namespace Recommender {

        class SmartTrainer {
        protected:
            Model &model;
            double learningRate = 0.01;
            T_Size numOfIterations = 2000;
            bool verbose = true;
            T_Size verboseStep = 100;
        public:
            explicit SmartTrainer(Model &model);

            void setLearningRate(double value);

            void setNumOfIterations(T_Size value);

            void setVerbose(bool value);

            void setVerboseStep(T_Size value);

            void train();
        };
    }
}

#endif //RECOMMENDER_SMARTTRAINER_H
