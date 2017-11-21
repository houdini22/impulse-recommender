#ifndef IMPULSE_RECOMMENDER_SERIALIZER_H
#define IMPULSE_RECOMMENDER_SERIALIZER_H

#include "include.h"

namespace Impulse {

    namespace Recommender {

        class Serializer {
        protected:
            Model model;

            Math::T_RawVector getRolled(Math::T_Matrix &var);

        public:
            explicit Serializer(Model &model);

            void toJSON(T_String path);
        };
    }
}

#endif //IMPULSE_RECOMMENDER_SERIALIZER_H
