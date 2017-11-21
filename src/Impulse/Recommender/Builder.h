#ifndef RECOMMENDER_BUILDER_H
#define RECOMMENDER_BUILDER_H

#include "include.h"

namespace Impulse {

    namespace Recommender {

        Math::T_Matrix createMatrix(Math::T_RawVector &vec, T_Size cols, T_Size rows);

        class Builder {
        public:
            static Model buildFromJSON(T_String path);
        };
    }
}

#endif //RECOMMENDER_BUILDER_H
