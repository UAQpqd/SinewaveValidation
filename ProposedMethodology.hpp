//
// Created by dev on 4/9/2018.
//

#ifndef SINEWAVEVALIDATION_PROPOSEDMETHODOLOGY_HPP
#define SINEWAVEVALIDATION_PROPOSEDMETHODOLOGY_HPP

#include <vector>

class Filter {
public:
    enum class FilterType { LOWPASS, HIGHPASS };
    Filter(FilterType t_type, float t_omega) :
        type(t_type), omega(t_omega) {};
    const FilterType type;
    const float omega;
};

class ProposedMethodology {
    struct DEBlock {
        std::vector<float> *inSignal;
        float outEstPhi;
        float outEstOmega;
    };
    struct RMSBlock {
        float inPhi;
        float inOmega;
        std::vector<float> outRMS;
    };
    struct MeanBlock {
        std::vector<float> *inData;
        float outMean;
    };
public:
    ProposedMethodology(
            const std::vector<float> *t_signalData,
            const unsigned int t_sps,
            const float t_epsilon,
            const float omega0star,
            const float freqSearchAbsTol,
            const std::vector<float> t_harmonicFactors
    );
private:
    const std::vector<float> *signalData;
    unsigned int sps;
    const float epsilon;
    Filter *fp1,*fp2;
    std::vector<float> harmonicFactors;
    std::vector<Filter> harmonicFilters;
};


#endif //SINEWAVEVALIDATION_PROPOSEDMETHODOLOGY_HPP
