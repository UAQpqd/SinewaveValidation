//
// Created by dev on 4/9/2018.
//

#include "ProposedMethodology.hpp"

ProposedMethodology::ProposedMethodology(
        const std::vector<float> *t_signalData,
        const unsigned int t_sps,
        const float t_epsilon,
        const float omega0star,
        const float freqSearchAbsTol,
        const std::vector<float> t_harmonicFactors) :
        signalData(t_signalData), sps(t_sps),
        epsilon(t_epsilon) {
    harmonicFactors = std::vector<float>(
                t_harmonicFactors.begin(),
                t_harmonicFactors.end());

}
