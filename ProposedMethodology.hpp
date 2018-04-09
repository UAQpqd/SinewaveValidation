//
// Created by dev on 4/9/2018.
//

#ifndef SINEWAVEVALIDATION_PROPOSEDMETHODOLOGY_HPP
#define SINEWAVEVALIDATION_PROPOSEDMETHODOLOGY_HPP

#include <vector>
#include <boost/compute.hpp>

#include "minusDarwin/Solver.hpp"
#include "DspFilters/ChebyshevII.h"
#include "DspFilters/ChebyshevI.h"
#include "DspFilters/Butterworth.h"
#include "DspFilters/Common.h"
#include "DspFilters/Filter.h"

void writeCsvFile(const std::vector<float> *data, unsigned int sps, std::string filename);



namespace bc = boost::compute;

const MinusDarwin::SolverParameterSet config = {
        2,
        200,
        20,
        MinusDarwin::GoalFunction::EpsilonReached,
        MinusDarwin::CrossoverMode::Best,
        1,
        0.005f,
        1.0f,
        0.25f,
        true
};

const char sinewaveFitSourceParallel[] =
        BOOST_COMPUTE_STRINGIZE_SOURCE(
                __kernel void calculateScoresOfPopulation(
                        __global const float *signalData,
                __global const float2 *population,
                __global float *scores,
                const unsigned int populationSize,
                const unsigned int signalDataSize,
                const float sumOfSquares,
                const float omegaMin,
                const float omegaMax,
                const float phiMax,
                const unsigned int sps,
                const float a)
        {
                const uint aId = get_global_id(0);
                const float2 agent = population[aId];
                float error = 0.0f;
                for (size_t p = 0; p < signalDataSize; p++) {
                float t = (float)p/(float)sps;
                float realOmega = omegaMin+agent.x*(omegaMax-omegaMin);
                float realPhi = agent.y*phiMax;
                float estimated =
                a*sin(realOmega*t+realPhi);
                error += (estimated-signalData[p])*(estimated-signalData[p]);
        }
                scores[aId] = error/sumOfSquares;
        });

class DEBlock {
public:
    DEBlock(std::vector<float> *t_inSignal,
            float t_inOmegaMin,
            float t_inOmegaMax,
            bc::command_queue *t_queue,
            bc::kernel *t_kernel) :
            inSignal(t_inSignal),
            inOmegaMin(t_inOmegaMin),
            inOmegaMax(t_inOmegaMax),
            outEstPhi(0.0f), outEstOmega(0.0f),
            queue(t_queue), kernel(t_kernel) {};
    void run(unsigned int sps,
             bc::context &ctx);

    std::vector<float> *inSignal;
    float inOmegaMin,inOmegaMax;
    float outEstPhi;
    float outEstOmega;
    bc::command_queue *queue;
    bc::kernel *kernel;
};

class ProposedMethodology {
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
            const unsigned int t_sps,
            const float t_epsilon,
            const float t_freq0star,
            const float t_freqSearchAbsTol,
            const std::vector<float> t_harmonicFactors
    );
    void run(const std::vector<float> *rp);
private:
    unsigned int sps;
    const float epsilon;
    const float freq0star;
    const float freqSearchAbsTol;
    Dsp::SimpleFilter<Dsp::ChebyshevII::LowPass<3>,1> fp1;
    Dsp::SimpleFilter<Dsp::ChebyshevII::HighPass<3>,1> fp2;
    DEBlock *flickerDEBlock, *fundamentalDEBlock;
    RMSBlock flickerRmsBlock, fundamentalRmsBlock;
    std::vector<float> harmonicFactors;
    std::vector<Dsp::SimpleFilter<Dsp::ChebyshevII::BandPass<3>,1> > harmonicFilters;
    std::vector<RMSBlock> harmonicsRMSBlocks;
};


#endif //SINEWAVEVALIDATION_PROPOSEDMETHODOLOGY_HPP
